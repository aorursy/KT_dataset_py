import sys, os

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

from sklearn import preprocessing

from sklearn import model_selection
DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression/"

K_FOLDS = 5

NUM_EPOCHS = 1000

TRAIN_BATCH_SIZE = 16

TEST_BATCH_SIZE = 4

QUANTILES = [0.1, 0.5, 0.9]

LEARNING_RATE = 4e-5

PRINT_EVERY = 50



SCALE_COLUMNS = ['Weeks', 'FVC', 'Percent', 'Age']

SEX_COLUMNS = ['Male', 'Female']

SMOKING_STATUS_COLUMNS = ['Currently smokes', 'Ex-smoker', 'Never smoked']

# the feature vector (passed into the model)

FV = SEX_COLUMNS + SMOKING_STATUS_COLUMNS + SCALE_COLUMNS



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# define splitter and min_max_scaler

kf = model_selection.KFold(K_FOLDS)

MIN_MAX_SCALER = preprocessing.MinMaxScaler()
# read train.csv and test.csv

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

train_df.drop_duplicates(keep=False, inplace=True, subset=['Patient', 'Weeks'])

test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test_df.head()
# fit the scaler and transform the data using the fit_transform function

train_df[SCALE_COLUMNS] = MIN_MAX_SCALER.fit_transform(train_df[SCALE_COLUMNS])
# specify the categorical columns and the categories incase any class is missing (in test)

# convert into one-hot-encoding

train_df['Sex'] = pd.Categorical(train_df['Sex'], categories=SEX_COLUMNS)

train_df['SmokingStatus'] = pd.Categorical(train_df['SmokingStatus'], categories=SMOKING_STATUS_COLUMNS)

train_df = train_df.join(pd.get_dummies(train_df['Sex']))

train_df = train_df.join(pd.get_dummies(train_df['SmokingStatus']))
train_df.head()
sub_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

# get the patient_id and the week from the Patient_Week column

sub_df['Patient'] = sub_df['Patient_Week'].apply(lambda x: x.split('_')[0])

sub_df['Weeks'] = sub_df['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub_df.head()
sub_df = sub_df.drop("FVC", axis=1).merge(test_df.drop('Weeks', axis=1), on='Patient')
# have to make it categorical coz sub's sex column has males only

sub_df['Sex'] = pd.Categorical(sub_df['Sex'], categories=SEX_COLUMNS)

sub_df['SmokingStatus'] = pd.Categorical(sub_df['SmokingStatus'], categories=SMOKING_STATUS_COLUMNS)

sub_df = sub_df.join(pd.get_dummies(sub_df['Sex']))

sub_df = sub_df.join(pd.get_dummies(sub_df['SmokingStatus']))
# use the global min_max_scaler with values from train.csv to scale the columns for the submission

sub_df[SCALE_COLUMNS] = MIN_MAX_SCALER.transform(sub_df[SCALE_COLUMNS])
sub_df.head()
class PulmonaryDataset(torch.utils.data.Dataset):

    def __init__(self, df, FV, test=False):

        self.df = df

        self.test = test

        self.FV = FV



    def __getitem__(self, idx):

        return {

            'features': torch.tensor(self.df[self.FV].iloc[idx].values),

            'target': torch.tensor(self.df['FVC'].iloc[idx])

        }



    def __len__(self):

        return len(self.df)
class PulmonaryModel(nn.Module):

    def __init__(self, in_features=9, out_quantiles=3):

        super(PulmonaryModel, self).__init__()

        self.fc1 = nn.Linear(in_features, 100)

        self.fc2 = nn.Linear(100, 100)

        self.fc3 = nn.Linear(100, out_quantiles)

    

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x
def quantile_loss(preds, target, quantiles):

    assert not target.requires_grad

    assert preds.size(0) == target.size(0)

    losses = []

    for i, q in enumerate(quantiles):

        errors = target - preds[:, i]

        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

    return loss
class AverageMeter:

    """

    Computes and stores the average and current value

    """

    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count
def train_one_epoch(model, train_data_loader, optimizer, train_loss):

    model.train()



    for i, data in enumerate(train_data_loader):

        features = data['features']

        targets = data['target']



        features = features.to(DEVICE).float()

        targets = targets.to(DEVICE).float()



        model.zero_grad()

        out = model(features)

        loss = quantile_loss(out, targets, QUANTILES)

        train_loss.update(loss, features.size(0))

        loss.backward()

        optimizer.step()
def eval_one_epoch(model, valid_data_loader, valid_loss, lr_scheduler):

    model.eval()



    with torch.no_grad():

        for i, data in enumerate(valid_data_loader):

            features = data['features']

            targets = data['target']



            features = features.to(DEVICE).float()

            targets = targets.to(DEVICE).float()

            

            out = model(features)

            loss = quantile_loss(out, targets, QUANTILES)

            valid_loss.update(loss, features.size(0))

    

    if lr_scheduler is not None:

        lr_scheduler.step(valid_loss.avg)
# REMOVE THE ONES FROM THE TRAIN_DF THAT ARE PRESENT IN TEST_DF AS WELL

TEST_PATIENTS = test_df['Patient'].unique().tolist()

valid_df = train_df[train_df['Patient'].isin(TEST_PATIENTS)]

train_df = train_df[~train_df['Patient'].isin(TEST_PATIENTS)]

TRAIN_PATIENTS = train_df['Patient'].unique().tolist()
for fold, (train_index, test_index) in enumerate(kf.split(TRAIN_PATIENTS)):

    model = PulmonaryModel(len(FV))

    model = model.to(DEVICE)



    df_train = train_df.iloc[train_index].reset_index(drop=True)

    df_valid = train_df.iloc[test_index].reset_index(drop=True)



    train_dataset = PulmonaryDataset(df_train, FV)

    valid_dataset = PulmonaryDataset(df_valid, FV)



    train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=TRAIN_BATCH_SIZE,

        shuffle=True,

        num_workers=4

    )



    valid_data_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=TEST_BATCH_SIZE,

        shuffle=False,

        num_workers=4

    )



    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.05, verbose=True)



    best_valid_loss = float('inf')

    

    train_loss = AverageMeter()

    valid_loss = AverageMeter()

    

    for epoch in range(NUM_EPOCHS):

        train_one_epoch(model, train_data_loader, optimizer, train_loss)

        eval_one_epoch(model, valid_data_loader, valid_loss, lr_scheduler)

        

        if epoch % PRINT_EVERY == 0:

            print(f"Fold {fold} Epoch {epoch}/{NUM_EPOCHS}, train_loss: {train_loss.avg}, val_loss: {valid_loss.avg}")



        if valid_loss.avg < best_valid_loss:

            best_valid_loss = valid_loss.avg

            torch.save({

                'model_state_dict': model.state_dict(),

                'optimizer_state_dict': optimizer.state_dict(),

            }, f"model_fold_{fold}.pt")

    print()
models = []

for fold in range(K_FOLDS):

    model = PulmonaryModel(len(FV))

    model = model.to(DEVICE)

    checkpoint = torch.load(f"model_fold_{fold}.pt")

    model.load_state_dict(checkpoint['model_state_dict'])

    models.append(model)
test_dataset = PulmonaryDataset(sub_df, FV)

test_data_loader = torch.utils.data.DataLoader(

    test_dataset,

    batch_size=TEST_BATCH_SIZE,

    shuffle=False,

    num_workers=4

)
avg_preds = np.zeros((len(test_dataset), len(QUANTILES)))

with torch.no_grad():

    for model in models:

        preds = []

        for j, test_data in enumerate(test_data_loader):

            features = test_data['features']

            targets = test_data['target']



            features = features.to(DEVICE).float()

            targets = targets.to(DEVICE).float()



            out = model(features)

            preds.append(out)

        preds = torch.cat(preds, dim=0).cpu().numpy()

        avg_preds += preds

avg_preds /= len(models)
# inverse the scaling operation for FVC

avg_preds -= MIN_MAX_SCALER.min_[SCALE_COLUMNS.index('FVC')]

avg_preds /= MIN_MAX_SCALER.scale_[SCALE_COLUMNS.index('FVC')]
avg_preds
sub_df['FVC'] = avg_preds[:, 1]

sub_df['Confidence'] = np.abs(avg_preds[:, 2] - avg_preds[:, 0])
sub_df.head(25)
sub_df[['Patient_Week', 'FVC', 'Confidence']].to_csv('submission.csv', index=False)