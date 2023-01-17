import os, sys

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

from sklearn import preprocessing
DATA_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression/"

MODEL_DIR = "/kaggle/input/osicqrmodel/"

QUANTILES = [0.2, 0.5, 0.8]

# columns to be scaled using min-max scaling

SCALE_COLUMNS = ['Weeks', 'FVC', 'Percent', 'Age']

SEX_COLUMNS = ['Male', 'Female']

SMOKING_STATUS_COLUMNS = ['Currently smokes', 'Ex-smoker', 'Never smoked']



# create the FV (feature vector) using the scaled columns + other columns

FV = SEX_COLUMNS + SMOKING_STATUS_COLUMNS + SCALE_COLUMNS

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# initialize the sklearn's min-max scaler

MIN_MAX_SCALER = preprocessing.MinMaxScaler()
# read the train_df and initialize the MIN_MAX_SCALER with train data

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
sub_df[SCALE_COLUMNS] = MIN_MAX_SCALER.transform(sub_df[SCALE_COLUMNS])
sub_df.head()
class PulmonaryDataset(Dataset):

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
test_dataset = PulmonaryDataset(sub_df, FV)



test_data_loader = DataLoader(

    test_dataset,

    batch_size=10,

    drop_last=False,

    num_workers=2

)
models = []

for fold in range(5):

    model = PulmonaryModel(len(FV))

    checkpoint = torch.load(os.path.join(MODEL_DIR, f"model_fold_{fold}.pt"))

    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(DEVICE)

    models.append(model)
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
avg_preds
# inverse the scaling operation for FVC

avg_preds -= MIN_MAX_SCALER.min_[SCALE_COLUMNS.index('FVC')]

avg_preds /= MIN_MAX_SCALER.scale_[SCALE_COLUMNS.index('FVC')]
avg_preds[:100]
sub_df['FVC'] = avg_preds[:, 1]

sub_df['Confidence'] = np.abs(avg_preds[:, 2] - avg_preds[:, 0])
sub_df.head(25)
sub_df[['Patient_Week', 'FVC', 'Confidence']].to_csv('submission.csv', index=False)