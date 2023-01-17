# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn.functional as F



from scipy.special import expit

import math

import random



from sklearn.model_selection import KFold, RepeatedStratifiedKFold

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt



import pdb
seed = 42

random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
sample = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



sample.head()
train_df = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

target_df = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')



train_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')
train_nonscored.iloc[:,1:].sum(axis=0).reset_index(name='count').sort_values(by='count', ascending=False)
train_df.head()
train_df['cp_type'].value_counts()
# Map categorical features, because they are binary, just keep 0/1



def preprocess(df):

    df['cp_type'] = df['cp_type'].map({'ctl_vehicle':0,'trt_cp':1})

    df['cp_dose'] = df['cp_dose'].map({'D1':0,'D2':1})

#     df['cp_time'] = df['cp_time'].map({24:0, 48:1, 72:2}) # keep order

    return df
train_df = preprocess(train_df)
# get feature columns

feature_cols = train_df.columns[1:]
# get target columns/labels

target_cols = target_df.columns[1:]  # without id



# Merge training features with target labels on the drug id

full_df = pd.merge(train_df, target_df, how='left', on='sig_id')
full_df
# Create validation folds

folds = 5

kf = KFold(n_splits=folds, random_state=0, shuffle=True)

full_df['fold'] = -1

for i, (train_index, valid_index) in enumerate(kf.split(X=full_df[feature_cols])):

    full_df.loc[valid_index, 'fold'] = i
# Dataset class



class TrainDataset(Dataset):

    def __init__(self, features, targets):

        self.features = features

        self.targets = targets

        

    def __len__(self):

        return len(self.targets)

    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()



        x = torch.tensor(self.features[idx, :], dtype=torch.float32)

        y = torch.tensor(self.targets[idx, :], dtype=torch.float32)

        return x, y    
class MoA(nn.Module):

    def __init__(self, n_features, n_targets, layers):

        super().__init__()

        

        self.n_features = n_features

        self.n_targets = n_targets

        self.layers = layers



        layerlist = []

        n_in = self.n_features

        for i in self.layers:

            layerlist.append(nn.Linear(n_in, i, bias=False))

            layerlist.append(nn.BatchNorm1d(i))

            layerlist.append(nn.PReLU())

            layerlist.append(nn.Dropout(p=0.5))

            n_in = i

            

        # ouptut 

        layerlist.append(nn.Linear(layers[-1], self.n_targets))

        

        self.model = nn.Sequential(*layerlist)

        

    def forward(self, x):

        return self.model(x)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

print(device)
# Test data preperation

test_df = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')





test_df = preprocess(test_df)

test_features = test_df[feature_cols].to_numpy()

    

test_dataset = TrainDataset(test_features, np.zeros((test_features.shape[0], target_cols.shape[0])))

    

test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
# epochs = 50

# model = MoA(n_features=len(feature_cols), n_targets=len(target_cols), layers=[1024,512,256,128]).to(device)

# criterion = nn.BCEWithLogitsLoss()



# optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.02, lr=3e-2)

# scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=3, verbose=True)
# learning rate finder

def find_lr(net, trn_loader, optimizer, criterion, init_value = 1e-8, final_value=10., beta = 0.98):

    num = len(trn_loader)-1

    mult = (final_value / init_value) ** (1/num)

    lr = init_value

    optimizer.param_groups[0]['lr'] = lr

    avg_loss = 0.

    best_loss = 0.

    batch_num = 0

    losses = []

    log_lrs = []

    for x, y in trn_loader:

        batch_num += 1

        #As before, get the loss for this mini-batch of inputs/outputs

        optimizer.zero_grad()

        x = x.to(device)

        y = y.to(device)

        y_pred = net(x)

        loss = criterion(y_pred, y)

        #Compute the smoothed loss

        avg_loss = beta * avg_loss + (1-beta) *loss.item()

        smoothed_loss = avg_loss / (1 - beta**batch_num)

        #Stop if the loss is exploding

        if batch_num > 1 and smoothed_loss > 4 * best_loss:

            return log_lrs, losses

        #Record the best loss

        if smoothed_loss < best_loss or batch_num==1:

            best_loss = smoothed_loss

        #Store the values

        losses.append(smoothed_loss)

        log_lrs.append(math.log10(lr))

        #Do the SGD step

        loss.backward()

        optimizer.step()

        #Update the lr for the next step

        lr *= mult

        optimizer.param_groups[0]['lr'] = lr

    return log_lrs, losses
class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self):

        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.001):

        confidence = 1. - smoothing

        logprobs = F.log_softmax(x, dim=-1)

        bcs_loss = nn.BCEWithLogitsLoss()(x, target)

        smooth_loss = -logprobs.mean(dim=-1)

        loss = confidence * bcs_loss + smoothing * smooth_loss

        return loss.mean()
def train(model, optimizer, criterion, metric, scheduler, batch_size, epochs,

          X_train, y_train, X_val, y_val, dev=False):

    

    train_dataset = TrainDataset(X_train, y_train)

    valid_dataset = TrainDataset(X_val, y_val)

    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    

    if dev:

        log_lrs, losses =  find_lr(model, train_loader, optimizer, criterion)

        plt.plot(log_lrs,losses)

        return

    

    lowest_loss = 999

    no_improve = 0

    

    for i in range(epochs):



        train_losses = []

        valid_losses = []

        metric_vals = []





        model.train()

        for x_train, y_train in train_loader:

            optimizer.zero_grad()



            x_train = x_train.to(device)

            y_train = y_train.to(device)

            # predict

            y_pred = model(x_train)

            train_loss = criterion(y_pred, y_train)



            train_losses.append(train_loss.item())

            # Update parameters

            train_loss.backward() 

            optimizer.step()



        model.eval()

        with torch.no_grad():

            for x_valid, y_valid in valid_loader:

                x_valid = x_valid.to(device)

                y_valid = y_valid.to(device)



                # predict

                y_pred = model(x_valid)

                valid_loss = criterion(y_pred, y_valid)

                valid_losses.append(valid_loss.item())

                metric_vals.append(metric(y_pred, y_valid).item())



            avg_loss = np.mean(valid_losses)

            avg_metric = np.mean(metric_vals)

            

        if avg_metric < lowest_loss:

            lowest_loss = avg_metric

            no_improve=0

        else:

            no_improve += 1

    

        print(f"Epoch {i}: Train Loss = {np.mean(train_losses)}, Valid Loss = {avg_loss}, Metric = {avg_metric}")

        

        if no_improve==5:

            print('No improvement, stopping')

            break

            

        scheduler.step(avg_loss)        



    return model
def test(model, test_loader):

    test_preds = []

    model.eval()

    with torch.no_grad():

        for  x_test, _ in test_loader:

            x_test = x_test.to(device)

            # pred

            y_pred = model(x_test)

            test_preds = np.append(test_preds, y_pred.cpu().numpy())

    

    return expit(test_preds) # need to apply sigmoind on logits
def create_submission(test_df, test_preds):

    

    sub_df = test_df[['sig_id']].copy()

    sub_df.loc[:, target_cols] = 0

    sub_df.loc[:, target_cols] = test_preds.reshape(len(test_df), len(target_cols))



    # set control group MoA to 0

    sub_df.loc[test_df[test_df['cp_type']==0].index,1:]=0



    return sub_df
folds = 5

epochs = 30

batch_size = 256

lr = 5e-3



for fold in range(folds):

    

    print(f'Training fold {fold}')

    

    # Initialize model

    model = MoA(n_features=len(feature_cols), n_targets=len(target_cols), layers=[1024,512,512,256]).to(device)

    criterion = nn.BCEWithLogitsLoss()

    smooth_criterion = LabelSmoothingCrossEntropy()

    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5, verbose=True)





    # Split data based on fold

    X_train = full_df[full_df['fold'] != fold][feature_cols].to_numpy()

    y_train = full_df[full_df['fold'] != fold][target_cols].to_numpy()

    

    X_valid = full_df[full_df['fold'] == fold][feature_cols].to_numpy()

    y_valid = full_df[full_df['fold'] == fold][target_cols].to_numpy()

    

    

    # train

    model = train(model, optimizer, smooth_criterion, criterion, scheduler, 

                  batch_size, epochs, X_train, y_train, X_valid, y_valid)



    # predict

    if fold == 0:

        test_preds = test(model, test_loader)/folds  # divide by folds to get average

    else:

        test_preds += test(model, test_loader)/folds  # divide by folds to get average

    



sub_df = create_submission(test_df, test_preds)
sub_df.to_csv('submission.csv', index=False)