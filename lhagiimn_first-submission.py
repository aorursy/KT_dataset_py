import numpy as np

import pandas as pd

import pydicom

import os

import random

import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold



from tqdm import tqdm

import torch



import warnings



warnings.simplefilter('ignore')



from torch.distributions import Normal

from torch.utils.data import Dataset, DataLoader

import torch

import torchvision

import torch.nn.functional as F

import torch.nn as nn

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR



# Competation metric for numpy array

def metric(outputs, std, target):

    confidence = std

    clip = np.where(confidence > 70, confidence, 70)

    delta = np.abs(outputs - target)

    delta = np.where(delta > 1000, 1000, delta)



    metrics = (delta * np.sqrt(2) / clip) + np.log(clip * np.sqrt(2))



    return np.mean(metrics)
class LungDataset(Dataset):

    def __init__(self, df, train=True, meta_features=None):



        self.df = df

        self.train = train

        self.meta_features = meta_features



    def __getitem__(self, index):



        # print(self.df.iloc[index]['Patient'])

        meta = np.array(self.df.iloc[index][self.meta_features].values, dtype=np.float32)



        if self.train:

            y = self.df.iloc[index]['target']

            return (meta), y

        else:

            return (meta)



    def __len__(self):

        return len(self.df)





class Bayesian_Regression(nn.Module):



    def __init__(self, input_size):

        super(Bayesian_Regression, self).__init__()



        self.fc1 = nn.Linear(input_size, 128)

        self.fc2 = nn.Linear(128, 64)

        self.fc3 = nn.Linear(64, 32)

        self.std = nn.Linear(32, 1)

        self.mean = nn.Linear(32, 1)



    def forward(self, X):

        h = self.fc1(X)

        h = self.fc2(h)

        h = self.fc3(h)



        std = F.softplus(self.std(h))

        mean = self.mean(h)



        return mean, std





from torch.utils.data import Dataset, DataLoader





class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""



    def __init__(self, patience=7, verbose=False, delta=0):



        self.patience = patience

        self.verbose = verbose

        self.counter = 0

        self.best_score = None

        self.early_stop = False

        self.val_loss_min = np.Inf

        self.delta = delta



    def __call__(self, val_loss, model, path):



        score = -val_loss



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(val_loss, model, path)

        elif score < self.best_score - self.delta:

            self.counter += 1

            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(val_loss, model, path)

            self.counter = 0



    def save_checkpoint(self, val_loss, model, path):

        '''Saves model when validation loss decrease.'''

        if self.verbose:

            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model, path)

        self.val_loss_min = val_loss





def training(model, train_data, val_data, epochs,

             idx, batch_size=64, lr=0.001, patience=5):



    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)



    if os.path.isfile('checkpoint_%s.pt' % idx):



        # load the last checkpoint with the best model

        model = (torch.load('checkpoint_%s.pt' % idx))



        return model, valloader



    else:



        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=3, verbose=True)

        train_losses = []

        valid_losses = []

        avg_train_losses = []

        avg_valid_losses = []

        val_loss_torch = []

        early_stopping = EarlyStopping(patience=patience, verbose=True)



        for e in range(epochs):



            model.train()

            for x, y in tqdm(trainloader):



                trainX = torch.tensor(x, device=device, dtype=torch.float32)

                trainY = torch.tensor(y, device=device, dtype=torch.double)



                # Forward pass

                mean, std = model(trainX)



                gaussian = torch.distributions.normal.Normal(mean.double(), std.double())

                pred_y = gaussian.sample([1000])

                likelihood = gaussian.log_prob(trainY)



                loss = torch.mean(torch.log(std)/2 + torch.square(trainY-mean)/(2*std))-torch.mean(likelihood)



                # Backward and optimize

                optimizer.zero_grad()

                loss.backward()



                # nn.utils.clip_grad_norm_(meta_reg.parameters(), 5)

                optimizer.step()

                train_losses.append(loss.item())

            

            pred_y_mean = []

            pred_y_std = []

            true_y = []

            model.eval()

            for val_x, val_y in tqdm(valloader):

                valX = torch.tensor(val_x, device=device, dtype=torch.float32)

                valY = torch.tensor(val_y, device=device, dtype=torch.double)



                mean, std = model(valX)



                gaussian = torch.distributions.normal.Normal(mean.double(), std.double())

                pred_y = gaussian.sample([1000])

                likelihood = gaussian.log_prob(trainY)



                val_loss = torch.mean(torch.log(std)/2 + torch.square(trainY-mean)/(2*std)) - torch.mean(likelihood)



                valid_losses.append(val_loss.item())

                val_loss_torch.append(val_loss.unsqueeze(0))

                

                pred_y_mean.append(mean)

                pred_y_std.append(std)

                true_y.append(valY)

            

            pred_y_mean = torch.cat(pred_y_mean, dim=0).detach().cpu().numpy()

            pred_y_std = torch.cat(pred_y_std, dim=0).detach().cpu().numpy()

            true_y = torch.cat(true_y, dim=0).detach().cpu().numpy()



            train_loss = np.average(train_losses)

            valid_loss = np.average(valid_losses)

            avg_train_losses.append(train_loss)

            avg_valid_losses.append(valid_loss)



            scheduler.step(torch.mean(torch.cat(val_loss_torch, dim=0)))

            epoch_len = len(str(epochs))



            print_msg = (f'[{e:>{epoch_len}}/{epochs:>{epoch_len}}] ' +

                         f'train_loss: {train_loss:.5f} ' +

                         f'valid_loss: {valid_loss:.5f}' + 

                        f'metric: {metric(pred_y_mean, pred_y_std, true_y)}')



            print(print_msg)



            early_stopping(valid_loss, model, path='checkpoint_%s.pt' % idx)



            if early_stopping.early_stop:

                print("Early stopping")

                break



        # load the last checkpoint with the best model

        model = torch.load('checkpoint_%s.pt' % idx)



        return model, valloader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = "../input/osic-pulmonary-fibrosis-progression"



tr = pd.read_csv(f"{ROOT}/train.csv")

tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])

chunk = pd.read_csv(f"{ROOT}/test.csv")



print("add infos")

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")

sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]

sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")
tr['WHERE'] = 'train'

chunk['WHERE'] = 'val'

sub['WHERE'] = 'test'

data = tr.append([chunk, sub])



data['min_week'] = data['Weeks']

data.loc[data.WHERE=='test','min_week'] = np.nan

data['min_week'] = data.groupby('Patient')['min_week'].transform('min')



base = data.loc[data.Weeks == data.min_week]

base = base[['Patient', 'FVC', 'Percent']].copy()

base.columns = ['Patient','min_FVC', 'min_percent']

base['nb'] = 1

base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

base = base[base.nb==1]

base.drop('nb', axis=1, inplace=True)



data = data.merge(base, on='Patient', how='left')

data['base_week'] = data['Weeks'] - data['min_week']

del base



COLS = ['Sex','SmokingStatus'] #,'Age'

FE = []

for col in COLS:

    for mod in data[col].unique():

        FE.append(mod)

        data[mod] = (data[col] == mod).astype(int)



data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )

data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )

data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )

data['min_percent_norm'] = (data['min_percent'] - data['min_percent'].min() ) / ( data['min_percent'].max() - data['min_percent'].min() )

data['target'] = data['FVC']#/data['FVC'].max()

FE += ['age','week','BASE', 'min_percent_norm']



print(FE)



# data.to_csv('df.csv', index=False)



tr = data.loc[data.WHERE=='train']

chunk = data.loc[data.WHERE=='val']

sub = data.loc[data.WHERE=='test']



kf = KFold(n_splits=5)



idx=0



pred_test = np.zeros((sub.shape[0], 2))

pred_train = np.zeros((tr.shape[0], 3))



models = []



for tr_idx, val_idx in kf.split(tr.index):



    model = Bayesian_Regression(input_size=len(FE))

    model = model.to(device)



    train_data = LungDataset(tr.loc[tr_idx], meta_features=FE)

    val_data = LungDataset(tr.loc[val_idx], meta_features=FE)



    model_nn, val_loader = training(model, train_data, val_data, 1000,

             idx, batch_size=64, lr=0.025, patience=5)

    

    models.append(model_nn)



    pred_y_val = []

    std_y_val = []

    true_y_val = []

    model.eval()

    for val_x, val_y in tqdm(val_loader):

        valX = torch.tensor(val_x, device=device, dtype=torch.float32)

        mean, std = model_nn(valX)



        #gaussian = torch.distributions.normal.Normal(mean.double(), std.double())

        #pred_y = gaussian.sample([1000])



        pred_y_val.append(mean)

        std_y_val.append(std)

        true_y_val.append(val_y)



    pred_y_mean = torch.cat(pred_y_val, dim=0).detach().cpu().numpy()

    pred_y_std = torch.cat(std_y_val, dim=0).detach().cpu().numpy()

    true_y = torch.cat(true_y_val, dim=0).detach().cpu().numpy()

    true_y = true_y #* data['FVC'].max()



    plt.plot(true_y)

    plt.plot(pred_y_mean)

    plt.plot(pred_y_mean+pred_y_std)

    plt.show()

    

    print(metric(pred_y_mean, pred_y_std, true_y))





    test_data = LungDataset(sub, train=False, meta_features=FE)

    testloader = DataLoader(test_data, batch_size=256, shuffle=False)



    pred_y_test = []

    std_y_test = []

    model.eval()

    for test_x in tqdm(testloader):

        testX = torch.tensor(test_x, device=device, dtype=torch.float32)

        mean, std = model_nn(testX)



        pred_y_test.append(mean)

        std_y_test.append(std)



    pred_y_mean = torch.cat(pred_y_test, dim=0).detach().cpu().numpy()

    pred_y_std = torch.cat(std_y_test, dim=0).detach().cpu().numpy()



    pred_test[:, 0] = pred_test[:, 0] + pred_y_mean[:, 0]

    pred_test[:, 1] = pred_test[:, 1] + pred_y_std[:, 0]



    idx=idx+1



pred_test = pred_test/5



sub = pd.read_csv(f"{ROOT}/sample_submission.csv")

sub[['FVC','Confidence']] = pred_test



sub.to_csv("submission.csv", index=False)