# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_set = pd.read_csv('../input/TTiDS20/train.csv')

zips = gpd.read_file('../input/TTiDS20/zipcodes.csv')

test = pd.read_csv('../input/TTiDS20/test_no_target.csv')

submission = pd.read_csv('../input/TTiDS20/sample_submission.csv')

zips.loc[:, 'zipcode'] = zips['zipcode'].astype('int64')
train_set.head()
# train_set.columns

train_set['brand'].unique()
zip_dict = {x: y for x, y in zip(zips['zipcode'], zips['city'])}
train_set.loc[:, 'city'] = train_set['zipcode'].map(zip_dict)

test.loc[:, 'city'] = test['zipcode'].map(zip_dict)

train_set
def fix_dates(string):

    if len(string) == 1:

        return '200'+string

    if len(string) == 2:

        return '19'+string

    if len(string) == 4:

        return string



train_set.loc[:, 'registration_year'] = train_set['registration_year'].astype('str')    

train_set.loc[:, 'registration_year'] = train_set['registration_year'].apply(fix_dates)



test.loc[:, 'registration_year'] = test['registration_year'].astype('str')    

test.loc[:, 'registration_year'] = test['registration_year'].apply(fix_dates)

train_set
def replace_numeric_nans(df, strategy='median', constant=None):

    for col in df.columns:

        if df[col].isna().sum() > 0 and df[col].dtype != 'object':

            if strategy == 'median':

                median = df[col].median()

                df.loc[:, col] = df[col].fillna(median)

            if constant is not None and strategy == 'constant':

                df.loc[:, col] = df[col].fillna(constant)

    if strategy == 'median':

        return (df, median)

    else:

        return (df, constant)



def replace_str_nans(df, strategy='unknown'):

    for col in df.columns:

        if df[col].isna().sum() > 0 and df[col].dtype == 'object':

            if strategy == 'unknown':

                #mode = df[col].mode()[0]

                #print(mode)

                df.loc[:, col] = df[col].fillna('unknown')

    return df    



train_set, med = replace_numeric_nans(train_set, 'median')

train_set = replace_str_nans(train_set)



test, _ = replace_numeric_nans(test, 'constant', med)

test = replace_str_nans(test)

train_set
from category_encoders import OneHotEncoder



def cat_encoder(df, columns, target=None, test=False, encoder=None):

    if test:

        return encoder.transform(df)

    else:

        encoder = OneHotEncoder(cols=columns, return_df=True)

        encoder.fit(df, target)

        df = encoder.transform(df)

        return (df, encoder)



cat_cols = ['type', 'registration_year', 'gearbox', 'model', 'fuel', 'brand']

X = train_set.drop(['Unnamed: 0', 'zipcode', 'price', 'city'], axis=1)

y = train_set['price']

test = test.drop(['Unnamed: 0', 'zipcode', 'city'], axis=1)



X, encoder = cat_encoder(X, cat_cols, y, False, None)

test = cat_encoder(test, cat_cols, None, True, encoder)

X
"""

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from ignite.contrib.metrics.regression import MedianAbsolutePercentageError



class CarsDataset(Dataset):

    def __init__(self, df, test=False):

        super(CarsDataset, self).__init__()

        self.test = test

        if test:

            self.df = df

        else:

            self.df = df.drop('price', axis=1)

            self.target = df['price']

    

    def __len__(self):

        return self.df.shape[0]

    

    def __getitem__(self, ind):

        if self.test:

            return self.df.values[ind]

        else:

            return (self.df.values[ind], self.target.values[ind])

        



class CarsModel(nn.Module):

    def __init__(self):

        super(CarsModel, self).__init__()

        self.lin1 = nn.Linear(385, 256)

        self.norm1 = nn.BatchNorm1d(256)

        self.lin2 = nn.Linear(256, 64)

        self.lin3 = nn.Linear(64, 1)

        #self.output = nn.Linear()

        

    def forward(self, x):

        #x = x.view(-1, 12)

        x = F.relu(self.lin1(x.float()))

        x = self.norm1(x)

        x = F.relu(self.lin2(x))

        x = F.relu(self.lin3(x))

        return x

    

dataset_train = CarsDataset(pd.concat([X, y], axis=1))

dataset_test = CarsDataset(test, test=True)



dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)

dataloader_test = DataLoader(dataset_test, batch_size=16)

#print(dataloader_train.dataset.df)

#print(dataloader_test.dataset.df)



def mean_absolute_percentage_error(y_true, y_pred):

    #y_true, y_pred = np.array(y_true), np.array(y_pred)

    y_true, y_pred = y_true.detach().numpy(), y_pred.detach().numpy()

    return torch.Tensor(np.mean(np.abs((y_true - y_pred) / y_true)), requires_grad=True)



def fit():

    model = CarsModel()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.MSELoss()

    epochs = 1

    

    

    for epoch in range(epochs):

        total_loss = 0

        for features, target in tqdm(dataloader_train):

            optimizer.zero_grad()

            # = each

            y = model(features)

            loss = criterion(y.float(), target.view(-1, 1).float())

            #print(loss.__dir__())

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print('epoch:', epoch, 'loss:', total_loss)

    return model



model = fit()



model.eval()



preds = []

for features in tqdm(dataloader_test):

    #print(features)

    y = model(features)

    preds.append(y[0].detach().numpy())

print(preds)

"""
from sklearn.model_selection import KFold, RepeatedKFold

from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor



kf = RepeatedKFold(n_splits=5, n_repeats=2)



model_1 = GradientBoostingRegressor(n_estimators=200, max_depth=20, verbose=1)

model_2 = RandomForestRegressor(n_estimators=100, verbose=2)



for train_index, test_index in kf.split(train_set):

    print('Fitting Fold...')

    

    X_train, X_test = X.values[train_index], X.values[test_index]

    y_train, y_test = y.values[train_index], y.values[test_index]

    model_1.fit(X_train, y_train)

    print(model_1.score(X_test, y_test))



preds = model_1.predict(test.values)

preds
submission.loc[:, 'Predicted'] = preds

submission.to_csv('submission.csv', index=False)