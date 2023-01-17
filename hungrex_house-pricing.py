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
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train_len = train.shape[0]
test_len = test.shape[0]
data = pd.concat([train, test]).reset_index(drop=True)
print(train.info())
print(test.info())
import seaborn as sns
import matplotlib.pyplot as plt
print(data.info())
plt.scatter(x = train[['SalePrice']] ,y = train[['TotalBsmtSF']])
## Pool QC
data["PoolQC"] = data["PoolQC"].fillna("None")
# Misc 
data["MiscFeature"] = data["MiscFeature"].fillna("None")
# alley
data["Alley"] = data["Alley"].fillna("None")
# fence
data["Fence"] = data["Fence"].fillna("None")
# fire place
data["FireplaceQu"] = data["FireplaceQu"].fillna("None")
for i in ['GarageArea', 'GarageCars', 'GarageYrBlt']:
    data[i] = data[i].fillna(0)
for i in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
    data[i] = data[i].fillna(0)
for i in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    data[i] = data[i].fillna('None')
data["MasVnrType"] = data["MasVnrType"].fillna("None")
data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
#mode return the highest frequency
data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])
data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
data['MSSubClass'] = data['MSSubClass'].fillna("None")

#MSSubClass=The building class
data['MSSubClass'] = data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
data['OverallCond'] = data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
data['YrSold'] = data['YrSold'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder

cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold']
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(data[c].values)) 
    data[c] = lbl.transform(list(data[c].values))

total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#correlation matrix
corrmat = data.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix
plt.subplots(figsize=(12, 9))

k = 15 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
print(cols)
cm = np.corrcoef(train[cols].values.T)
print(data[cols].values.T.shape)
print(data[cols].values.shape)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.3f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


train = data[:train_len]
test = data[train_len:]
import torch 
import torch.nn.functional as F

from tqdm import tqdm
class Net(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(Net, self).__init__() 
        self.h1 = torch.nn.Linear(n_in, 12)
        self.h2 = torch.nn.Linear(12, 8)
        self.h3 = torch.nn.Linear(8, 6)
        self.h4 = torch.nn.Linear(6, 3)
        self.predict = torch.nn.Linear(3, n_out)
    def forward(self, x):
        x = F.relu(self.h1(x))
#         x = F.dropout(x, p=0.1)
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        x = F.relu(self.h4(x))
        return self.predict(x)
    

# print(len(train_x))
# print(train_x.iloc[1])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader
import torch.utils.data as data
class Data(data.Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label
    def __len__(self):
        return len(self.feature)
    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx]
#old without dataloader
mask = ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'GarageArea', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
train_x = train[mask]
train_y = train['SalePrice']


model = Net(n_in = 10, n_out= 1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

loss_F = torch.nn.MSELoss(reduction='mean')

# Clean up gradient in model parameters.
model.zero_grad()

for epoch in tqdm(range(10)):
    for i in range(len(train_x)):
        predict_y = model(torch.tensor(train_x.iloc[i]).float())
        loss = loss_F(predict_y, torch.tensor(train_y).float())
        
        loss.backward() # back propragation
        optimizer.step() # update the paramters 
        optimizer.zero_grad() # optimizer 0

#         print(loss)
from torch.utils.data import DataLoader

mask = ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'GarageArea', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
train_x = train[mask]
train_y = train['SalePrice']


def train_model(train_x, train_y, batch_size, lr, epoch, n_in):
    model = Net(n_in = n_in, n_hid = 6, n_out= 1)

    if(train_y == None):
            return model(train_x.float())
    else:
        train_data = Data(train_x,train_y)
        train_loader = DataLoader(train_data, shuffle = True, batch_size = batch_size)


        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        loss_F = torch.nn.MSELoss(reduction='mean')

        # Clean up gradient in model parameters.
        model.zero_grad()
        for epoch in tqdm(range(epoch)):
            for (tr, target) in train_loader:
                predict_y = model(tr.float())
                loss = loss_F(predict_y, target.float())
                optimizer.zero_grad() # optimizer 0
                loss.backward() # back propragation
                optimizer.step() # update the paramters 

    #             predict_y = model(torch.tensor(train_x.iloc[i]).float())
    #             loss = loss_F(predict_y, torch.tensor(train_y[i]).float())
    # #             print("epoch", epoch, loss)


# print(type(train_x))
train_model(torch.tensor(train_x.values), torch.tensor(train_y.values), 32, 0.001, 30, 11)
#test without DataLoader
mask = ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'GarageArea', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
test_x = test[mask]
test_y = []


for i in range(len(test)):
    predict_y = model(torch.tensor(test_x.iloc[i]).float())
    test_y.append(float(predict_y))

test['SalePrice'] = (test_y)
# print(test.head())

print(test['SalePrice'].describe())
mask = ['GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF', 'GarageArea', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea']
test_x = test[mask]
test_y = []


for i in range(len(test)):
    predict_y = model(torch.tensor(test_x.iloc[i]).float())
    test_y.append(float(predict_y))

test['SalePrice'] = (test_y)
# print(test.head())

print(test['SalePrice'].describe())
submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
submission.to_csv('submission.csv', index=False)