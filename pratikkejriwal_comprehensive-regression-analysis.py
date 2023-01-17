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
import seaborn as sns

import matplotlib.pyplot as plt
# read the training and testing data

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



train_data.sample(5)
# general information about the columns

train_data.info()
# check for missing data in training data

train_data.isna().sum().sort_values(ascending=False)[:20]
# percentage of missing values per column

(train_data.isna().sum() / train_data.shape[0]).sort_values(ascending=False)[:20]
# removing unwanted columns

train_data = train_data.drop(['PoolQC','MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)

test_data = test_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)
missing_data_col = ['LotFrontage','GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual','MasVnrArea','MasVnrType','Electrical']

train_data[missing_data_col].dtypes
# instead of fillna using Imputer

# replacing float data with mean and string with mode of corresponding columns

from sklearn.impute import SimpleImputer

float_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')



train_data[['LotFrontage', 'GarageYrBlt', 'MasVnrArea']] = float_imputer.fit_transform(train_data[['LotFrontage', 'GarageYrBlt', 'MasVnrArea']])
string_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

temp_var = ['GarageCond','GarageType','GarageFinish','GarageQual','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual','MasVnrType','Electrical']

train_data[temp_var] = string_imputer.fit_transform(train_data[temp_var])
# no null data now

train_data.isnull().sum().sort_values(ascending=False)[:5]
# missing values in test data

test_data.isnull().sum().sort_values(ascending=False)[:30]
test_missing = ['LotFrontage','GarageCond','GarageQual','GarageYrBlt','GarageFinish','GarageType','BsmtCond','BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinType2','MasVnrType','MasVnrArea','MSZoning','BsmtHalfBath','Utilities','Functional','BsmtFullBath','BsmtUnfSF','SaleType','BsmtFinSF2','BsmtFinSF1','Exterior2nd','Exterior1st','TotalBsmtSF','GarageCars','KitchenQual','GarageArea']
# training data and testing data columns having NaN values differ 

missing_data_col == test_missing
test_data[test_missing].dtypes.sort_values()
# filling nan values for float dtype features

float_missing = ['LotFrontage','GarageCars','TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','BsmtFullBath','BsmtHalfBath','MasVnrArea','GarageArea','GarageYrBlt']

test_float_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

test_data[float_missing] = test_float_imputer.fit_transform(test_data[float_missing])
# filling nan values for str dtype features

str_missing = ['BsmtFinType2','GarageCond','GarageQual','Exterior1st','Exterior2nd','GarageFinish','SaleType','GarageType','BsmtCond','Functional','Utilities','BsmtQual','KitchenQual','BsmtExposure','MasVnrType','BsmtFinType1','MSZoning']

test_str_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

test_data[str_missing] = test_str_imputer.fit_transform(test_data[str_missing])
test_data.isnull().sum().sort_values(ascending=False)[:5]
# Swarm Plot of OverallQual vs SalePrice with variations based on OverallCond

plt.figure(figsize=(23,12))

sns.swarmplot(train_data['OverallQual'], train_data['SalePrice'], hue=train_data['OverallCond'], palette='husl')

plt.title("Swarm Plot of OverallQual vs SalePrice with variations based on OverallCond")

plt.xlabel("OverallQual")

plt.ylabel("Sale Price")

plt.show()
# Swarm Plot of Neighborhood vs SalePrice with variations based on MSZoning

plt.figure(figsize=(23,12))

sns.swarmplot(train_data['Neighborhood'], train_data['SalePrice'], hue=train_data['MSZoning'])

plt.title("Swarm Plot of Neighborhood vs SalePrice with variations based on MSZoning")

plt.xlabel("Neighborhood")

plt.ylabel("Sale Price")

plt.show()
# Swarm Plot of Neighborhood vs SalePrice with variations based on MSSubClass

plt.figure(figsize=(23,12))

sns.swarmplot(train_data['Neighborhood'], train_data['SalePrice'], hue=train_data['MSSubClass'], palette='winter')

plt.title("Swarm Plot of Neighborhood vs SalePrice with variations based on MSSubClass")

plt.xlabel("Neighborhood")

plt.ylabel("Sale Price")

plt.show()
# Swarm Plot of Neighborhood vs SalePrice with variations based on HouseStyle

plt.figure(figsize=(23,12))

sns.swarmplot(train_data['Neighborhood'], train_data['SalePrice'], hue=train_data['HouseStyle'])

plt.title("Swarm Plot of Neighborhood vs SalePrice with variations based on HouseStyle")

plt.xlabel("Neighborhood")

plt.ylabel("Sale Price")

plt.show()
# Swarm Plot of Neighborhood vs SalePrice with variations based on Foundation

plt.figure(figsize=(23,12))

sns.swarmplot(train_data['Neighborhood'], train_data['SalePrice'], hue=train_data['Foundation'])

plt.title("Swarm Plot of Neighborhood vs SalePrice with variations based on Foundation")

plt.xlabel("Neighborhood")

plt.ylabel("Sale Price")

plt.show()
# Violin Plot of ExterQual vs SalePrice with variations based on ExterCond

plt.figure(figsize=(23,12))

sns.violinplot(train_data['ExterQual'], train_data['SalePrice'], hue=train_data['ExterCond'], inner='quartile', palette='RdBu')

plt.title("Violin Plot of ExterQual vs SalePrice with variations based on ExterCond")

plt.xlabel("External Quality")

plt.ylabel("Sale Price")

plt.show()
# Violin Plot of BasementQual vs SalePrice with variations based on BsmtCond

plt.figure(figsize=(23,12))

sns.violinplot(train_data['BsmtQual'], train_data['SalePrice'], hue=train_data['BsmtCond'], inner='quartile', palette='RdBu')

plt.title("Violin Plot of BasementQual vs SalePrice with variations based on BsmtCond")

plt.xlabel("Basement Quality")

plt.ylabel("Sale Price")

plt.show()
# KDE Plot

plt.figure(figsize=(15,8))

sns.kdeplot(train_data['SalePrice'])

plt.title('KDE Plot for Sales Price')

plt.show()
# Regression Plot GrLivArea vs Sale Price

plt.figure(figsize=(15,8))

sns.regplot(train_data['GrLivArea'], train_data['SalePrice'])

plt.title("Regression Plot of Living Area vs SalePrice")

plt.xlabel("Living Area")

plt.ylabel("Sale Price")

plt.show()
# Regression Plot LotArea vs SalePrice

plt.figure(figsize=(15,8))

sns.regplot(train_data['LotArea'], train_data['SalePrice'])

plt.title("Regression Plot of Lot Area vs SalePrice")

plt.xlabel("Lot Area")

plt.ylabel("Sale Price")

plt.show()
from sklearn.preprocessing import LabelEncoder



# these features appear to have ordering in them

ordinal_columns = ['Street','LotShape','LandContour','Utilities','LandSlope','BldgType','HouseStyle','ExterQual','ExterCond','BsmtQual','BsmtCond',

                   'BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','CentralAir','KitchenQual','Functional','GarageFinish','GarageQual','GarageCond',

                   'PavedDrive']

train_le = {}

for col in ordinal_columns:

    train_le[col] = LabelEncoder()

    

    train_data[col] = train_le[col].fit_transform(train_data[col])
test_le = {}

for col in ordinal_columns:

    test_le[col] = LabelEncoder()

    

    test_data[col] = test_le[col].fit_transform(test_data[col])
# need to combine both the dataframes or else encoding creates dataframes of different sizes



train_data['train'] = 1

test_data['test'] = 1



combined_df = pd.concat([train_data, test_data])
# use Pandas get_dummies for One-hot Encoding

combined_df = pd.get_dummies(combined_df)
train_data = combined_df[combined_df['train'] == 1]

test_data = combined_df[combined_df['test'] == 1]

train_data = train_data.drop(['train', 'test'], axis=1)

test_data = test_data.drop(['train', 'test', 'SalePrice'], axis=1)
# shape is same for both the datasets

print(train_data.shape, test_data.shape)
from sklearn.preprocessing import RobustScaler



scale = RobustScaler()



features = train_data.drop('SalePrice', axis=1)

features = pd.DataFrame(data=scale.fit_transform(features), columns=features.columns)

target = train_data['SalePrice']
test_scale = RobustScaler()



test_df = pd.DataFrame(data=test_scale.fit_transform(test_data), columns=test_data.columns)
# Pearson Correlation

from yellowbrick.target import FeatureCorrelation

plt.figure(figsize=(25,35))

viz = FeatureCorrelation(labels = features.columns, method='pearson', sort=True)

viz.fit(features, target)

viz.poof()

plt.show()
# Correlation as per Mutual Info Regression

plt.figure(figsize=(25,35))

viz = FeatureCorrelation(labels=features.columns, method='mutual_info-regression', sort=True)

viz.fit(features, target)

viz.poof()

plt.show()
# Heatmap Visualization of Pearson's Coefficient

datacor = train_data.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(datacor, dtype=np.bool))



plt.figure(figsize=(25,20))

sns.heatmap(datacor, mask=mask, annot=False, square=True, cmap='RdBu')

plt.title("Heatmap Visualization of Pearson's Coefficient")

plt.show()
# create a function that would split data into training and testing

def split_data(features, target):

    

    from sklearn.model_selection import train_test_split

    

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)

    

    return x_train, x_test, y_train, y_test
# helper function to evaluate different scores

from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error

def get_score(y_test, y_pred):

    r2 = r2_score(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = mean_squared_error(y_test, y_pred, squared=False)

    rmsle = mean_squared_log_error(y_test, y_pred) 

    return (r2, mse, rmse, rmsle)
# function to Plot the Actual vs Predicted Sale Prices

def plot_data(y_test, y_pred):

    plt.figure(figsize=(18,12))

    plt.plot(y_test.values, label='Actual', c='r')

    plt.plot(y_pred, label='Predicted', c='b')

    plt.title('Actual vs Predicted Sale Price of the House')

    plt.ylabel('Sale Price')

    plt.legend()
# might have to download this package to visualize the PyTorch model

!pip install hiddenlayer
import torch

import torch.nn as nn

import graphviz

import hiddenlayer as hl
# if cuda enabled gpu is present, train on that else on cpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
X_train, X_test, Y_train, Y_test = split_data(features, target)
# convert this data to tensors and transfer them to device

x_train = torch.tensor(X_train.values, device=device, dtype=torch.float)

y_train = torch.from_numpy(Y_train.values).view(1, -1)[0]



x_test = torch.tensor(X_test.values, device=device, dtype=torch.float)

y_test = torch.from_numpy(Y_test.values).view(1, -1)[0]



# y is converted from [dim, 1] to [dim] as this is what is expected by the NN model
y_train = y_train.type(torch.FloatTensor)

y_test = y_test.type(torch.FloatTensor)
y_train = y_train.to(device)

y_test = y_test.to(device)
# utils.data contains data loaders

import torch.utils.data as data_utils
# we first need to convert the training set as a Tensor Dataset

train_data = data_utils.TensorDataset(x_train, y_train)
# specify batch size for this train_loader, if cuda device is present use larger batch_size else a smaller one

# shuffle the dataset before it is fed into the model, to remove any chances to NN picking up irrelevant patterns



train_loader = data_utils.DataLoader(train_data, batch_size=100, shuffle=True)
print(x_train.shape, y_train.shape)
# training points will be divided into batches

len(train_loader)
# iterate through the batches from the loader

x_batch, y_batch = iter(train_loader).next()
# NN architecture -> size of each layer



input_size = x_batch.shape[1]

output_size = 1 # as prediction would be Sale Price

hidden_size = 25
# using a basic Sequential NN model

# input layer which has size = # of features in x_train

# introduced a dropout layer to avoid overfitting

model = nn.Sequential(nn.Linear(input_size, hidden_size),

                      nn.ReLU(),

                      nn.Dropout(p=0.2),

                      nn.Linear(hidden_size, output_size))



# transfer the model to device

model.to(device)

model
# visualize the model

hl_graph = hl.build_graph(model, torch.zeros([1, input_size], device=device))

hl_graph.theme = hl.graph.THEMES["blue"].copy() 

hl_graph
# use objective loss function as the MSE Loss

criterion = nn.MSELoss()
# learning rate for the model

learning_rate = 1e-3
# using Adam optimizer - Momentum based Optimizer

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

optimizer
# set the model to be in training mode as it behaves differently in training and validation phases

model.train()



total_batch = len(train_loader)



# set the number of epochs you want to train for

epochs = 3500



for epoch in range(epochs+1):

    for i, (features, target) in enumerate(train_loader):

        

        # prediction based solely on features

        predict = model(features)

        

        predict = predict.view(1, -1)[0]

        

        # get the loss function which will guide the optimizer to reduce the loss as per the model performance

        loss = criterion(predict, target)

        

        # zero-out the previously calculated gradients

        optimizer.zero_grad()

        

        # perform back propagation

        loss.backward()

        

        # update the model parameters

        optimizer.step()

        

        # print out the training score after few epochs

        if epoch % 500 == 0:

            print(f'| Epoch: {epoch+1:04} | Batch: {i+1:02} | Training Loss: {loss.item():.3f} |')
# before we can use NN for prediction, switch to evaluation mode

# as there are layers in NN that perform differently during training and prediction phases



model.eval()



# as we don't want to calculate gradients while evaluating, turn-off the grad function

with torch.no_grad():

    y_pred_tensor = model(x_test)
# convert the pytorch cuda tensor to cpu

y_pred = y_pred_tensor.to('cpu')
r2, mse, rmse, rmsle = get_score(Y_test, y_pred)



print(f'R2 score of the model is {r2:.3f}')

print(f'MSE score of the model is {mse:.3f}')

print(f'RMSE score of the model is {rmse:.3f}')

print(f'RMSLE score of the model is {rmsle:.5f}')
plot_data(Y_test, y_pred)
# convert the test_data to cuda tensor

test_tensor = torch.tensor(test_df.values, device=device, dtype=torch.float)
# prediction from the model

pred = model(test_tensor)
# convert the size of the model from [size, 1] to [size]

pred = pred.view(1, -1)[0]
# copy tensor from gpu to cpu and then convert it to numpy

pred = pred.to('cpu').detach().numpy()
sub_df = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': pred})

sub_df.to_csv('submit.csv', index=False)
sub_df.sample(5)