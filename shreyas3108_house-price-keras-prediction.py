# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# The Data Cleaning Part was based on the idea of https://www.kaggle.com/meikegw/filling-up-missing-values 

# Special thanks to meikegw for a wonderful notebook on missing value and data preprocessing. 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
train.head()
train.describe()
train.columns
import matplotlib.pyplot as plt 

import seaborn as sns
var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice')
train.plot.scatter(x = 'TotalBsmtSF',y='SalePrice')
train.plot.scatter(x ='YearBuilt',y = 'SalePrice' )
ab = train.corr()
ab
f,ax = plt.subplots(figsize=(12, 9))

sns.heatmap(ab , vmax = .75)
print(train.isnull().sum())
miss=train.columns[train.isnull().any()].tolist()
miss
train[miss].isnull().sum()
train['SqrtLotArea']=np.sqrt(train['LotArea'])

sns.pairplot(train[['LotFrontage','SqrtLotArea']].dropna())
cond = train['LotFrontage'].isnull()

train.LotFrontage[cond]=train.SqrtLotArea[cond]
del train['SqrtLotArea']
train[['MasVnrType','MasVnrArea']][train['MasVnrType'].isnull()==True]

def cat_exploration(column):

    return train[column].value_counts()
# Imputing the missing values

def cat_imputation(column, value):

    train.loc[train[column].isnull(),column] = value
cat_exploration('Alley')
cat_imputation('MasVnrType', 'None')

cat_imputation('MasVnrArea', 0.0)
basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']

train[basement_cols][train['BsmtQual'].isnull()==True]
for cols in basement_cols:

    if 'FinSF'not in cols:

        cat_imputation(cols,'None')
# Impute most frequent value

cat_imputation('Electrical','SBrkr')
train.head()
cat_exploration('FireplaceQu')
train['Fireplaces'][train['FireplaceQu'].isnull()==True].describe()
cat_imputation('FireplaceQu','None')

pd.crosstab(train.Fireplaces,train.FireplaceQu)
garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']

train[garage_cols][train['GarageType'].isnull()==True]
for cols in garage_cols:

    if train[cols].dtype==np.object:

        cat_imputation(cols,'None')

    else:

        cat_imputation(cols, 0)
cat_exploration('PoolQC')

train['PoolArea'][train['PoolQC'].isnull()==True].describe()
cat_imputation('PoolQC', 'None')

cat_imputation('Fence', 'None')

cat_imputation('MiscFeature', 'None')
def show_missing():

    missing = train.columns[train.isnull().any()].tolist()

    return missing

train[show_missing()].isnull().sum()
cat_imputation('Alley','None')
train[show_missing()].isnull().sum()
train.head()
train.columns
from keras.models import Sequential

from keras.optimizers import SGD,RMSprop

from keras.layers import Dense,Dropout,Activation

from keras.layers.normalization import BatchNormalization
y = np.log1p(train[['SalePrice']])
y.mean()
test = pd.read_csv("../input/test.csv")
test.head()
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                    test.loc[:,'MSSubClass':'SaleCondition']))
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
x_train = np.array(all_data[:train.shape[0]])

x_test = np.array(all_data[test.shape[0]+1:])
x_train
from sklearn.model_selection import train_test_split 
X_train, X_valid, y_train, y_valid = train_test_split(x_train, y)
from keras.activations import relu
X_train.shape
model = Sequential()

model.add(Dense(1024,input_dim = 302,kernel_initializer='uniform'))

model.add(Activation(relu))

model.add(BatchNormalization())

model.add(Dropout(0.6))

model.add(Dense(512,input_dim=1028,activation='relu',kernel_initializer='uniform'))

model.add(BatchNormalization())

model.add(Dropout(0.8))

model.add(Dense(256))

model.add(Dropout(0.8))

model.add(Dense(128))

model.add(Dense(1))

model.compile( optimizer='adam',loss='mse',metrics=['mean_squared_error'])
model.summary()
model.fit(X_train,y_train,validation_data=(X_valid,y_valid),nb_epoch=30,batch_size=128)
np.sqrt(model.evaluate(X_valid,y_valid))
def rmse(predictions, targets):

    return np.sqrt(((predictions - targets) ** 2).mean())
preds = model.predict(np.array(x_test))
x_test
subm = pd.read_csv("../input/sample_submission.csv")
subm.shape
subm.iloc[:,1] = np.array(model.predict(np.array(x_test)))
print(subm[['SalePrice']].mean())
subm['SalePrice'] = np.expm1(subm[['SalePrice']])

print(subm[['SalePrice']].mean())
subm.to_csv('sub1.csv', index=None)