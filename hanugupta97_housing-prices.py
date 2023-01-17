# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
train_ID = train['Id']

test_ID = test['Id']

train = train.drop(columns = ['Id'])

test = test.drop(columns = ['Id'])

train.shape, test.shape
fig, ax = plt.subplots()

ax.scatter( train['GrLivArea'],train['SalePrice'])

plt.xlabel('GrLivArea')

plt.ylabel('SalesPrice')

plt.show
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



fig, ax = plt.subplots()

ax.scatter( train['GrLivArea'],train['SalePrice'])

plt.xlabel('GrLivArea')

plt.ylabel('SalesPrice')

plt.show

from scipy import stats

from scipy.stats import norm, skew

sns.distplot(train['SalePrice'],fit = norm)

(mu, sigma) = norm.fit(train['SalePrice'])

plt.legend(['Normal Distribution $\mu =$ {:.2f} and $\sigma = $ {:.2f}'.format(mu,sigma)], loc = 'best')

plt.xlabel('SalePrice')

plt.ylabel('Frequrency')

fig = plt.figure()



res = stats.probplot(train['SalePrice'], plot = plt)

plt.show()
y_train = train.SalePrice.values

train = train.drop(columns = ['SalePrice'])

Data = pd.concat([train,test]).reset_index(drop = True)

Data.shape

data_na = (Data.isnull().sum()/len(Data))*100

data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending = False)

missing_na = pd.DataFrame({'Missing':data_na})

missing_na.head(10)
fig, ax = plt.subplots(figsize = (10,8))

sns.barplot(data_na.index,data_na)

plt.xticks(rotation=90)

plt.xlabel('features')

plt.ylabel('frequency_percentage')

plt.show()
train.corr()
missing_na.head(10)
Data['PoolQC'] = Data['PoolQC'].fillna('None')

Data['MiscFeature'] = Data['MiscFeature'].fillna('None')

Data['Alley'] = Data['Alley'].fillna('None')

Data['Fence'] = Data['Fence'].fillna('None')

Data['FireplaceQu'] = Data['FireplaceQu'].fillna('None')
Data['LotFrontage'] = Data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
for col in (['GarageFinish','GarageYrBlt','GarageQual','GarageCond']):

    Data[col] = Data[col].fillna('none')
for col in (['GarageCars', 'GarageArea','BsmtFullBath','BsmtHalfBath','BsmtFinSF1',

             'MasVnrArea','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFinSF1']):

    Data[col] = Data[col].fillna(0)
Data['MSZoning'] = Data['MSZoning'].fillna('A')

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    Data[col] = Data[col].fillna('None')
Data['Utilities'] = Data['Utilities'].fillna('ELO')

Data['Exterior1st'] = Data['Exterior1st'].fillna('Others')

Data['Exterior2nd'] = Data['Exterior2nd'].fillna('Others')
Data['MasVnrType'] = Data['MasVnrType'].fillna('None')

Data['Electrical'] = Data['Electrical'].fillna('SBrkr')

Data['KitchenQual'] = Data['KitchenQual'].fillna('TA')

Data['Functional'] = Data['Functional'].fillna('Typ')

Data['GarageType'] = Data['GarageType'].fillna('NA')

Data['SaleType'] = Data['SaleType'].fillna('Oth')
all(Data.isnull())
Data.info()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Data['MSSubClass'] = Data['MSSubClass'].apply(str)

Data['OverallCond'] = Data['OverallCond'].astype(str)

Data['YrSold'] = Data['YrSold'].astype(str)

Data['MoSold'] = Data['MoSold'].astype(str)

    
Data.columns
cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold']

for i in cols:

    le = LabelEncoder()

    le.fit(list(Data[i].values))

    Data[i] = le.transform(list(Data[i].values))

Data.head()
Data['TotalSf'] = Data['TotalBsmtSF'] + Data['1stFlrSF'] + Data['2ndFlrSF']
#Data = Data.drop(columns = ['TotalBsmtSF','1stFlrSF','2ndFlrSF'])
Data1 = Data[cols]

train_data = Data1.iloc[:train.shape[0],:]

test_data = Data1.iloc[test.shape[0]-1:,:]

train_data.shape, test_data.shape
from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(50, 40))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(Data1)
from sklearn.linear_model import SGDRegressor, ridge_regression, LinearRegression

from sklearn.model_selection import cross_validate

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.svm import LinearSVR
SGD = SGDRegressor(loss = 'squared_loss', penalty = 'l2')

#RR = ridge_regression()

LR = LinearRegression()

KNR = KNeighborsRegressor(n_neighbors = 7, weights = 'uniform')

MLP = MLPRegressor()

SVR = LinearSVR()
x_train, x_test, y_tra, y_test = train_test_split(train_data, y_train, test_size = 0.2, random_state = 0)
from sklearn.metrics import r2_score , mean_squared_error as MSE
SGD.fit(x_train,y_tra)

RR = ridge_regression(x_train, y_tra, alpha = .2)

LR.fit(x_train, y_tra)

KNR.fit(x_train,y_tra)

MLP.fit(x_train,y_tra)

SVR.fit(x_train,y_tra)
estimators = [SGD, LR, KNR,MLP,SVR]

y_hat = {}

for i in range(len(estimators)):

    y = estimators[i].predict(x_test)

    y_hat[str(i)] =  r2_score(y_test, y)
y_hat
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = LR.predict(test_data)

sub.to_csv('submission.csv',index=False)