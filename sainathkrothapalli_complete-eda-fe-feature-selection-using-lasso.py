# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', None)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print(train.shape,test.shape)
train.describe()
train.info()
test.isnull().sum()
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu','GarageType', 

            'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond', 

            'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"MasVnrType"):

    train[col] = train[col].fillna('None')

    test[col] = test[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 

            'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',"MasVnrArea"):

    train[col] = train[col].fillna(0)

    test[col] = test[col].fillna(0)

train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].apply(lambda x: x.fillna(x.median()))

test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].apply(lambda x: x.fillna(x.median()))

train['Electrical'] =train['Electrical'].fillna(train['Electrical'].mode()[0])
test['MSZoning'].fillna(test['MSZoning'].mode()[0],inplace=True)

test['SaleType'].fillna(test['SaleType'].mode()[0],inplace=True)

test['Utilities'].fillna(test['Utilities'].mode()[0],inplace=True)

test['Exterior1st'].fillna(test['Exterior1st'].mode()[0],inplace=True)

test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0],inplace=True)

test['KitchenQual'].fillna(test['KitchenQual'].mode()[0],inplace=True)

test['Functional'].fillna(test['Functional'].mode()[0],inplace=True)
train['MSSubClass'] = train['MSSubClass'].astype(str)

train['OverallCond'] = train['OverallCond'].astype(str)

train['YrSold'] =train['YrSold'].astype(str)

train['MoSold'] = train['MoSold'].astype(str)
train.isnull().sum().sum()
test.isnull().sum().sum()
train['YrBltAndRemod']=train['YearBuilt']+train['YearRemodAdd']

train['TotalSF']=train['TotalBsmtSF'] +train['1stFlrSF'] +train['2ndFlrSF']



train['Total_sqr_footage'] = (train['BsmtFinSF1'] +train['BsmtFinSF2'] +train['1stFlrSF'] +train['2ndFlrSF'])



train['Total_Bathrooms'] = (train['FullBath'] + (0.5 * train['HalfBath']) + train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath']))



train['Total_porch_sf'] = (train['OpenPorchSF'] + train['3SsnPorch'] +train['EnclosedPorch'] + train['ScreenPorch'] +train['WoodDeckSF'])





train['haspool'] = train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

train['has2ndfloor'] = train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

train['hasgarage'] = train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

train['hasbsmt'] = train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

train['hasfireplace'] = train['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
train['SalePrice'].hist()
print(train['SalePrice'].skew())
train['SalePrice']=np.log1p(train['SalePrice'])

train['SalePrice'].hist()
train['SalePrice'].skew()
import matplotlib.pyplot as plt

import seaborn as sns

corr=train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr);
cor=train.corr()['SalePrice']

cor[cor>0.5].sort_values(ascending=False)
def scatterplot(df,feature):

    sns.scatterplot(x=feature,y='SalePrice',data=df)



scatterplot(train,'GrLivArea')
train=train[train['GrLivArea']<4000]

train.reset_index(drop = True, inplace = True)

sns.scatterplot(x='GrLivArea',y='SalePrice',data=train)
scatterplot(train,'GarageArea')
train=train[train['GarageArea']<1220]

train.reset_index(drop = True, inplace = True)

scatterplot(train,'GarageArea')
scatterplot(train,'TotalBsmtSF')
train=train[train['TotalBsmtSF']<2800]

train.reset_index(drop = True, inplace = True)

scatterplot(train,'TotalBsmtSF')
scatterplot(train,'1stFlrSF')
fea=['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']

sns.heatmap(train[fea].corr(),annot=True)
num_features=train.select_dtypes(include='int64' or 'float64')

num_features

discrete_features=[feature for feature in num_features if len(train[feature].unique())<25 and feature not in ['Id']]

continuous_features=[i for i in num_features if i not in discrete_features+['Id']+['SalePrice']]

#train[continuous_features]
discrete_features
train.groupby('OverallQual')['SalePrice'].median().plot.bar()

for i in continuous_features:

    train[i].hist(bins=25)

    plt.xlabel(i)

    plt.ylabel('SalePrice')

    plt.show()
for i in num_features:

    if i not in('Id'):

        print(i,"-->",train[i].skew())

high_skew=[]

for i in num_features:

    if i not in('Id') and train[i].skew()>0.5:

        print(i,"-->",train[i].skew())

        high_skew.append(i)

from sklearn.preprocessing import PowerTransformer

pt=PowerTransformer()

train[high_skew] =pt.fit_transform(train[high_skew])
from scipy import stats

import statsmodels.api as sm 

stats.probplot(train['GrLivArea'], plot=plt)

stats.probplot(train['SalePrice'], plot=plt)
cat_features=train.select_dtypes(include='object')

cat_features
chart=sns.countplot(train['Neighborhood'])

chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
sns.countplot(train['MSZoning'])
sns.boxplot(x='OverallQual',y='SalePrice',data=train)
sns.swarmplot(x='LotShape',y='SalePrice',data=train)
sns.stripplot(x='LotShape',y='SalePrice',data=train)
sns.stripplot(x='LotShape',y='SalePrice',data=train,jitter=False)
sns.violinplot(x='LotShape',y='SalePrice',data=train)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



for i in cols:

    lc= LabelEncoder() 

    train[i]=lc.fit_transform(train[i])
cat_features
from sklearn.preprocessing import OneHotEncoder

one=OneHotEncoder()

col=[i for i in cat_features if i not in('MSSubClass','OverallCond','MoSold','YrSold')]

dummy_data=pd.get_dummies(train[col],drop_first=True)
dummy_data
train=pd.concat([train,dummy_data],axis=1)

train.drop(col,axis=1,inplace=True)
train
Y=train['SalePrice']

X=train.drop(['Id','SalePrice'],axis=1)
from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel

feature_selection= SelectFromModel(Lasso(alpha=0.005, random_state=0)) 

feature_selection.fit(X,Y)
selected_feat = X.columns[(feature_selection.get_support())]

print('total features: {}'.format((X.shape[1])))

print('selected features: {}'.format(len(selected_feat)))

selected_feat