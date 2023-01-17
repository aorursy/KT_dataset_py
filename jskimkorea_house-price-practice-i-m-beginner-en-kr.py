# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sample_submission=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
test.head()
print(train.info(), '\n', '-'*50)

print(test.info())
y_train=train['SalePrice']
# Drop 'Id' columns in train, test('Id' 컬럼을 제거 합니다.)

train_Id=train['Id']

test_Id=test['Id']

train.drop('Id', axis=1, inplace=True)

test.drop('Id', axis=1, inplace=True)
for col in ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']:

    train[col]=train[col].astype('object')

    test[col]=test[col].astype('object')
# Correlation coefficient(상관계수)

num_col=train.select_dtypes(exclude='object')



num_col_corr=num_col.corr()



f, ax=plt.subplots(figsize=(30,30))

sns.heatmap(num_col_corr, annot=True, ax=ax)
num_col_corr['SalePrice'].sort_values(ascending=False)
# Only select columns with a correlation of 0.3 or higher.(상관계수가 0.3 이상인 컬럼들만 골라냅니다.) 

high_corr_num_col=[]

for col in list(num_col_corr['SalePrice'].index):

    if (abs(num_col_corr['SalePrice'][col])>0.3):

        high_corr_num_col.append(col)

        

high_corr_num_col.remove('SalePrice')
high_corr_num_col
object_col=train.select_dtypes('object')
object_col.head()
f, axes=plt.subplots(10,5, figsize=(30,50))

ax=axes.ravel()

for i, col in enumerate(object_col.columns):

    sns.boxplot(x=object_col[col], y=train['SalePrice'], ax=ax[i])
high_corr_object_col=['MSZoning','Alley','LotShape','LandContour','Neighborhood','Condition1','Condition2','OverallCond','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual',

'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','FireplaceQu',

'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','MiscFeature','SaleType','SaleCondition']
print('numberical columns : ',high_corr_num_col)

print('categorical columns : ', high_corr_object_col)
# Combine the two columns list.(두 컬럼 리스트를 합칩니다.)

corr_columns=high_corr_num_col+high_corr_object_col
train[corr_columns].head()
test[corr_columns].head()
ntrain=train.shape[0]

ntest=test.shape[0]



all_data = pd.concat((train[corr_columns], test[corr_columns])).reset_index(drop=True)
all_data.shape
all_data
pd.DataFrame({'columns':all_data.isnull().sum().sort_values(ascending=False).index,

             'missing data':all_data.isnull().sum().sort_values(ascending=False).values/all_data.shape[0]})
missing_data_col=[col for col in all_data.isnull().sum().sort_values(ascending=False).index[:27]]
missing_data_col
all_data.drop(['PoolQC','MiscFeature','Alley'], axis=1, inplace=True)
all_data['FireplaceQu'].fillna('None', inplace=True)

all_data['GarageCond'].fillna('None', inplace=True)

all_data['GarageQual'].fillna('None', inplace=True)

all_data['GarageFinish'].fillna('None', inplace=True)

all_data['GarageType'].fillna('None', inplace=True)

all_data['BsmtExposure'].fillna('None', inplace=True)

all_data['BsmtCond'].fillna('None', inplace=True)

all_data['BsmtQual'].fillna('None', inplace=True)

all_data['BsmtFinType2'].fillna('None', inplace=True)

all_data['KitchenQual'].fillna('None', inplace=True)

all_data['SaleType'].fillna('None', inplace=True)

all_data['Exterior1st'].fillna('None', inplace=True)

all_data['Electrical'].fillna('None', inplace=True)

all_data['BsmtFinType1'].fillna('None', inplace=True)

all_data['MasVnrType'].fillna('None', inplace=True)

all_data['Exterior2nd'].fillna('None', inplace=True)

all_data['MSZoning'].fillna('None', inplace=True)

all_data['LotFrontage'].fillna(all_data['LotFrontage'].median(), inplace=True)

all_data['GarageYrBlt'].fillna(all_data['GarageYrBlt'].median(), inplace=True)

all_data['GarageCars'].fillna(all_data['GarageCars'].median(), inplace=True)

all_data['TotalBsmtSF'].fillna(all_data['TotalBsmtSF'].median(), inplace=True)

all_data['BsmtFinSF1'].fillna(all_data['BsmtFinSF1'].median(), inplace=True)

all_data['GarageArea'].fillna(all_data['GarageArea'].median(), inplace=True)

all_data['MasVnrArea'].fillna(all_data['MasVnrArea'].median(), inplace=True)
all_data=pd.get_dummies(all_data)
all_data.head()
# Isolate the train, test data.(train, test 데이터를 분리합니다.)

train=all_data[:ntrain]

test=all_data[ntrain:]
sns.distplot(y_train)
y_train=np.log(y_train)

sns.distplot(y_train)
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score, train_test_split
linear_regression=LinearRegression(n_jobs=-1)

svr=make_pipeline(RobustScaler(), SVR())

forest=RandomForestRegressor(random_state=42)

GDR=GradientBoostingRegressor(random_state=42)



print('LinearRegression score : ', cross_val_score(linear_regression, train.values, y_train, cv=KFold(5, shuffle=True, random_state=42)).mean())

print('svr score : ', cross_val_score(svr, train.values, y_train, cv=KFold(5, shuffle=True, random_state=42)).mean())

print('forest score : ', cross_val_score(forest, train.values, y_train, cv=KFold(5, shuffle=True, random_state=42)).mean())

print('GDR score : ', cross_val_score(GDR, train.values, y_train, cv=KFold(5, shuffle=True, random_state=42)).mean())
forest.fit(train.values, y_train)
predict=forest.predict(test.values)

predict=np.exp(predict)
sub=sample_submission

sub['SalePrice']=predict

sub.to_csv('submission.csv', index=False)