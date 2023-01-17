# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

test_read = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_read.head()
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df.head()
df.info()
test_read.info()
#visualise null values

sns.heatmap(df.isnull(), yticklabels=False, cbar='False')
sns.heatmap(test_read.isnull(), yticklabels=False, cbar='False')
#outlier removal

fig, ax = plt.subplots()

ax.scatter(x = df['GrLivArea'], y = df['SalePrice'], c='pink')

plt.ylabel('SalePrice', fontsize=9)

plt.xlabel('GrLivArea', fontsize=9)

plt.show()
#df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)



fig, ax = plt.subplots()

ax.scatter(df['GrLivArea'], df['SalePrice'], c='pink')

plt.ylabel('SalePrice', fontsize=9)

plt.xlabel('GrLivArea', fontsize=9)

plt.show()
df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)

df['MasVnrType'].fillna(df['MasVnrType'].mode()[0], inplace=True)

df['MasVnrArea'].fillna(df['MasVnrArea'].mean(), inplace=True)

df['BsmtQual'].fillna(df['BsmtQual'].mode()[0], inplace=True)

df['BsmtCond'].fillna(df['BsmtCond'].mode()[0], inplace=True)

df['GrLivArea'].fillna(df['GrLivArea'].mean(), inplace=True)

df['SalePrice'].fillna(df['SalePrice'].mean(), inplace=True)

df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0], inplace=True)

df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0], inplace=True)

df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0], inplace=True)

df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)

df['GarageType'].fillna(df['GarageType'].mode()[0], inplace=True)

df['GarageYrBlt'].fillna(df['GarageYrBlt'].mode()[0], inplace=True)

df['GarageFinish'].fillna(df['GarageFinish'].mode()[0], inplace=True)

df['GarageQual'].fillna(df['GarageQual'].mode()[0], inplace=True)

df['GarageCond'].fillna(df['GarageCond'].mode()[0], inplace=True)

df.drop(['PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'Alley', 'Id'], axis=1, inplace=True) #drop, too many null values
#repeat for test_df

test_read['LotFrontage'].fillna(test_read['LotFrontage'].mean(), inplace=True)

test_read['MasVnrType'].fillna(test_read['MasVnrType'].mode()[0], inplace=True)

test_read['MasVnrArea'].fillna(test_read['MasVnrArea'].mean(), inplace=True)

test_read['BsmtQual'].fillna(test_read['BsmtQual'].mode()[0], inplace=True)

test_read['BsmtCond'].fillna(test_read['BsmtCond'].mode()[0], inplace=True)

test_read['BsmtExposure'].fillna(test_read['BsmtExposure'].mode()[0], inplace=True)

test_read['BsmtFinType1'].fillna(test_read['BsmtFinType1'].mode()[0], inplace=True)

test_read['BsmtFinType2'].fillna(test_read['BsmtFinType2'].mode()[0], inplace=True)

test_read['Electrical'].fillna(test_read['Electrical'].mode()[0], inplace=True)

test_read['GarageType'].fillna(test_read['GarageType'].mode()[0], inplace=True)

test_read['GarageYrBlt'].fillna(test_read['GarageYrBlt'].mode()[0], inplace=True)

test_read['GarageFinish'].fillna(test_read['GarageFinish'].mode()[0], inplace=True)

test_read['GarageQual'].fillna(test_read['GarageQual'].mode()[0], inplace=True)

test_read['GarageCond'].fillna(test_read['GarageCond'].mode()[0], inplace=True)

test_read['MSZoning'].fillna(test_read['MSZoning'].mode()[0], inplace=True)

test_read['Utilities'].fillna(test_read['Utilities'].mode()[0], inplace=True)

test_read['Exterior1st'].fillna(test_read['Exterior1st'].mode()[0], inplace=True)

test_read['Exterior2nd'].fillna(test_read['Exterior2nd'].mode()[0], inplace=True)

test_read['BsmtFinSF1'].fillna(test_read['BsmtFinSF1'].mean(), inplace=True)

test_read['BsmtFinSF2'].fillna(test_read['BsmtFinSF2'].mean(), inplace=True)

test_read['BsmtUnfSF'].fillna(test_read['BsmtUnfSF'].mean(), inplace=True)

test_read['TotalBsmtSF'].fillna(test_read['TotalBsmtSF'].mean(), inplace=True)

test_read['BsmtFullBath'].fillna(test_read['BsmtFullBath'].mean(), inplace=True)

test_read['BsmtHalfBath'].fillna(test_read['BsmtHalfBath'].mean(), inplace=True)

test_read['KitchenQual'].fillna(test_read['KitchenQual'].mode()[0], inplace=True)

test_read['GarageCars'].fillna(test_read['GarageCars'].mean(), inplace=True)

test_read['Functional'].fillna(test_read['Functional'].mode()[0], inplace=True)

test_read['GarageArea'].fillna(test_read['GarageArea'].mean(), inplace=True)

test_read['SaleType'].fillna(test_read['SaleType'].mode()[0], inplace=True)

test_read.drop(['PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'Alley', 'Id'], axis=1, inplace=True) #drop, too many null values
sns.heatmap(test_read.isnull(), yticklabels=False, cbar='False')
sns.heatmap(df.isnull(), yticklabels=False, cbar='False')
#columns with categorical values

columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig',

         'Condition2','BldgType','Condition1','HouseStyle','SaleType',

         'SaleCondition','ExterCond','ExterQual','Foundation','BsmtQual',

         'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

         'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',    

         'CentralAir', 'Electrical','KitchenQual','Functional','GarageType',

         'GarageFinish','GarageQual','GarageCond','PavedDrive','LandSlope','Neighborhood']



len(columns)
df.shape
final_df = pd.concat([df,test_read],axis=0)
final_df.shape
final_df = pd.get_dummies(final_df, columns=columns, drop_first=True)

final_df
final_df.shape
#drop duplicates

final_df = final_df.loc[:,~final_df.columns.duplicated()]
final_df.drop_duplicates(inplace=True)
final_df.shape
df=final_df.iloc[:1460,:]

test_df=final_df.iloc[1460:,:]

test_df.drop(['SalePrice'],axis=1,inplace=True)
test_df.shape
df.shape
#correlation matrix

corrmat = df.corr()

corrmat
df.head()
x = df.drop(['SalePrice'], axis=1)

y = df['SalePrice']



#log transform

y= np.log1p(y)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)                                                
from sklearn.model_selection import GridSearchCV

import joblib

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

warnings.filterwarnings('ignore', category=DeprecationWarning)
def print_results(results):

    print('BEST PARAMS: {}\\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']

    stds = results.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, results.cv_results_['params']):

        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)

lr.score(X_train, y_train)
#feature scaling for ridge

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_scaled = sc.transform(X_train)
from sklearn.linear_model import Ridge

ridge = Ridge()

ridge.fit(X_scaled, y_train)

ridge.score(X_scaled, y_train)
import xgboost

xgb_reg = xgboost.XGBRegressor(random_state=42)

xgb_reg.fit(X_train, y_train)

print(xgb_reg.score(X_train, y_train))



y_pred=xgb_reg.predict(test_df)

y_pred
y_pred[y_pred<0] = 0

y_pred = np.expm1(y_pred)

y_pred
#make submission df

prediction = pd.DataFrame(y_pred)

submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

prediction_df = pd.concat([submission['Id'], prediction], axis=1)

prediction_df.columns=['Id','SalePrice']

prediction_df.to_csv('sample_submission.csv',index=False)