# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# Any results you write to the current directory are saved as output.
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train_data['SalePrice'].isnull().sum()
train_data['SalePrice'].value_counts()
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].median())
train_data['MiscVal']
train_data['BsmtCond']=train_data['BsmtCond'].fillna(train_data['BsmtCond'].mode()[0])

train_data['BsmtQual']=train_data['BsmtQual'].fillna(train_data['BsmtQual'].mode()[0])
train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna(train_data['FireplaceQu'].mode()[0])

train_data['GarageType'] = train_data['GarageType'].fillna(train_data['GarageType'].mode()[0])
train_data['GarageFinish'] = train_data['GarageFinish'].fillna(train_data['GarageFinish'].mode()[0])

train_data['GarageQual'] = train_data['GarageQual'].fillna(train_data['GarageQual'].mode()[0])

train_data['GarageCond'] = train_data['GarageCond'].fillna(train_data['GarageCond'].mode()[0])
train_data.drop(['GarageYrBlt','MiscFeature'],axis=1,inplace=True)

train_data.drop(['Alley','PoolQC','Fence'], axis = 1,inplace = True)
train_data.drop(['Id'], axis = 1, inplace = True)
train_data['MasVnrType']=train_data['MasVnrType'].fillna(train_data['MasVnrType'].mode()[0])

train_data['MasVnrArea']=train_data['MasVnrArea'].fillna(train_data['MasVnrArea'].mode()[0])

train_data['BsmtExposure']=train_data['BsmtExposure'].fillna(train_data['BsmtExposure'].mode()[0])

train_data['BsmtFinType2']=train_data['BsmtFinType2'].fillna(train_data['BsmtFinType2'].mode()[0])
train_data.shape
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False)
train_data.dropna(inplace = True)
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False)
train_data.shape
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',

         'Condition2','BldgType','Condition1','HouseStyle','SaleType',

        'SaleCondition','ExterCond',

         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',

         'CentralAir',

         'Electrical','KitchenQual','Functional',

         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']
len(columns)
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test['BsmtCond'].isnull().sum()
test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].median())
test['MSZoning']=test['MSZoning'].fillna(test['MSZoning'].mode()[0])
test.drop(['Alley'],axis=1,inplace=True)
test['BsmtCond']=test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])

test['BsmtQual']=test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])
test['FireplaceQu']=test['FireplaceQu'].fillna(test['FireplaceQu'].mode()[0])

test['GarageType']=test['GarageType'].fillna(test['GarageType'].mode()[0])
test.drop(['GarageYrBlt'],axis=1,inplace=True)
test['GarageFinish']=test['GarageFinish'].fillna(test['GarageFinish'].mode()[0])

test['GarageQual']=test['GarageQual'].fillna(test['GarageQual'].mode()[0])

test['GarageCond']=test['GarageCond'].fillna(test['GarageCond'].mode()[0])



test.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
test.shape
test.drop(['Id'],axis=1,inplace=True)
test['MasVnrType']=test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])

test['MasVnrArea']=test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0])
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
test['BsmtFinType2']=test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0])
test['Utilities']=test['Utilities'].fillna(test['Utilities'].mode()[0])

test['Exterior1st']=test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])

test['Exterior2nd']=test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])

test['BsmtFinType1']=test['BsmtFinType1'].fillna(test['BsmtFinType1'].mode()[0])

test['BsmtFinSF1']=test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())

test['BsmtFinSF2']=test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean())

test['BsmtUnfSF']=test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())

test['TotalBsmtSF']=test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())

test['BsmtFullBath']=test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0])

test['BsmtHalfBath']=test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0])

test['KitchenQual']=test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])

test['Functional']=test['Functional'].fillna(test['Functional'].mode()[0])

test['GarageCars']=test['GarageCars'].fillna(test['GarageCars'].mean())

test['GarageArea']=test['GarageArea'].fillna(test['GarageArea'].mean())

test['SaleType']=test['SaleType'].fillna(test['SaleType'].mode()[0])
def category_onehot_multcols(multcolumns):

    df_final=train_test

    i=0

    for fields in multcolumns:

        

        print(fields)

        df1=pd.get_dummies(train_test[fields],drop_first=True)

        

        train_test.drop([fields],axis=1,inplace=True)

        if i==0:

            df_final=df1.copy()

        else:

            

            df_final=pd.concat([df_final,df1],axis=1)

        i=i+1

       

        

    df_final=pd.concat([train_test,df_final],axis=1)

        

    return df_final
train_test=pd.concat([train_data,test],axis=0)
train_test.shape
train_test=category_onehot_multcols(columns)
train_test.shape
train_test =train_test.loc[:,~train_test.columns.duplicated()]
train_test.shape
df_Train=train_test.iloc[:1422,:]

df_Test=train_test.iloc[1422:,:]
df_Train
df_Test
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import scale
y_train = pd.DataFrame({"SalePrice": df_Train['SalePrice']})

X_train = df_Train.drop('SalePrice', axis = 1)

X_test = df_Test.drop('SalePrice', axis = 1)
alphas = 10**np.linspace(10,-2,100)*0.5
lasso = Lasso(max_iter = 10000, normalize = True)

coefs = []



for a in alphas:

    lasso.set_params(alpha=a)

    lasso.fit(scale(X_train), y_train)

    coefs.append(lasso.coef_)
ax = plt.gca()

ax.plot(alphas*2, coefs)

ax.set_xscale('log')

plt.axis('tight')

plt.xlabel('houseParameters')

plt.ylabel('SalePrice')
from sklearn.metrics import mean_squared_error
lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)

lassocv.fit(X_train, y_train)



lasso.set_params(alpha=lassocv.alpha_)

lasso.fit(X_train, y_train)
def score(y_pred, y_true):

    error = np.square(np.log10(y_pred +1) - np.log10(y_true +1)).mean() ** 0.5

    score = 1 - error

    return score
actual_cost = list(df_Train['SalePrice'])

actual_cost = np.asarray(actual_cost)
#Predicting for X_train

y_pred_lass =lasso.predict(X_train)



#Printing the Score with RMLSE

print("\n\nLasso SCORE : ", score(y_pred_lass, actual_cost))
y_test = lasso.predict(X_test)
testing = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
testing['Id']
submission = pd.DataFrame({

    "Id": testing["Id"],

    "SalePrice": y_test

})

submission.to_csv('submission.csv', index = False)

submission = pd.read_csv('submission.csv')
ridge = Ridge(normalize = True)

coef = []



for a in alphas:

    ridge.set_params(alpha = a)

    ridge.fit(scale(X_train), y_train)

    coef.append(ridge.coef_)

    

np.shape(coef)
ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)

ridgecv.fit(X_train, y_train)

ridgecv.alpha_
ridge4 = Ridge(alpha = ridgecv.alpha_, normalize = True)

ridge4.fit(X_train, y_train)
#Predicting on X_train

y_pred_ridge =ridge4.predict(X_train)



#Printing the Score with RMLSE

print("Ridge SCORE : ", score(y_pred_ridge, actual_cost))