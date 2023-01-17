import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train_test=pd.concat([train,test],axis=0,sort=False)
train_test.head()
pd.set_option('display.max_rows', 5000)
train_test_null_info=pd.DataFrame(train_test.isnull().sum(),columns=['Count of NaN'])
train_test_dtype_info=pd.DataFrame(train_test.dtypes,columns=['DataTypes'])
train_tes_info=pd.concat([train_test_null_info,train_test_dtype_info],axis=1)
train_tes_info
g = {str(k): list(v) for k,v in train_test.groupby(train_test.dtypes, axis=1)}
g
numeric=['LotFrontage','GarageCars','GarageArea','TotalBsmtSF','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath']
categorical = ['KitchenQual','Exterior1st','SaleType','Exterior2nd','Functional','Utilities','MSZoning']
for i in categorical:
    train_test[i].fillna(train_test[i].value_counts().to_frame().index[0], inplace=True)
for i in numeric:
    train_test[i].fillna(train_test[i].mean(),inplace=True)
for i in range(81):    
    if train_test[train_test.columns[i]].dtype=='O':
        train_test[train_test.columns[i]]=train_test[train_test.columns[i]].astype(str)
    train_test.plot(kind='scatter',x=train_test.columns[i],y='SalePrice')
    plt.xlabel(train_test.columns[i])
    plt.ylabel('Price')
    plt.title('Interesting graph')
relevant_fields=['OverallQual','YearBuilt','TotalBsmtSF','GrLivArea','LotFrontage','1stFlrSF']
for i in relevant_fields:
    train_test.plot(kind='scatter',x=i,y='SalePrice')
    plt.xlabel(i)
    plt.ylabel('Price')
    plt.title('Interesting graph')
train=train_test[0:1460]
test=train_test[1460:2919]
train_y = train.SalePrice
predictor_cols = ['OverallQual','YearBuilt','TotalBsmtSF','GrLivArea','LotFrontage','1stFlrSF']

# Create training predictors data
train_X = train[predictor_cols]

my_model = linear_model.LinearRegression()
my_model.fit(train_X, train_y)
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)