import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.plot.scatter('GrLivArea', 'SalePrice')
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
null_values_train = train.isnull().sum()
print(null_values_train[null_values_train>0].sort_values(ascending=False))
null_values_test = test.isnull().sum()
print(null_values_test[null_values_test>0].sort_values(ascending=False))
def cleanNullValues(df,train_df):
    """
    Clean the NULL values from the dataframe df.
    
    - To impute LotFrontage we create a dictionary with Neighborhood->median LotFrontage 
    using the training set.
    - To impute attributes by most common value we use training set most common value.
    
    This prevents data leakage from the test set during imputation.
    """
    #Categorical variables
    categorical = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType','GarageFinish',
                   'GarageCond','BsmtFinType2','BsmtExposure','BsmtFinType1','BsmtCond','BsmtQual',
                   'MasVnrType']
    for col in categorical:
        df[col] = df[col].fillna('None')

    #Numerical variables
    numerical = ['GarageYrBlt','GarageQual','MasVnrArea','BsmtHalfBath','BsmtFullBath','GarageArea',
                 'GarageCars','TotalBsmtSF','BsmtUnfSF','BsmtFinSF2','BsmtFinSF1']
    for col in numerical:
        df[col] = df[col].fillna(0)
        
    #Few NULL values
    few = ['Electrical','MSZoning','Utilities','SaleType','KitchenQual','Exterior2nd','Exterior1st']
    for col in few:
        df[col] = df[col].fillna(train[col].mode()[0])
    
    #Functional
    df['Functional'] = df['Functional'].fillna('Typ')

    #LotFrontage (Adapted from https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)
    neighborhood2lotfrontage = train.loc[:,['Neighborhood','LotFrontage']].groupby(
                                       ['Neighborhood']).median()['LotFrontage'].to_dict()
    for key, _ in df.groupby("Neighborhood")["LotFrontage"]:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(neighborhood2lotfrontage[key]))
        
    return df

train = cleanNullValues(train,train)
test = cleanNullValues(test,train)

#Check that all NULL values are cleaned
print('Training set:')
print(sum(train.isnull().sum()>0))
print('Test set:')
print(sum(test.isnull().sum()>0))
train.to_csv('train_clean.csv',index=False)
test.to_csv('test_clean.csv',index=False)