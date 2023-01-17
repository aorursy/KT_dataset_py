import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

sample_sub = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
train.head(10)
train.columns
test.head(10)
train.shape
train.info()
train.isnull().sum()
test.info()
test.isnull().sum()
train['MSZoning'].value_counts()
sns.heatmap(train.isnull(),yticklabels = False, cbar= False)
# Filling missing value

train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
train.drop(['Alley'], axis=1, inplace=True)
train['BsmtQual'] = train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])

train['BsmtCond'] = train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])
train['FireplaceQu'] = train['FireplaceQu'].fillna(train['FireplaceQu'].mode()[0])

train['GarageType'] = train['GarageType'].fillna(train['GarageType'].mode()[0])
train.drop(['GarageYrBlt'], axis = 1, inplace=True)
train['GarageFinish'] = train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])

train['GarageQual'] = train['GarageQual'].fillna(train['GarageQual'].mode()[0])

train['GarageCond'] = train['GarageCond'].fillna(train['GarageCond'].mode()[0])
train.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace= True)
train.shape
test.shape
train.drop(['Id'], axis=1, inplace=True)
train.isnull().sum()
train['MasVnrType']=train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])

train['MasVnrArea']=train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0])
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='coolwarm')
train['BsmtExposure'] = train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')
train['BsmtFinType2'] = train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])
train.dropna(inplace=True)
train.shape
test.head()
#check null values

test.isnull().sum()
## Fill Missing Values



test['LotFrontage']=test['LotFrontage'].fillna(test['LotFrontage'].mean())



test['MSZoning']=test['MSZoning'].fillna(test['MSZoning'].mode()[0])
test.shape
test.drop(['Alley'],axis=1,inplace=True)
test.shape
test['BsmtCond']=test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])

test['BsmtQual']=test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])

test['FireplaceQu']=test['FireplaceQu'].fillna(test['FireplaceQu'].mode()[0])

test['GarageType']=test['GarageType'].fillna(test['GarageType'].mode()[0])


test.drop(['GarageYrBlt'],axis=1,inplace=True)
test.shape
test['GarageFinish']=test['GarageFinish'].fillna(test['GarageFinish'].mode()[0])

test['GarageQual']=test['GarageQual'].fillna(test['GarageQual'].mode()[0])

test['GarageCond']=test['GarageCond'].fillna(test['GarageCond'].mode()[0])



test.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
test.shape
test.drop(['Id'],axis=1,inplace=True)
test['MasVnrType']=test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])

test['MasVnrArea']=test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0])
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
test['BsmtExposure']=test['BsmtExposure'].fillna(test['BsmtExposure'].mode()[0])
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
test['BsmtFinType2']=test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0])
test.loc[:, test.isnull().any()].head()
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
test.shape
columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2','BldgType','HouseStyle', 'SaleType',

          'SaleCondition', 'ExterCond',

          'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive' 

        ]
len(columns)
# Csonverting Categorical Features into dummy variables

def category_onehot_multcols(multcolumns):

    train_final = final_train

    i=0

    for fields in multcolumns:

        print(fields)

        train1 = pd.get_dummies(final_train[fields], drop_first=True)

        

        final_train.drop([fields],axis=1, inplace=True)

        if i==0:

            train_final = train1.copy()

        else:

            

            train_final = pd.concat([train_final, train1], axis=1)

        i=i+1 

        

    train_final = pd.concat([final_train, train_final], axis = 1)

    

    return train_final

    

        
main_train = train.copy


test.shape
test.head()
final_train = pd.concat([train,test], axis = 0)
final_train['SalePrice']
final_train.shape
final_train= category_onehot_multcols(columns)
final_train = final_train.loc[:, ~final_train.columns.duplicated()]
final_train.shape

train_df = final_train.iloc[:1422,:]

test_df = final_train.iloc[1422:,:]
train_df.head()
test_df.head()
train_df.shape
test_df.drop(['SalePrice'],axis=1, inplace = True)
X_train = train_df.drop(['SalePrice'],axis=1)

y_train = train_df['SalePrice']
test_df.head()
import xgboost

classifier=xgboost.XGBRegressor()
import xgboost

regressor=xgboost.XGBRegressor()
booster=['gbtree','gblinear']

base_score=[0.25,0.5,0.75,1]
# Hyper Parameter Optimization



n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }
# Set up the random search with 4-fold cross validation

from sklearn.model_selection import RandomizedSearchCV

random_cv = RandomizedSearchCV(estimator=regressor,

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_absolute_error',n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42)
random_cv.fit(X_train, y_train)
random_cv.best_estimator_
regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,

       max_depth=2, min_child_weight=1, missing=None, n_estimators=900,

       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,

       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

       silent=True, subsample=1)
regressor.fit(X_train,y_train)
import pickle

filename = 'finalized_model.pkl'

pickle.dump(classifier, open(filename, 'wb'))
y_pred = regressor.predict(test_df)
y_pred
rf_pred = regressor.predict(test_df)
# create sample submission file and submit

prep = pd.DataFrame(y_pred)

sub_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

datasets = pd.concat([sub_df['Id'],prep], axis=1)

datasets.columns = ['Id', 'SalePrice']

datasets.to_csv('sample_submission.csv', index=False)