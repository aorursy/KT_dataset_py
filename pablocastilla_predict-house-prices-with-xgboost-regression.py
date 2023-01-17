import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost

import csv as csv

from xgboost import plot_importance

from matplotlib import pyplot

from sklearn.model_selection import cross_val_score,KFold

from sklearn.cross_validation import  train_test_split

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

from sklearn.grid_search import GridSearchCV   #Perforing grid search

from scipy.stats import skew

from collections import OrderedDict
train_dataset=pd.read_csv('../input/train.csv', header=0)

test_dataset=pd.read_csv('../input/test.csv', header=0)
categorical_features=['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities',

                      'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',

                      'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond',

                      'Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating',

                      'HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu',

                     'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence',

                     'MiscFeature','SaleType','SaleCondition']

every_column_except_y= [col for col in train_dataset.columns if col not in ['SalePrice','Id']]

train_dataset.describe()
#thanks to https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models



every_column_non_categorical= [col for col in train_dataset.columns if col not in categorical_features and col not in ['Id'] ]



#log transform skewed numeric features:

numeric_feats = train_dataset[every_column_non_categorical].dtypes[train_dataset.dtypes != "object"].index



#skewed_feats = train_dataset[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

#skewed_feats = skewed_feats[skewed_feats > 0.75]

#skewed_feats = skewed_feats.index



train_dataset[numeric_feats] = np.log1p(train_dataset[numeric_feats])



every_column_non_categorical= [col for col in test_dataset.columns if col not in categorical_features and col not in ['Id'] ]

numeric_feats = test_dataset[every_column_non_categorical].dtypes[test_dataset.dtypes != "object"].index

test_dataset[numeric_feats] = np.log1p(test_dataset[numeric_feats])
features_with_nan=['Alley','MasVnrType','BsmtQual','BsmtQual','BsmtCond','BsmtCond','BsmtExposure',

                   'BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish']

#function that creates a column for every value it might have

def ConverNaNToNAString(data, columnList):

    for x in columnList:       

        data[x] =str(data[x])              

            



ConverNaNToNAString(train_dataset, features_with_nan)

ConverNaNToNAString(test_dataset, features_with_nan)
#function that creates a column for every value it might have

def CreateColumnPerValue(data, columnList):

    for x in columnList:



        values=pd.unique(data[x])

        

        for v in values:

            column_name=x+"_"+str(v)   

            data[column_name]=(data[x]==v).astype(float)

    

        data.drop(x, axis=1, inplace=True)

        
#CreateColumnPerValue(train_dataset,categorical_features)

#CreateColumnPerValue(test_dataset,categorical_features)



train_dataset = pd.get_dummies(train_dataset,columns =categorical_features)

test_dataset = pd.get_dummies(test_dataset,columns =categorical_features)



#train_dataset = train_dataset.fillna(train_dataset.mean())

#test_dataset = test_dataset.fillna(test_dataset.mean())
model = xgboost.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.07,

                 max_depth=3,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42) 
every_column_except_y= [col for col in train_dataset.columns if col not in ['SalePrice','Id']]

model.fit(train_dataset[every_column_except_y],train_dataset['SalePrice'])



OrderedDict(sorted(model.booster().get_fscore().items(), key=lambda t: t[1], reverse=True))
#Let's remove the less important ones 

most_relevant_features= list( dict((k, v) for k, v in model.booster().get_fscore().items() if v >= 10).keys())

print(most_relevant_features)
plt.scatter(train_dataset.GrLivArea, train_dataset.SalePrice, c = "blue", marker = "s")

plt.title("GrLivArea vs SalePrice")

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")

plt.show()
plt.scatter(train_dataset.LotArea, train_dataset.SalePrice, c = "blue", marker = "s")

plt.title("LotArea vs SalePrice")

plt.xlabel("LotArea")

plt.ylabel("SalePrice")

plt.show()
#removing outliers

train_dataset = train_dataset[train_dataset.GrLivArea < 8.25]

train_dataset = train_dataset[train_dataset.LotArea < 11.5]

train_dataset = train_dataset[train_dataset.SalePrice<13]

train_dataset = train_dataset[train_dataset.SalePrice>10.75]

train_dataset.drop("Id", axis=1, inplace=True)
train_x=train_dataset[most_relevant_features]

train_y=train_dataset['SalePrice']



#for tuning parameters

#parameters_for_testing = {

#    'colsample_bytree':[0.4,0.6,0.8],

#    'gamma':[0,0.03,0.1,0.3],

#    'min_child_weight':[1.5,6,10],

#    'learning_rate':[0.1,0.07],

#    'max_depth':[3,5],

#    'n_estimators':[10000],

#    'reg_alpha':[1e-5, 1e-2,  0.75],

#    'reg_lambda':[1e-5, 1e-2, 0.45],

#    'subsample':[0.6,0.95]  

#}



                    

#xgb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,

#     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)



#gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6,iid=False, verbose=10,scoring='neg_mean_squared_error')

#gsearch1.fit(train_x,train_y)

#print (gsearch1.grid_scores_)

#print('best params')

#print (gsearch1.best_params_)

#print('best score')

#print (gsearch1.best_score_)



best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.07,

                 max_depth=3,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42)

best_xgb_model.fit(train_x,train_y)
test_dataset['Prediction'] = np.expm1(best_xgb_model.predict(test_dataset[most_relevant_features]))

filename = 'submission.csv'

pd.DataFrame({'Id': test_dataset.Id, 'SalePrice': test_dataset.Prediction}).to_csv(filename, index=False)
print(test_dataset['Prediction'].head())

print(test_dataset['Prediction'].count())