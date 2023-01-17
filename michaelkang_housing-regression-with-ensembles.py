# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import skew



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.linear_model import Lasso, BayesianRidge

from sklearn import ensemble, tree, linear_model

from sklearn.metrics import log_loss, mean_squared_error

from sklearn.ensemble import RandomForestRegressor



import xgboost as xgb
#Importing data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/train.csv')
#Documentation for the Ames Housing Data indicates that there are outliers present in the data

#http://ww2.amstat.org/publications/jse/v19n3/Decock/DataDocumentation.txt

fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'])

#We find that there are two outliers with extremely large GrLivArea that are of a low price (as the documentation indicates)

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'])
#Concatenating into one dateframe

data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))
#Dropping any features with the majority of examples missing

data = data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)
#Make a correlation map to determine which features are not very correlated with SalePrice

corrmat = train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
#Now dropping numerical features that aren't correlated with Saleprice

data = data.drop(['MasVnrArea','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'LowQualFinSF',

                  'BsmtFullBath', 'BsmtHalfBath', 'KitchenAbvGr', 'EnclosedPorch', 'GarageYrBlt',

                  'Functional', 'GarageArea', '3SsnPorch', 'MiscVal'], axis=1)

#Dropping categorical features that aren't correlated with Saleprice

data = data.drop(['Utilities'], axis=1)
#Now, proceeding through the data sequentially. Using value_counts() for categorical and describe() for numerical

#MSSubClass=The building class. This should be translated into a categorical value. Could also use LabelEncoder

data['MSSubClass'] = data['MSSubClass'].apply(str)



#MSZoning=The general zoning classification. 'RL' is the most common by far. Filling missing values with 'RL'

data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])



#LotFrontage=Linear feet of street connected to property. Assuming NA implies no street. Might there be a non-zero amount of feet connected to a property?

data['LotFrontage'] = data['LotFrontage'].fillna(0)



#Changing OverallCond into a categorical variable

data['OverallCond'] = data['OverallCond'].astype(str)



#Both Exterior 1 & 2 have only one missing value. Will just substitute in the most common string

data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])



#MasVnrType=Masonry veneer type. Missing values are likely because none. Should be the most common value ('None')

data['MasVnrType'] = data['MasVnrType'].fillna(data['MasVnrType'].mode()[0])



#For all these categorical basement-related features, NaN means that there isn't a basement

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    data[col] = data[col].fillna('No Garage')



#TotalBsmtSF's missing values are likely zero for having no basement

data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(0)



#Electrical has mostly 'SBrkr'. Setting that for the missing values.

data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])



#Replacing missing values in KitchenQual with the most frequent

data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])



#Same as for the basement with the garage. Replacing missing data with 'No Garage'

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    data[col] = data[col].fillna('No Garage')



#GarageCars=# of cars. No garage = no cars in said garage.

data['GarageCars'] = data['GarageCars'].fillna(0)



#SaleType=Type of sale. Fill with most frequent

data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])



#Year and month sold are transformed into categorical features.

data['YrSold'] = data['YrSold'].astype(str)

data['MoSold'] = data['MoSold'].astype(str)



# Adding total sqfootage feature and removing Basement, 1st and 2nd floor features

data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

data.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)
total = data.isnull().sum().sort_values(ascending=False)

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head()
#SalePrice is skewed left. Taking the log to try to normalize.

Sales = sns.distplot(train['SalePrice'])

train['SalePrice'] = np.log(train['SalePrice'])
Sales = sns.distplot(train['SalePrice'])

#We notice that some of the numerical values are also a little skewed.

numeric_feats = data.dtypes[data.dtypes != "object"].index         

skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

data[skewed_feats] = np.log1p(data[skewed_feats])
#Getting dummies!. Bumps feature count from 57 --> 282

data = pd.get_dummies(data)
#Pulling train/test sets out from concatenated set. Needed to have put them together as some categorical variables dont appear in both sets.

X_train = data[:train.shape[0]]

X_test = data[train.shape[0]:]

Y_train = train.SalePrice
#Now training models

x1, x2, y1, y2 = train_test_split(X_train, Y_train, train_size=0.75, random_state=42)
#Ridge

#alphas=[0.1,0.001,0.0001,1,2,3,4,5,6,7,8,9,10,11,12]

#model = Ridge(random_state=2)

#grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))

#grid.fit(X_train, Y_train)

#print(grid.best_score_)

#print(grid.best_estimator_.alpha)

ridge = Ridge(alpha = 5, random_state=2).fit(x1, y1)

score_ridge = ridge.score(x2, y2)
#Lasso

#alphas=[0.3, 0.1, 0.03, 0.001, 0.003, 0.0001,1,2,3,4,5,6,7,8,9,10,11,12]

#model = Lasso(random_state=17)

#grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))

#grid.fit(X_train, Y_train)

#print(grid.best_score_)

#print(grid.best_estimator_.alpha)

lasso = Lasso(alpha = 0.001, random_state=23).fit(x1, y1)

score_lasso = lasso.score(x2, y2)
#Elastic Net

ENSTest = linear_model.ElasticNetCV(alphas=[0.0001,0.0003, 0.0005, 0.001, 0.03, 0.01, 0.3, 0.1, 3, 1, 10, 30], l1_ratio=[.01, 0.3, .1, .3, .5, .9, .99], max_iter=5000, random_state=3).fit(x1, y1)

score_EN = ENSTest.score(x2, y2)
#Random Forest

random_forest = RandomForestRegressor(n_estimators=2900, random_state=11).fit(x1,y1)

score_forest = random_forest.score(x2, y2)
#Gradient Boosting

GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',

                                               min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=5).fit(x1, y1)

score_GBest = GBest.score(x2, y2)
#XGBoosting

#dtrain = xgb.DMatrix(X_train, label = Y_train)

#dtest = xgb.DMatrix(X_test)

#params = {"max_depth":6, "eta":0.1}

#model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

#model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=6, learning_rate=0.1, seed=7) #the params were tuned using xgb.cv

model_xgb.fit(x1, y1)

score_xgb = model_xgb.score(x2, y2)
#Found that random forest wasn't very accurate and dropped the score consistently.

scores = (score_ridge + score_lasso + score_EN+ score_GBest + score_xgb)/5
#Now we take the average of these models fit on all the data and submit!

ridge = Ridge(alpha = 5, random_state=2).fit(X_train, Y_train)

pred_ridge = ridge.predict(X_test)



lasso = Lasso(alpha = 0.001, random_state=23).fit(X_train, Y_train)

pred_lasso = lasso.predict(X_test)



ENSTest = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000, random_state=3).fit(X_train, Y_train)

pred_EN = ENSTest.predict(X_test)



random_forest = RandomForestRegressor(n_estimators=2900, random_state=11).fit(X_train, Y_train)

pred_forest = random_forest.predict(X_test)



GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',

                                               min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=5).fit(X_train, Y_train)

pred_GBest = GBest.predict(X_test)



model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=6, learning_rate=0.1, seed=7) #the params were tuned using xgb.cv

model_xgb.fit(X_train, Y_train)

pred_xgb = model_xgb.predict(X_test)



pred = (pred_ridge + pred_EN + pred_lasso+ pred_GBest + pred_xgb)/5

Y_pred = np.exp(pred)

#pd.DataFrame({'Id': np.arange(Y_pred.shape[0])+1461, 'SalePrice': Y_pred}).to_csv('C:/Users/Michael Kang/Desktop/Data_Files/Housing/Housing_predictions5.csv', index =False)
