# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import random

from pandas.plotting import scatter_matrix



# Misc

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA



# Ensemble Models

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



# Package for stacking models

from vecstack import stacking



# Stats

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)



import warnings

warnings.filterwarnings("ignore")

# Ignore warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print ("Train data shape:", train.shape)

print ("Test data shape:", test.shape)
# Taking a peak of the train data set



# Data Dictionary

# SalePrice — the property’s sale price in dollars. This is the target variable that you’re trying to predict.

# MSSubClass — The building class

# MSZoning — The general zoning classification

# LotFrontage — Linear feet of street connected to property

# LotArea — Lot size in square feet

# Street — Type of road access

# Alley — Type of alley access

# LotShape — General shape of property

# LandContour — Flatness of the property

# Utilities — Type of utilities available

# LotConfig — Lot configuration



train.head()
# Taking a peak of the test data set



test.head()
test.dtypes
test.describe()
train.describe()
train.YearBuilt.describe()
train.SalePrice.describe()
print ("Skew is:", train.SalePrice.skew())

fig = plt.figure()

plt.hist(train.SalePrice, color='blue')

fig.suptitle('Sale Price of Homes')

plt.xlabel('Price')

plt.ylabel('Number of Homes Sold')

plt.show()
target = np.log(train.SalePrice)

print ("Skew is:", target.skew())

fig = plt.figure()

plt.hist(target, color='blue')

fig.suptitle('Sale Price of Homes')

plt.xlabel('Price')

plt.ylabel('Number of Homes Sold')

plt.show()
numeric_features = train.select_dtypes(include=[np.number])

numeric_features.dtypes
# correlation heat map setup for seaborn

# Compare the correlation between the first ten features



corr = numeric_features.corr()

corr['SalePrice'].sort_values(ascending=False)[:10]
# Compare the correlation between the next ten features after



corr['SalePrice'].sort_values(ascending=False)[-10:]
quality_pivot = train.pivot_table(index='OverallQual',

                  values='SalePrice', aggfunc=np.median)



quality_pivot
#visualizing the relationship between overall quality and median sale price.



quality_pivot.plot(kind='bar', color='blue')

plt.xlabel('Overall Quality')

plt.ylabel('Median Sale Price')

plt.show()



#Visualizing the relationship between overall grade living area square feet to sale price.



plt.scatter(x=train['GrLivArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Above grade (ground) living area square feet')

plt.show()



#Visualizing the relationship between overall garage area square feet to sale price.



plt.scatter(x=train['GarageArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
#Removed outliers for GarageArea and GrLivArea



train = train[train['GarageArea'] < 1100]

train = train[train['GrLivArea'] < 4500]
#Visualizing the output again without the outliers



plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))

plt.xlim(-200,1600) # This forces the same scale as before

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()



plt.scatter(x=train['GrLivArea'], y=np.log(train.SalePrice))

plt.xlim(-200,5000) # This forces the same scale as before

plt.ylabel('Sale Price')

plt.xlabel('Above grade (ground) living area square feet')

plt.show()
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'

nulls
print ("Unique values are:", train.MiscFeature.unique())
categoricals = train.select_dtypes(exclude=[np.number])

categoricals.describe()
train.Street.value_counts()
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
train.enc_street.value_counts()
condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='green')

plt.xlabel('Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
#Hot encoding SaleCondition to partial by assigning either a 1 or 0.



def encode(x):

 return 1 if x == 'Partial' else 0

train['enc_condition'] = train.SaleCondition.apply(encode)

test['enc_condition'] = test.SaleCondition.apply(encode)
#Visualizing the new feature that was encoded.



condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='red')

plt.xlabel('Encoded Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
# We want to clean any missing data before we begin modeling.



data = train.select_dtypes(include=[np.number]).interpolate().dropna()



sum(data.isnull().sum() != 0)
y = np.log(train.SalePrice)

X = data.drop(['SalePrice', 'Id'], axis=1)
# Create training and test dataset



y = np.log(train.SalePrice)

X = data.drop(['SalePrice', 'Id'], axis=1)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

                          X, y, random_state=50, test_size=.25)
from sklearn import linear_model

from sklearn.model_selection import GridSearchCV



lr = linear_model.LinearRegression()

lr_model = lr.fit(X_train, y_train)

print ("R^2 is: \n", lr_model.score(X_test, y_test))



lr_predictions = lr_model.predict(X_test)



#importing sklearn metrics and RMSE function. RMSE measures the distance between or predicted values and actual values.



from sklearn.metrics import mean_squared_error

print ('MSE is: \n', mean_squared_error(y_test, lr_predictions))

print ('RMSE is: \n', np.sqrt(mean_squared_error(y_test, lr_predictions)))



print ('cross_val_score: \n', cross_val_score(lr_model, X_train, y_train, cv=10).mean())
plt.figure(figsize=(15,8))

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

sns.regplot(y_test, lr_predictions, color='red')

plt.show()
rm = linear_model.Ridge()

ridge_model = rm.fit(X_train, y_train)

preds_ridge = ridge_model.predict(X_test)



print ("Model Score (R^2) is: \n", ridge_model.score(X_test, y_test))



rm_predictions = ridge_model.predict(X_test)



#importing sklearn metrics and RMSE function. RMSE measures the distance between or predicted values and actual values.



from sklearn.metrics import mean_squared_error

print ('MSE is: \n', mean_squared_error(y_test, rm_predictions))

print ('RMSE is: \n', np.sqrt(mean_squared_error(y_test, rm_predictions)))



print ('cross_val_score: \n', cross_val_score(ridge_model, X_train, y_train, cv=10).mean())
plt.figure(figsize=(15,8))

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

sns.regplot(y_test, rm_predictions, color='black')

plt.show()
from sklearn.linear_model import Lasso



lasso = linear_model.Lasso()

lasso_model = lasso.fit(X_train, y_train)



print ("Model Score (R^2) is: \n", lasso_model.score(X_test, y_test))



lasso_predictions = lasso_model.predict(X_test)



#importing sklearn metrics and RMSE function. RMSE measures the distance between or predicted values and actual values.



from sklearn.metrics import mean_squared_error

print ('MSE is: \n', mean_squared_error(y_test, lasso_predictions))

print ('RMSE is: \n', np.sqrt(mean_squared_error(y_test, lasso_predictions)))



print ('cross_val_score: \n', cross_val_score(lasso_model, X_train, y_train, cv=10).mean())
plt.figure(figsize=(15,8))

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

sns.regplot(y_test, lasso_predictions, color='brown')

plt.show()
from sklearn.linear_model import ElasticNet



en = ElasticNet(random_state=0)

en_model = en.fit(X_train, y_train)



print ("Model Score (R^2) is: \n", en_model.score(X_test, y_test))



en_predictions = en_model.predict(X_test)



#importing sklearn metrics and RMSE function. RMSE measures the distance between or predicted values and actual values.



from sklearn.metrics import mean_squared_error

print ('MSE is: \n', mean_squared_error(y_test, en_predictions))

print ('RMSE is: \n', np.sqrt(mean_squared_error(y_test, en_predictions)))



print ('cross_val_score: \n', cross_val_score(en_model, X_train, y_train, cv=10).mean())
plt.figure(figsize=(15,8))

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

sns.regplot(y_test, en_predictions, color='grey')

plt.show()
from sklearn.tree import DecisionTreeRegressor



dt = DecisionTreeRegressor(random_state = 0)

dt_model = dt.fit(X_train, y_train)



print ("Model Score (R^2) is: \n", dt_model.score(X_test, y_test))



dt_predictions = dt_model.predict(X_test)



#importing sklearn metrics and RMSE function. RMSE measures the distance between or predicted values and actual values.



from sklearn.metrics import mean_squared_error

print ('MSE is: \n', mean_squared_error(y_test, dt_predictions))

print ('RMSE is: \n', np.sqrt(mean_squared_error(y_test, dt_predictions)))



print ('cross_val_score: \n', cross_val_score(dt_model, X_train, y_train, cv=10).mean())
plt.figure(figsize=(15,8))

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

sns.regplot(y_test, dt_predictions, color='b')

plt.show()
# Fit Random Forest on Training Set



from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators=500, random_state=0)

rf_model = rf.fit(X_train, y_train)



print ("Model Score (R^2) is: \n", rf_model.score(X_test, y_test))



rf_predictions = rf_model.predict(X_test)



#importing sklearn metrics and RMSE function. RMSE measures the distance between or predicted values and actual values.



from sklearn.metrics import mean_squared_error

print ('MSE is: \n', mean_squared_error(y_test, rf_predictions))

print ('RMSE is: \n', np.sqrt(mean_squared_error(y_test, rf_predictions)))

print ('cross_val_score: \n', cross_val_score(rf_model, X_train, y_train, cv=10).mean())
plt.figure(figsize=(15,8))

plt.scatter(y_test, rf_predictions, c= 'green')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

ax = sns.regplot(y_test, rf_predictions, color="r")

plt.show()
from sklearn import ensemble

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor

from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_error, r2_score



parameters = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}



gbr = ensemble.GradientBoostingRegressor(**parameters)



gbr.fit(X_train, y_train)



gbr_pred = gbr.predict(X_test)

gbr_pred= gbr_pred.reshape(-1,1)



print ("Model Score (R^2) is: \n", gbr.score(X_test, y_test))

print('MSE:', mean_squared_error(y_test, gbr_pred))

print('RMSE:', np.sqrt(mean_squared_error(y_test, gbr_pred)))

print ('cross_val_score: \n', cross_val_score(gbr, X_train, y_train, cv=10).mean())
plt.figure(figsize=(15,8))

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.scatter(y_test, gbr_pred, color='purple')

plt.show()
# Initialize the model

random_forest = RandomForestRegressor(n_estimators=1200,

                                      max_depth=15,

                                      min_samples_split=5,

                                      min_samples_leaf=5,

                                      max_features=None,

                                      random_state=42,

                                      oob_score=True

                                     )



# Perform cross-validation to see how well our model does 

kf = KFold(n_splits=5)

y_pred = cross_val_score(random_forest, X_test, y_test, cv=kf, n_jobs=-1)

y_pred.mean()
# Fit the model to our data

random_forest.fit(X_test, y_test)
# Make predictions on test data

rf_pred = random_forest.predict(X_test)

rf_pred.mean()
# Initialize our model

g_boost = GradientBoostingRegressor( n_estimators=6000, learning_rate=0.01,

                                     max_depth=5, max_features='sqrt',

                                     min_samples_leaf=15, min_samples_split=10,

                                     loss='ls', random_state =42

                                   )



# Perform cross-validation to see how well our model does 

kf = KFold(n_splits=5)

y_pred = cross_val_score(g_boost, X, y, cv=kf, n_jobs=-1)

y_pred.mean()
# Fit our model to the training data

g_boost.fit(X_test,y_test)
gbm_pred = g_boost.predict(X_test)

gbm_pred.mean()
xg_boost = XGBRegressor( learning_rate=0.01,

                         n_estimators=6000,

                         max_depth=4, min_child_weight=1,

                         gamma=0.6, subsample=0.7,

                         colsample_bytree=0.2,

                         objective='reg:linear', nthread=-1,

                         scale_pos_weight=1, seed=27,

                         reg_alpha=0.00006

                       )



# Perform cross-validation to see how well our model does 

kf = KFold(n_splits=5)

y_pred = cross_val_score(xg_boost, X_test, y_test, cv=kf, n_jobs=-1)

y_pred.mean()
# Fit our model to the training data

xg_boost.fit(X_test,y_test)
# Make predictions on the test data

xgb_pred = xg_boost.predict(X_test)

xgb_pred.mean()
# Initialize our model

lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=6,

                                       learning_rate=0.01, 

                                       n_estimators=6400,

                                       verbose=-1,

                                       bagging_fraction=0.80,

                                       bagging_freq=4, 

                                       bagging_seed=6,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                    )



# Perform cross-validation to see how well our model does

kf = KFold(n_splits=5)

y_pred = cross_val_score(lightgbm, X, y, cv=kf)

print(y_pred.mean())
# Fit our model to the training data

lightgbm.fit(X_test,y_test)
# Make predictions on test data

lgb_pred = lightgbm.predict(X_test)

lgb_pred.mean()
# List of the models to be stacked

models = [g_boost, random_forest, xg_boost, lightgbm]
def rmse(y, y_pred):

    return np.sqrt(mean_squared_error(np.log(y), np.log(y_pred)))



# Perform Stacking

S_train, S_test = stacking(models,

                           X_train, y_train, X_test,

                           regression=True,

                           mode='oof_pred_bag',

                           metric=rmse,

                           n_folds=5,

                           random_state=25,

                           verbose=2

                          )
# Initialize 2nd level model

xgb_lev2 = XGBRegressor(learning_rate=0.1, 

                        n_estimators=500,

                        max_depth=3,

                        n_jobs=-1,

                        random_state=17

                       )



# Fit the 2nd level model on the output of level 1

xgb_lev2.fit(S_train, y_train)
# Make predictions on the localized test set

stacked_pred = xgb_lev2.predict(S_test)

print("RMSE of Stacked Model: {}".format(rmse(y_test,stacked_pred)))
final = pd.DataFrame()

final['Id'] = test.Id
features = test.select_dtypes(

        include=[np.number]).drop(['Id'], axis=1).interpolate()
model = lr_model

model2 = ridge_model

model3 = rf_model



predictions_lr = model.predict(features)

predictions_ridge = model2.predict(features)

predictions_rf = model3.predict(features)
final_predictions = np.exp(predictions_lr)



final_predictions2 = np.exp(predictions_ridge)



final_predictions3 = np.exp(predictions_rf)
print ("Original predictions are: \n", predictions_lr[:5], "\n")

print ("Final predictions are: \n", final_predictions[:5])



print ("Original predictions are: \n", predictions_ridge[:5], "\n")

print ("Final predictions are: \n", final_predictions2[:5])



print ("Original predictions are: \n", predictions_rf[:5], "\n")

print ("Final predictions are: \n", final_predictions3[:5])
final['SalePrice'] = final_predictions

final.head(20)

final['SalePrice'] = final_predictions2

final.head(20)

final['SalePrice'] = final_predictions3

final.head(20)