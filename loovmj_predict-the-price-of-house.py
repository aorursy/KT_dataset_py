import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
test_data = pd.read_csv('../input/test.csv')

train_data = pd.read_csv('../input/train.csv')
train_data.info()
test_data.head()
test_data.info()
import matplotlib.pyplot as plt

train_data.hist(bins=20, figsize=(20,15))

plt.show()
train_data.corr()[train_data.corr() > 0.5]
import seaborn as sb
corr = train_data.corr()

fig, ax = plt.subplots(figsize=(30,30))

sb.heatmap(corr, annot=True, square=True, ax=ax, cmap='Blues')

plt.xticks(fontsize=20);

plt.yticks(fontsize=20);
# Sales price is highly related with OverallQual, YearBuilt, YearRemodAdd, Total BsmSF, 

# TotalBsmtSF, 1stFlrSF, GrLivArea, FullBath, TotRmsAbvGrd, GarageCar, GarageArea
corr_2 = train_data[['SalePrice', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF',

                    '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']].corr()



fig, ax = plt.subplots(figsize=(15,15))

sb.heatmap(corr_2, annot=True, square=True, ax=ax, cmap='Blues')

plt.xticks(fontsize=10);

plt.yticks(fontsize=10);
from pandas.plotting import scatter_matrix

attributes = ['SalePrice', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF',

             '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']



scatter_matrix(train_data[attributes], figsize=(20, 20));

train_data[attributes].isnull().sum()
numerical = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF','1stFlrSF', 'GrLivArea', 'FullBath', 

             'TotRmsAbvGrd', 'GarageCars', 'GarageArea']



all_data = pd.concat([train_data[numerical], test_data[numerical]])

# features

features_train = all_data[:train_data.shape[0]]

features_test = all_data[train_data.shape[0]:]



# labels

train_labels = train_data['SalePrice']



# feature scalling

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



features_scaling = pd.DataFrame(data=all_data)

features_scaling[numerical] = scaler.fit_transform(all_data[numerical])

display(features_scaling)
# fill median values to NaN with Imputer

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='median')

imputer.fit(features_scaling[numerical])



# these are calculated median for each column 

imputer.statistics_
# replace NaN with the median but this is numpy array

rep_features_train = imputer.transform(features_train)



# change into Data Frame

rep_features_train_df = pd.DataFrame(rep_features_train, 

                                     columns=features_train.columns)

display(rep_features_train_df.head())
# Null check for test data

test_data[numerical].isnull().sum()
# replace NaN with the median but this is numpy array

replaced_test = imputer.transform(test_data[numerical])



# change into Data Frame

replaced_test_df = pd.DataFrame(

    replaced_test, columns=test_data[numerical].columns)



display(replaced_test_df.head())
# implement the random forest model

# tuning parameters with Gridsearch



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



clf = RandomForestRegressor(random_state=42)



param_grid = {'n_estimators' : [3, 10, 30, 100],

             'max_depth': [3, 5, 8],

             'max_features': [2, 4, 6, 8, 10]}



# info for scoring : https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

grid_search = GridSearchCV(clf, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)

grid_fit = grid_search.fit(rep_features_train_df, train_labels)



# show the best parameters 

grid_search.best_params_
# check feature importance of this random forests

# 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF','1stFlrSF', 'GrLivArea', 'FullBath', 

# 'TotRmsAbvGrd', 'GarageCars', 'GarageArea'

feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances
feature_importances.max()

# OverallQual seems like the most importance features...
# use tha parameter above and make the model

rdm_forestReg = RandomForestRegressor(max_depth=8, max_features=6, 

                                 n_estimators=100,random_state=42)

rdm_forestReg.fit(rep_features_train_df, train_labels)

rdm_forestReg_pred = rdm_forestReg.predict(replaced_test_df)



display(rdm_forestReg_pred)

#display(len(rdm_forestReg_pred))
# evaluate the model

from sklearn.model_selection import cross_val_score



# the cross-validation rmse error

rmse= np.sqrt(-cross_val_score(rdm_forestReg, 

                               rep_features_train_df, train_labels, scoring="neg_mean_squared_error", cv = 5))

rmse
rmse.mean()
rmse.std()
solution = pd.DataFrame({'id': test_data['Id'], 'SalePrice':rdm_forestReg_pred})

solution.head()
solution.to_csv('random_forest_house_price.csv', index=False)