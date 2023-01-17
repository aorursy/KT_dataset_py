%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

import numpy as np

import pandas as pd

import sklearn as sk

from math import sqrt

import warnings

warnings.filterwarnings('ignore')

import statsmodels.api as sm

import yellowbrick

from sklearn.linear_model import Ridge

from sklearn.preprocessing import Imputer

from yellowbrick.regressor import ResidualsPlot

import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression
df = pd.read_csv("../input/2018as.csv" ) 
# shape of dataset, rows and columns

print("rows, columns: "+str(df.shape))

print(len(str(df.shape))*'-')

print(df.dtypes.value_counts())

print(len(str(df.shape))*'-')

df.head(2)
# check for null values  

for column in df:

    if df[column].isnull().any():

       print('{0} has {1} null values'.format(column, df[column].isnull().sum()))
# Removing all Faceoff related stats from the dataframe

df.drop(['FO%','FOW','FOL'], axis=1, inplace=True)

# check again for null values

df.isnull().values.any()
## Descriptive statistics of dataset

df.describe(include='all').transpose()

 
df.drop(['CF', 'CA','CF% rel', 'C/60', 'Crel/60', 'FF', 'FA','FF% rel','oiSH%', 'oiSV%'], axis=1, inplace=True)

df.shape
df.drop(['A', 'PTS'], axis=1, inplace=True)

df.shape
df.head()
# Goals by position 

ax = sns.boxplot(x = "Pos", y = "G", data = df)

ax.set_title("Goals by position")

plt.xlabel("Position")

plt.ylabel("Goals")

plt.show()
# goal distribution plot

ax = sns.distplot(df['G'])
# regression plot ... goals by games played

ax = sns.lmplot(x = "GP", y = "G", data = df)

plt.xlabel("Games Played")

plt.ylabel("Goals")

plt.show()
# one hot encoding the Pos feature.

df = pd.get_dummies(df)
# correlation matrix

corr = df.corr() 

# plot the heatmap

plt.figure(figsize=(10,7))

ax = sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)

 
from sklearn.model_selection import train_test_split

import xgboost as xgb



X =  df.drop(['G'], axis=1)

# target column: we are predicting goals

y = df.G

# split training and test

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.3, random_state=123)

 
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline 



my_pipeline = Pipeline([('xgb', XGBRegressor())])
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error



param_grid = {

    "xgb__n_estimators": [10, 50, 100, 500],

    "xgb__learning_rate": [0.1, 0.5, 1],

}

fit_params = {"xgb__eval_set": [(val_X, val_y)], 

              "xgb__early_stopping_rounds": 10, 

              "xgb__verbose": False} 



searchCV = GridSearchCV(my_pipeline, cv=10, param_grid=param_grid, fit_params=fit_params)

searchCV.fit(train_X, train_y)  

 

best_parameters=searchCV.best_estimator_.get_params()

for param_name in sorted(param_grid.keys()):

        print("\t%s: %r" % (param_name, best_parameters[param_name]))

preds = searchCV.predict(val_X)
from sklearn.metrics import mean_squared_error

searchCV.fit(train_X,train_y)

preds = searchCV.predict(val_X)

preds
rmse = np.sqrt(mean_squared_error(val_y, preds))

print("RMSE: %f" % (rmse))
importance = searchCV.best_estimator_.named_steps["xgb"].feature_importances_ 

feat_importances = pd.Series(importance, index=train_X.columns)

ax = feat_importances.nlargest(20).plot(kind='barh')
