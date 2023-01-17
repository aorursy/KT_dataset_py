# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.cm as cm

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/train.csv")

test = pd.read_csv("/kaggle/input/test.csv")

variable_description = pd.read_csv("/kaggle/input/variable_description.csv")
# print varible list

variable_description
# print first 5 rows of train dataset

train.head()
test1 = test.drop('crime_rate',axis=1)

test1.head()
train.describe()
train.isnull().sum()
train.dtypes
for col in train.columns:

    plot = plt.boxplot(train[col])

    print(f'plot of feature {col} is {plot}')

    plt.show()
# We are deleting the outliers from quartile 12th percentile and 88th percentile as appropriate for our model.

Q1 = train.quantile(0.12)

Q3 = train.quantile(0.88)

IQR = Q3 - Q1

print(IQR)
#Deleting Outliers

train1 = train[~((train < (Q1 - 1.5 * IQR)) |(train > (Q3 + 1.5 * IQR))).any(axis=1)]

train1.shape
train.shape
plt.figure(figsize=(16,8))

corr = train.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
#Fetaures 

X = train1.drop('crime_rate', axis=1).copy()
X.head()
#label

y = train1['crime_rate'].copy()
y.head()
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y,train_size=0.8, test_size=0.2,random_state = 0)
from sklearn.linear_model import LinearRegression

regr = LinearRegression()

regr.fit(train_X,train_y)
from sklearn.metrics import r2_score

predict = regr.predict(val_X)

r2_score(val_y,predict)
from sklearn.model_selection import GridSearchCV

import xgboost

#for tuning parameters

# parameters_for_testing = {

#    'colsample_bytree':[0.4,0.6,0.8],

# #    'gamma':[0,0.03,0.1,0.3],

# #    'min_child_weight':[1.5,6,10],

# #    'learning_rate':[0.1,0.07],

# #    'max_depth':[3,5],

# #    'n_estimators':[10000],

# #    'reg_alpha':[1e-5, 1e-2,  0.75],

# #    'reg_lambda':[1e-5, 1e-2, 0.45],

# #    'subsample':[0.6,0.95]  

# }



                    

# xgb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,

#     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=-1, scale_pos_weight=1, seed=27)



# gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6,iid=False, verbose=10,scoring='neg_mean_squared_error')

# gsearch1.fit(train_X,train_y)

# # print (gsearch1.grid_scores_)

# print('best params')

# print (gsearch1.best_params_)

# print('best score')

# print (gsearch1.best_score_)



#After tuning

xgb_model = xgboost.XGBRegressor(learning_rate =0.01, n_estimators=10000, max_depth=5,

    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=-1, scale_pos_weight=1, seed=27)



xgb_model.fit(train_X,train_y)

predict = xgb_model.predict(val_X)

r2_score(val_y,predict)
from sklearn.datasets import load_boston

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import VotingRegressor



# Training classifiers

reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)

reg2 = RandomForestRegressor(random_state=1, n_estimators=10)

reg3 = LinearRegression()

ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])

ereg = ereg.fit(train_X,train_y)

predict = ereg.predict(val_X)

r2_score(val_y,predict)
ereg_final = ereg.fit(X,y)

predict = ereg_final.predict(test1)

sub = pd.DataFrame(data=predict

                   ,columns=["crime_rate"])
sub.to_csv("submission.csv")