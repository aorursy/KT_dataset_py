# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install --upgrade sk-dist

!pip install pyspark
from skdist.distribute.search import DistGridSearchCV

from xgboost import XGBRegressor

from pyspark.sql import SparkSession

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

import pandas as pd


def get_xy(df):

    y = df.quality

    X = df.drop('quality',axis = 1)

    return X, y



spark = (

    SparkSession

    .builder

    .getOrCreate()

    )

sc = spark.sparkContext



path = '../input/red-wine-quality-cortez-et-al-2009/'

df = pd.read_csv(path+'winequality-red.csv')



X, y = get_xy(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



grid = dict(

    learning_rate=[.05, .01],

    max_depth=[4, 6, 8],

    colsample_bytree=[.6, .8, 1.0],

    n_estimators=[100, 200, 300]

)



cv = 5

reg_scoring = "neg_mean_squared_error"



### distributed grid search

model = DistGridSearchCV(

    XGBRegressor(objective='reg:squarederror'),

    grid, sc, cv=cv, scoring=reg_scoring

    )



model.fit(X_train, y_train)



print("-- Grid Search --")

print("Best Score: {0}".format(model.best_score_))

print("Best colsample_bytree: {0}".format(model.best_estimator_.colsample_bytree))

print("Best learning_rate: {0}".format(model.best_estimator_.learning_rate))

print("Best max_depth: {0}".format(model.best_estimator_.max_depth))

print("Best n_estimators: {0}".format(model.best_estimator_.n_estimators))



y_pred = model.predict(X_test)



print("MSE: {0}".format(mean_squared_error(y_test, y_pred)))


