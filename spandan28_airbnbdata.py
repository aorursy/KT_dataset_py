# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
airbnb_dataframe = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

airbnb_dataframe.head(5)
airbnb_dataframe = airbnb_dataframe.drop(columns=['id', 'name', 'host_id', 'host_name','last_review'])

airbnb_dataframe.dtypes


airbnb_dataframe["neighbourhood"] = pd.Categorical(airbnb_dataframe["neighbourhood"])

airbnb_dataframe["neighbourhood_group"] = pd.Categorical(airbnb_dataframe["neighbourhood_group"])

airbnb_dataframe["room_type"] = pd.Categorical(airbnb_dataframe["room_type"])

airbnb_dataframe = pd.concat([airbnb_dataframe, pd.get_dummies(airbnb_dataframe["neighbourhood"], prefix="neighborhood"),

                              pd.get_dummies(airbnb_dataframe["neighbourhood_group"], prefix ="neigh_group"),

                             pd.get_dummies(airbnb_dataframe["room_type"], prefix ="room_type")], axis=1)



airbnb_dataframe = airbnb_dataframe.drop(columns=["neighbourhood","neighbourhood_group","room_type"])

airbnb_dataframe = airbnb_dataframe.dropna()

Y = airbnb_dataframe["price"]

X = airbnb_dataframe.drop(columns=["price"])
from sklearn import preprocessing

cols = X.columns

X_scaled = preprocessing.scale(X)

X_scaled = pd.DataFrame(X_scaled, columns=cols)



## note that scaled data has 0 mean and 1 variance

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size = 0.1, random_state = 1)
X_train.head(10)
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)

xgb_model.fit(X_train, Y_train)

predictions = xgb_model.predict(X_test)

print(explained_variance_score(predictions,Y_test))
plt.scatter(predictions, Y_test, marker="o")