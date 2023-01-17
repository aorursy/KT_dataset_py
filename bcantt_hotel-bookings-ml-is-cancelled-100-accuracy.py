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
data = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import linear_model

from sklearn import svm

from sklearn import tree

import xgboost as xgb

from sklearn.ensemble import BaggingRegressor

import numpy as np 

import pandas as pd 

import random
data = data.fillna(data.median())
le = preprocessing.LabelEncoder()

for name in data.columns:

    if data[name].dtypes == "O":

        print(name)

        data[name] = data[name].astype(str)

        le.fit(data[name])

        data[name] = le.transform(data[name])
data
X = data.drop('is_canceled',axis = 1)

y = data['is_canceled']
regr = RandomForestClassifier()

regr.fit(X, y)



predictions_proba = regr.predict_proba(X)

predictions = regr.predict(X)
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy_score(y, predictions)
indexes = data.corr().index
predictions_proba = [i[1] for i in predictions_proba]
max_cor = 0

max_val_list = []

for i in range(100):

    total_score = pd.DataFrame()

    val_list = []

    for val,ind in zip([random.randint(-100,100) for i in range(len(indexes))],indexes):

        total_score[ind] = val * data[ind]

        val_list.append(val)

        

            

    score = total_score.sum(axis = 1).corr(pd.Series(predictions),method = 'pearson')

    if score > max_cor:

        max_cor = score

        max_val_list = val_list

        print(max_cor)

        print(val_list)



total_score = pd.DataFrame()

for val,ind in zip([i for i in max_val_list],indexes):

    total_score[ind] = val * data[ind]
import seaborn as sns
ax = sns.scatterplot(x = total_score.sum(axis = 1),y = pd.Series(predictions))
df = pd.DataFrame(total_score.sum(axis = 1).values,pd.Series(predictions)).reset_index()
data['scores'] = total_score.sum(axis = 1).values
X = data.drop('is_canceled',axis = 1)

y = data['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
regr = RandomForestClassifier()

regr.fit(X_train, y_train)



predictions = regr.predict(X_test)
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy_score(predictions, y_test)
predictions
y_test