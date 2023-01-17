# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import xgboost
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
train = pd.read_excel("../input/research_student (1).xlsx")
train.head(10)
train = train.dropna()
train.tail(7)
train.shape
train.info
train.describe()
train.columns
train.Rank.value_counts()
train.Branch.value_counts()
train.head(125)
train.info()
scale_list = [ 'Marks[10th]', 'Marks[12th]', 'GPA 1', 'Rank', 'Normalized Rank', 'CGPA','GPA 2', 'GPA 3', 'GPA 4', 'GPA 5',
       'GPA 6']
sc = train[scale_list]
sc.head()
scaler = StandardScaler()
sc = scaler.fit_transform(sc)
sc[0]
sc[1]
train[scale_list]=sc
train.head()
train[scale_list].head()
train.head()
train.info()
encoding_list = ['Branch','Gender','Board[10th]','Board[12th]','Category']
train[encoding_list] = train[encoding_list].apply(LabelEncoder().fit_transform)
train.head()
train.info()
x = train.drop('CGPA',axis=1)
y = train['CGPA']
x.info()
X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=0.3)
X_train.shape
X_test.shape
logreg=LinearRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
y_test
y_pred
print(metrics.mean_squared_error(y_test, y_pred))
xgb = xgboost.XGBRegressor(n_estimators=25000, learning_rate=0.06, gamma=0, subsample=0.6,
                           colsample_bytree=0.7, min_child_weight=4, max_depth=3)
xgb.fit(X_train,y_train)
train = train.rename(columns={'Board[10th]': 'Board10th', 'Board[12th]': 'Board12th','Marks[10th]': 'Marks10th','Marks[12th]': 'Marks12th','GPA 1': 'GPA1','GPA 2': 'GPA2','GPA 3': 'GPA3','GPA 4': 'GPA4','GPA 5': 'GPA5'})
train.head()
