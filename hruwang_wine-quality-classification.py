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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pylab as plot
data=pd.read_csv('../input/winequality-red.csv')
data.info()
print(data.describe())
drop_quality = data.drop('quality', axis=1)
quality = data.quality
 
DQ_train, DQ_test, Q_train, Q_test = train_test_split(drop_quality, quality,test_size=0.2, random_state=123, stratify=quality)
DQ_train_scaled = preprocessing.scale(DQ_train)   #Data Scaling
pipeline = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(n_estimators=100))
hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(DQ_train, Q_train)
Q_pred = clf.predict(DQ_test)
 
print('Prediction R_Squared:',r2_score(Q_test, Q_pred))
print('Prediction MSE:',mean_squared_error(Q_test, Q_pred))
wine_dataset=pd.read_csv('../input/winequality-red.csv')
X_train,X_test,y_train,y_test = train_test_split(drop_quality,quality,random_state=0)
print('X_train shape:{}'.format(X_train.shape))
print('X_test shape:{}'.format(X_test.shape))
print('y_train shape:{}'.format(y_train.shape))
print('y_test shape:{}'.format(y_test.shape))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
print(knn)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=1, p=2,
           weights='uniform')
print('Prediction SCORE：{:.2f}'.format(knn.score(X_test,y_test)))