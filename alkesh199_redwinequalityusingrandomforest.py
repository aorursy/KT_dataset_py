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

import matplotlib.pyplot as plt

import sklearn as sns



dataset = pd.read_csv('../input/winequality-red.csv')
dataset.head(10)







dataset.describe()
dataset.isnull().sum()
dataset.columns
import seaborn as sns



sns.catplot(data = dataset,x='quality',y='fixed acidity',kind = 'bar')

sns.catplot(data = dataset,x = 'quality',y = 'volatile acidity', kind = 'bar')

sns.catplot(data = dataset,x = 'quality',y = 'citric acid',kind = 'bar')
sns.catplot(data = dataset,x='quality',y='residual sugar',kind = 'bar')

sns.relplot(data = dataset,x = 'quality',y ='residual sugar',kind = 'line')
sns.catplot(data = dataset,x = 'quality',y = 'chlorides',kind = 'bar')

corr_mat = dataset.corr()

corr_mat.style.background_gradient(cmap='coolwarm')
sns.catplot(data = dataset,x = 'quality',y='alcohol',kind = 'bar')

sns.catplot(data = dataset,x = 'quality',y = 'sulphates',kind = 'bar')

sns.catplot(data=dataset,kind = 'box')

label_range_value  = (2, 6.5, 10)

class_name = ['bad', 'good']

dataset['quality'] = pd.cut(dataset['quality'], bins = label_range_value, labels = class_name)
X = dataset.iloc[:,0:11].values

y = dataset.iloc[:,11:12].values
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X_train,y_train)

log_reg.score(X_train,y_train)

y_pred_log = log_reg.predict(X_test)

log_reg.score(X_test,y_test)#0.895
from sklearn.model_selection import cross_val_score

cv_score_all = cross_val_score(estimator = log_reg,X= X_train,y=y_train,cv = 5)

cv_score_all

cv_score_all.mean()

cv_score_all.std()
from sklearn.model_selection import GridSearchCV



grid_values = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}

model_log_reg = GridSearchCV(log_reg, param_grid=grid_values)



model_log_reg.fit(X_train,y_train)

model_log_reg.score(X_train,y_train)



model_log_reg.score(X_test,y_test)#0.9



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 34)

rf.fit(X_train,y_train)

rf.score(X_train,y_train)

y_pred_rf = rf.predict(X_test)

rf.score(X_test,y_test)
from sklearn.model_selection import cross_val_score

cv_score_all = cross_val_score(estimator = rf,X= X_train,y=y_train,cv = 5)

cv_score_all

cv_score_all.mean()

cv_score_all.std()