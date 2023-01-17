import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.head()
train_data.info()
train_data.isnull().sum()
train_data.describe()
train_data.shape
train_data['target'].value_counts().plot(kind='bar')
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error,classification_report,confusion_matrix,accuracy_score,roc_curve,auc,roc_auc_score
y = train_data['target']

X = train_data.drop(['id','target'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)



X_test = scaler.fit_transform(X_test)
log_model = LogisticRegression()
log_model.fit(X_train,y_train)

y_pred = log_model.predict(X_test)

print(log_model.score(X_test, y_test))
#Thanks to https://www.kaggle.com/ateplyuk/dntoverfit-starter

c_val = [0.001, 0.01, 0.1, 1, 10, 100]

penalty_val = ['l1','l2']

params = { 'penalty': penalty_val,

          'C' : c_val,

         'solver':['liblinear']}
grid = GridSearchCV(log_model,param_grid=params,cv=3)
grid.fit(X_train,y_train)

y_pred = grid.predict(X_test)

print(log_model.score(X_test, y_test))
grid.score(X_test, y_test)
grid.best_params_
X_test_test = test_data.drop(['id'],axis =1)



X_test_test = scaler.fit_transform(X_test_test)



y_pred_test = grid.predict(X_test_test)





output = pd.DataFrame({'id': test_data.id,'target': y_pred_test})

output.to_csv('submission2.csv', index=False)

output.head()