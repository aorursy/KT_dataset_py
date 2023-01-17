import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # plotting graphs
data_train = pd.read_csv('../input/instant-gratification/train.csv')

data_test = pd.read_csv ('../input/instant-gratification/test.csv')
data_train.head(5)
data_test.head(5)
data_train.isnull().sum()
corr=data_train.corr()

sns.heatmap(corr)
X_train = data_train.iloc[:,1:256]

y_train = data_train.iloc[:,-1]
X_test = data_test.iloc[:,1:256]
from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression()

log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_train)
y_test=log_reg.predict(X_test)
log_reg.score(X_train,y_train)
log_reg.score(X_test,y_test)
from sklearn.metrics import confusion_matrix

cn = confusion_matrix(y_train,y_pred)

cn
from sklearn.metrics import precision_score,f1_score,recall_score

precision_score(y_train,y_pred)

recall_score(y_train,y_pred)

f1_score(y_train,y_pred)