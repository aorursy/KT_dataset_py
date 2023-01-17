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
df=pd.read_csv('../input/Iris.csv')
df.head()
df.isnull().sum()
X=df.iloc[:,1:5].values
y=df.iloc[:,5].values
y = np.where(y=='Iris-setosa',-1,1)
y
X.shape
y.shape
np.random.seed(0)
idx=np.random.permutation(y.shape[0])
idx
y=y[idx]
X=X[idx]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=0)
y_train.shape
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve
from sklearn.pipeline import Pipeline
param_range=[0.00001,0.0001, 0.001, 0.01, 0.1, 1.0, 10.0,100.0,1000.0,10000.0]
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(penalty='l2',
                                               random_state=0))])
train_scores, test_scores=validation_curve(estimator=pipe_lr,
                                           X=X_train,
                                           y=y_train,
                                           param_name='clf__C',
                                           param_range=param_range,
                                           cv=10)
train_scores
train_mean = np.mean(train_scores, axis=1)#mean row wise
train_mean
test_mean=np.mean(test_scores, axis=1)
train_std=np.std(train_scores, axis=1)
test_std=np.std(test_scores, axis=1)
plt.plot(param_range,train_mean,color='blue',marker='o', 
         markersize=5,label="train accuracy")
plt.fill_between(param_range,train_mean+train_std,
                 train_mean-train_std,color='blue',
                 alpha=0.15)
plt.plot(param_range,test_mean,color='green',marker='s',
         markersize=5,linestyle='--',
         label="test accuracy")
plt.fill_between(param_range,test_mean+test_std,
                 test_mean-test_std,color='green',
                 alpha=0.15)
plt.xscale('log')
plt.legend(loc='best')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.9, 1.0])
plt.show()
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.transform(X_test)
lr=LogisticRegression(penalty='l2',C=0.0001,random_state=0)
lr.fit(X_train_std,y_train)
lr.score(X_train_std,y_train)
lr.score(X_test_std,y_test)
