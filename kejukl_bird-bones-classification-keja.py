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
import matplotlib.pyplot as plt
bird=pd.read_csv('../input/bird.csv')
bird.id=bird.id.apply(str)
bird_size=bird.groupby('type').size().sort_values()
print(bird_size)
%matplotlib inline
ax=bird_size.plot(kind='bar',color='blue',figsize=(8,6),rot=45)
print('before dropping NA the shape is')
print(bird.shape)
bird=bird.dropna(axis=0,how='any')
print('after dropping NA the shape is')
print(bird.columns)
#Standard Scaling:
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler(with_mean=False)
#data=pd.DataFrame(scaler.fit_transform(data))
col=['huml', 'humw', 'ulnal', 'ulnaw', 'feml', 'femw', 'tibl', 'tibw',
       'tarl', 'tarw','type']
data=bird[col]
import seaborn as sns
sns.lmplot( x="huml", y="ulnal", data=data, fit_reg=False, 
           hue="type", legend=False)
data['hand']=bird.ulnal+bird.huml
data['leg']=bird.feml+bird.tibl+bird.tarl
data['limbratio']=data['hand'].div(data.leg)
#Related to bone density
data['area1']=data['huml']*data['humw']
data['area2']=data['ulnal']*data['ulnaw']
data['area3']=data['feml']*data['femw']
data['area4']=data['tibl']*data['tibw']
data['area5']=data['tarl']*data['tarw']
data['area_ratio_limb']=data.area1+data.area2.div(data.area3+data.area4+data.area5)

import seaborn as sns
sns.lmplot(x='area_ratio_limb',y='limbratio',data=data,hue='type',fit_reg=False,legend=False)
Xcol=['huml','ulnal','feml','tibl','tarl',
      'humw','ulnaw','femw','tibw','tarw',
      'area_ratio_limb','area1','area2','area3',
      'area4','area5','area_ratio_limb','limbratio',]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(
                                               data[Xcol],data.type,
                                               test_size=0.25,random_state=42
                                              )
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(data[Xcol],data.type,
                                               test_size=0.25,random_state=42
                                              )
print(X_train.columns)
print('shape of data',data.shape)
print('shape of training set',X_train.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
X_train.dropna(axis=0,how='any').shape
lr=LogisticRegression()
parameter={
    'penalty':["l2","l1"],
    'C':[0.5,0.6,0.75]
    }
optimised_lr= GridSearchCV(estimator=lr, param_grid=parameter, scoring="accuracy", cv=4)
optimised_lr.fit(X_train,Y_train)
y_predict=optimised_lr.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(Y_test, y_predict))