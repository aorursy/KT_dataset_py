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
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
avocado = pd.read_csv("../input/avocado.csv")
print ('There are',len(avocado.columns),'columns:')
for x in avocado.columns:
    print(x+' ',end=',')
avocado.head()
avocado.tail()
avocado.info()
avocado['type'].value_counts()
sns.heatmap(avocado.isnull(),yticklabels=False)
avocado.columns.values
sns.jointplot(x='Large Bags',y='Small Bags',data=avocado)
sns.jointplot(x='XLarge Bags',y='Large Bags',data=avocado)
sns.jointplot(x='Small Bags',y='XLarge Bags',data=avocado)
sns.countplot(avocado['type'])
avocado = avocado.drop(['Unnamed: 0','Date'],axis=1)
avocado.head()
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
len(avocado['region'].value_counts())
fh = FeatureHasher(n_features=5,input_type='string')
hashed_features = fh.fit_transform(avocado['region']).toarray()
avocado = pd.concat([avocado,pd.DataFrame(hashed_features)],axis=1)
avocado.head()
avocado = avocado.drop('region',axis=1)
X = avocado.drop('type',axis=1)
y = avocado['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
pred1 = rfc.predict(X_test)
print(classification_report(y_test,pred1))
print(confusion_matrix(y_test,pred1))
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred2 = knn.predict(X_test)
print(classification_report(y_test,pred2))
print(confusion_matrix(y_test,pred2))
params = {'C':[1,10,100,1000,10000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),params,verbose=3)

grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
pred3 = grid.predict(X_test)
print(classification_report(y_test,pred3))
print(confusion_matrix(y_test,pred3))






