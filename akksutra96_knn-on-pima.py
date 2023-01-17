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
df=pd.read_csv("../input/diabetes.csv")
df.head()
df.shape
y=df['Outcome']
y.shape
X=df.drop('Outcome',axis=1)
X.head()
df['Outcome'].value_counts()
268/(500+268)
from sklearn.model_selection import train_test_split,GridSearchCV

X_tr,X_test,y_tr,y_test=train_test_split(X,y,test_size=0.35,stratify=y,random_state=21)
from sklearn.neighbors import KNeighborsClassifier



model=KNeighborsClassifier()
param_grid = {'n_neighbors':np.arange(1,50)}







knn= GridSearchCV(model,param_grid,cv=5,scoring='roc_auc',n_jobs=-1,pre_dispatch='2*n_jobs')

knn.fit(X_tr,y_tr)



knn.best_params_
knn.best_score_
y_pred=knn.predict(X_test)


print(y_pred.size)

print(y_tr.size)

#import confusion_matrix

from sklearn.metrics import confusion_matrix



cnf=confusion_matrix(y_test,y_pred)
cnf
from sklearn.metrics import classification_report





print(classification_report(y_test,y_pred))


