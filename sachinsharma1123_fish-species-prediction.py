# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/fish-market/Fish.csv')
df
df.info()
#check for null values

df.isnull().sum()
df['Species'].unique()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['Species']=le.fit_transform(df['Species'])
df
y=df['Species']

x=df.drop(['Species'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

lr=LogisticRegression(max_iter=100000)

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(y_test,pred_1)
score_1
from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,11):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    preds=knn.predict(x_test)

    scores=accuracy_score(y_test,preds)

    list_1.append(scores)

    
import matplotlib.pyplot as plt

plt.plot(range(1,11),list_1)

plt.xlabel('k values')

plt.ylabel('accuracy scores')

plt.show()
print(max(list_1))
from sklearn.ensemble import RandomForestClassifier



rfc=RandomForestClassifier()

params={'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

 'max_features': ['auto', 'sqrt'],

       'n_estimators': [100,150,200,300,350,400,450,500,550]}

gs=GridSearchCV(rfc,params,cv=5,verbose=0)
gs.fit(x_train,y_train)

print(gs.best_params_)


rfc=RandomForestClassifier(max_depth=30,max_features='sqrt',n_estimators=100)

rfc.fit(x_train,y_train)

pred_2=rfc.predict(x_test)

score_2=accuracy_score(y_test,pred_2)
score_2
from sklearn.svm import SVC

clf_1=SVC()
param_grid = {'C': [0.1, 1, 10, 100, 1000],  

              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 

              'kernel': ['rbf']}

gs_2=GridSearchCV(clf_1,param_grid,cv=5,verbose=0)

gs_2.fit(x_train,y_train)

print(gs_2.best_params_)
clf_1=SVC(C=1000,gamma=0.0001,kernel='rbf')

clf_1.fit(x_train,y_train)

pred_3=clf_1.predict(x_test)

score_3=accuracy_score(y_test,pred_3)
score_3
#logistics regression gives the best accuracy score among all the classifiers
new_df=pd.DataFrame({'actual':y_test,

                    'predicted':pred_1})
new_df