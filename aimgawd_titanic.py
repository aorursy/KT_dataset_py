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
train=pd.read_csv("/kaggle/input/titanic/train.csv")

test=pd.read_csv("/kaggle/input/titanic/test.csv")

print(train.shape)
train.head()
x=train['Age'].mean()

train['Age'].fillna(x,inplace=True)

xx=test['Age'].mean()

test['Age'].fillna(xx,inplace=True)

print(train.info())
print(train.loc[2,'Age'])
z=[]

for i in range(891):

    if(train.loc[i,'Sex']=='male'):

        z.append(0)

    else:

        z.append(1)

train['sex']=z



z=[]

for i in range(1309-891):

    if(test.loc[i,'Sex']=='male'):

        z.append(0)

    else:

        z.append(1)

test['sex']=z
train=train.drop(['Name','Embarked','Ticket','Cabin','Sex','PassengerId'],axis=1)

test=test.drop(['Name','Embarked','Ticket','Cabin','Sex'],axis=1)

print(train.info())

print(test.info())
test.head()
from sklearn.model_selection import train_test_split

X=train.drop('Survived',axis=1).values

y=train['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

l=[]

p=[]

for g in range(1,50):

    steps = [('scaler', StandardScaler()),('knn', KNeighborsClassifier(n_neighbors=g))]

    pipeline = Pipeline(steps)

    knn_scaled = pipeline.fit(X_train, y_train)

    cc=knn_scaled.score(X_test,y_test)

    l.append(cc)

    p.append(g)



plt.plot(p,l)

plt.show()
print(l[3])   ###MAX ACC FOR 4 NEIGHBORS
steps2 = [('scaler', StandardScaler()),

        ('knn', KNeighborsClassifier(n_neighbors=4))]

        

# Create the pipeline: pipeline

pipeline2 = Pipeline(steps)

knn_scaled2 = pipeline.fit(X_train, y_train)



print(knn_scaled2.score(X_test,y_test))
x=test.Fare.mean()

test['Fare'].fillna(x,inplace=True)

test.info()
c=test.drop('PassengerId',axis=1).values

predicted=knn_scaled2.predict(c)

test['Survived']=predicted

test

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVR
##USING RANDOM FOREST REGRESSOR 

rf=RandomForestClassifier(random_state=42)

rf.get_params()
from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train,y_train)

rf_random.best_params_
realrf=RandomForestClassifier(n_estimators= 2000,min_samples_split=2,min_samples_leaf= 4,max_features='auto',max_depth= 30,bootstrap= False)
realrf.fit(X_train,y_train)

print(realrf.score(X_test,y_test))
steps3 = [('scaler', StandardScaler()),

        ('rfrf', RandomForestClassifier(n_estimators= 2000,min_samples_split=2,min_samples_leaf= 4,max_features='auto',max_depth= 30,bootstrap= False))]

        

# Create the pipeline: pipeline

pipeline3 = Pipeline(steps3)

rf_scale= pipeline3.fit(X_train, y_train)



print(rf_scale.score(X_test,y_test))


dfs=pd.DataFrame({'PassengerId':test['PassengerId'].values,'Survived':rf_scale.predict(test.drop(['PassengerId','Survived'],axis=1).values)})
xd=dfs.to_csv('abcd.csv',index=False)