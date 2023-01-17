# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/train.csv')

print(df.head())
df = df.drop('Name',axis=1)

df = df.drop('Ticket',axis = 1)
print(df.info())

print(df.describe())
print(df.head(20))
df = df.drop('Cabin',axis=1)

df.info()
df_origin = pd.get_dummies(df)

df_origin = df_origin.drop(['Sex_male','Embarked_C'],axis=1)

X = df_origin.drop(['Survived','PassengerId'],axis=1).values

y = df_origin['Survived'].values

print(X)

print(y)

from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN',strategy='mean',axis=0)

imp.fit(X)

X = imp.transform(X)
from sklearn.preprocessing import scale

X_scaled = scale(X)
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier

steps = [('scaler',StandardScaler()),('knn',KNeighborsClassifier())]

parameters = {'knn__n_neighbors':np.arange(1,50)}

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=21)

pipeline = Pipeline(steps)

cv = GridSearchCV(pipeline,param_grid = parameters,cv=5)

cv.fit(X_train,y_train)

print(cv.score(X_test,y_test))

from sklearn.svm import SVC

steps = [('scaler',StandardScaler()),('SVM',SVC())]

parameters = {'SVM__C':[0.01,0.03,0.1,0.3,1,3,10],'SVM__gamma':[0.01,0.03,0.1,0.3,1,3,10]}

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

pipeline = Pipeline(steps)

cv = GridSearchCV(pipeline,param_grid = parameters,cv=10)

cv.fit(X,y)

print(cv.score(X_test,y_test))
dftest = pd.read_csv("../input/test.csv")

print(dftest.info())
dftest = dftest.drop(['Name','Cabin','Ticket'],axis=1)

df_test_origin = pd.get_dummies(dftest)

df_test_origin = df_test_origin.drop(['Sex_male','Embarked_C'],axis=1)

X_final_test = df_test_origin.drop(['PassengerId'],axis=1).values

imp_test = Imputer(missing_values='NaN',strategy='mean',axis=0)

imp_test.fit(X_final_test)

X_final_test = imp.transform(X_final_test)

scaler = StandardScaler()

scaler.fit(X_final_test)

X_final_test=scaler.transform(X_final_test)

y_pred = cv.predict(X_final_test)

print(y_pred)

print(len(y_pred))

final = {'PassengerId':df_test_origin['PassengerId'],'Survived':y_pred}

finaldf = pd.DataFrame(final)

finaldf.head()

finaldf.to_csv("answer.csv",index=False)