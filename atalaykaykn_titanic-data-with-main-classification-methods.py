# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/titanic/test.csv')
df2 = pd.read_csv('/kaggle/input/titanic/train.csv')
df= pd.concat([df2,df])
df.shape
df.head()
pd.get_dummies(df['Embarked'])
df =pd.concat([pd.get_dummies(df['Embarked']),df],axis=1)
df.drop(['Embarked'],axis=1,inplace=True)
df.head()
df.drop(['PassengerId','Cabin','Ticket','Name'],axis=1,inplace=True)
df.dropna()


df2=df
plt.figure(figsize=(20,6))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
df2.dropna(axis=0,inplace=True)

y = df2['Survived']
df2.drop('Survived',axis=1,inplace=True)
label_encoder = preprocessing.LabelEncoder()
df2['Sex']= label_encoder.fit_transform(df['Sex']) 
X = df2
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_X,y,test_size=0.2,random_state=54)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
logisticRegressionPredict =logmodel.predict(X_test)
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)
logreg2=LogisticRegression(C=10.0,penalty="l2")
logreg2.fit(X_train,y_train)
print("score",logreg2.score(X_test,y_test))
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg2, X_train, y_train, cv=10)
print('Cross-Validation Accuracy Scores', scores)

from sklearn.neighbors import KNeighborsClassifier
help(KNeighborsClassifier())
knn = KNeighborsClassifier()
params={'n_neighbors':range(1,50), 'weights':['uniform','distance'], 'metric':['euclidean','manhattan']}
logreg_cv=GridSearchCV(knn,params,cv=10)
logreg_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)
knn2= KNeighborsClassifier(metric='manhattan',n_neighbors=12,weights='uniform')
knn2.fit(X_train,y_train)
print("score",knn2.score(X_test,y_test))
scores = cross_val_score(knn2, X_train, y_train, cv=10)
print('Cross-Validation Accuracy Scores', scores)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
params={'max_depth':range(1,21), 'min_samples_leaf':[1, 5, 10, 20, 50, 100], 'criterion':['gini', 'entropy']}
logreg_cv=GridSearchCV(dt,params,cv=10)
logreg_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)
dt2 = DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_leaf=1)
dt2.fit(X_train,y_train)
print("score",dt2.score(X_test,y_test))
scores = cross_val_score(dt2, X_train, y_train, cv=10)
print('Cross-Validation Accuracy Scores', scores)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
help(RandomForestClassifier)
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)

print("tuned hpyerparameters :(best parameters) ",CV_rfc.best_params_)
print("accuracy :",CV_rfc.best_score_)
rfc2 = RandomForestClassifier(bootstrap=True,max_depth=90,max_features=3,min_samples_leaf=4,min_samples_split=8,n_estimators=100)
rfc2.fit(X_train,y_train)
print("score",rfc2.score(X_test,y_test))
scores = cross_val_score(rfc2, X_train, y_train, cv=10)
print('Cross-Validation Accuracy Scores', scores)
from sklearn.svm import SVC
svc = SVC()

help(SVC)
paramsvc={'C':[0.001, 0.01, 0.1, 1, 10], 'gamma':[0.001, 0.01, 0.1, 1]}
paramsvc ={'kernel':('linear', 'rbf'), 'C':(1,0.25,0.5,0.75),'gamma': (1,2,3,'auto'),'decision_function_shape':('ovo','ovr'),'shrinking':(True,False)}
logreg_cv=GridSearchCV(svc,paramsvc,cv=10)
logreg_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)
svc2= SVC(C=0.75,decision_function_shape='ovo',gamma=3,kernel='rbf',shrinking=True)
svc2.fit(X_train,y_train)
print("score",svc2.score(X_test,y_test))
scores = cross_val_score(svc2,X_train, y_train, cv=10)
print('Cross-Validation Accuracy Scores', scores)


