import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.head()
train = train.drop(["PassengerId","Ticket","Name","Fare","Cabin","Embarked"],axis=1)
train.head()

train["Family"] = train["SibSp"]+train["Parch"]+1
train = train.drop(["SibSp","Parch"],axis=1)
train = train.dropna()

def changer(sex):
    if sex=="male" :
        return 1
    else:
        return 0
        
train.Sex = train.Sex.map(changer)
X= train.drop("Survived",axis=1)
y= train.Survived


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
from sklearn.svm import SVC # Support Vector Classifier
model = SVC()
model.fit(X_train,y_train)
pred = model.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,pred))
print('test accuracy: {}'.format(model.score(X_test,y_test)))
score= []
from sklearn.neighbors import KNeighborsClassifier
for i in range(1,100):
    model2 = KNeighborsClassifier(n_neighbors=i)
    model2.fit(X_train,y_train)
    pred2=model2.predict(X_test)
    score.append(classification_report(y_test,pred2))
    
np.argmax(score)
model2 = KNeighborsClassifier(n_neighbors=3)
model2.fit(X_train,y_train)
pred2=model2.predict(X_test)
print(classification_report(y_test,pred2))
print('test accuracy: {}'.format(model2.score(X_test,y_test)))
from sklearn.linear_model import LogisticRegressionCV

lr=LogisticRegressionCV(cv=10)
lr.fit(X_train,y_train)
predict=lr.predict(X_test)
print(classification_report(y_test,predict))
print('test accuracy: {}'.format(lr.score(X_test,y_test)))
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=10,min_samples_split=0.3)
dt.fit(X_train,y_train)
pred = dt.predict(X_test)
print(classification_report(y_test,pred))
print('test accuracy: {}'.format(dt.score(X_test,y_test)))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()


rfc_params = {'n_estimators':np.arange(1,200),'max_depth':np.arange(0,20)}
rfc_grid = GridSearchCV(rfc,rfc_params,cv=20)
rfc_grid.fit(X,y)
rfc_grid_cv.best_score_