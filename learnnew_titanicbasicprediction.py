import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#print(os.listdir("../input"))
train=pd.read_csv('../input/train.csv')
train.drop('Cabin',axis=1,inplace=True)

test=pd.read_csv('../input/test.csv',na_values=' ')
test.drop('Cabin',axis=1,inplace=True)

#print(train.head())
#print(train.describe())
#print(test.info())
#print(train['Ticket'])
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
sex = pd.get_dummies(train['Sex'],drop_first=True)  
embark = pd.get_dummies(train['Embarked'],drop_first=True)

train2=train.drop(['Sex','Embarked','Name','Ticket','Fare'],axis=1)
train = pd.concat([train2,sex,embark],axis=1)




sex = pd.get_dummies(test['Sex'],drop_first=True)  
embark = pd.get_dummies(test['Embarked'],drop_first=True)

test2=test.drop(['Sex','Embarked','Name','Ticket','Fare'],axis=1)
test = pd.concat([test2,sex,embark],axis=1)
test = test.fillna('')

from sklearn.model_selection import train_test_split
X_train=train.drop(['Survived'],axis=1)
y_train=train['Survived']
X_test=test
#X_test['Fare'].fillna('S', inplace=True)
X_test=X_test.fillna("345")
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
predictions=lr.predict(X_test)
acc_log = round(lr.score(X_train, y_train) * 100, 2)
acc_log
from sklearn.neighbors import KNeighborsClassifier
acc_log_i=1
for i in range(1,150):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred =knn.predict(X_test)
    acc_log = round(knn.score(X_train, y_train) * 100, 2)
    #print(acc_log)
    if acc_log_i<acc_log :
        n=i
        acc_log_i=acc_log

n
knn=KNeighborsClassifier(n_neighbors=n)  
knn.fit(X_train,y_train)
pred =knn.predict(X_test)
acc_log = round(knn.score(X_train, y_train) * 100, 2)
acc_log


submission = pd.DataFrame({
        "PassengerId": X_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('submission1.csv', index=False)




