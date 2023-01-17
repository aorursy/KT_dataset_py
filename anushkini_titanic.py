import numpy as np

import pandas as pd
df1 = pd.read_csv('../input/train.csv')

df2 = pd.read_csv('../input/test.csv')
df2.head(30)
column_target = ['Survived']

column_train = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
X_train = df1[column_train]

Y_train = df1[column_target]

X_test = df2[column_train]
print(X_train['Sex'].isnull().sum())

print(X_train['Pclass'].isnull().sum())

print(X_train['Age'].isnull().sum())

print(X_train['SibSp'].isnull().sum())

print(X_train['Parch'].isnull().sum())

print(X_train['Fare'].isnull().sum())

#print(X_train['Cabin'].isnull().sum())

#print(X_train['Embarked'].isnull().sum())
print(X_test['Sex'].isnull().sum())

print(X_test['Pclass'].isnull().sum())

print(X_test['Age'].isnull().sum())

print(X_test['SibSp'].isnull().sum())

print(X_test['Parch'].isnull().sum())

print(X_test['Fare'].isnull().sum())

#print(X_test['Cabin'].isnull().sum())

#print(X_test['Embarked'].isnull().sum())
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].median())

X_train['Cabin'] = X_train['Cabin'].fillna(0)

X_train['Embarked'] = X_train['Embarked'].fillna(0)



X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean())

X_test['Cabin'] = X_test['Cabin'].fillna(0)

X_test['Fare'] = X_test['Fare'].fillna(X_test['Fare'].mean())
X_train.head()
d = {'male':0,'female':1}



X_train['Sex'] = X_train['Sex'].apply(lambda x:d[x])

X_test['Sex'] = X_test['Sex'].apply(lambda x:d[x])

#d2 = {'0':0,'S':1,'C':3,'Q':2}



X_train['Embarked'] = X_train['Embarked'].astype('category')

X_test['Embarked'] = X_test['Embarked'].astype('category')



X_train['Embarked'] = X_train['Embarked'].cat.codes

X_test['Embarked'] = X_test['Embarked'].cat.codes



#X_train['Embarked'] = X_train['Embarked'].apply(lambda x:d2[x])

#X_test['Embarked'] = X_test['Embarked'].apply(lambda x:d2[x])


for i in range(len(X_train['Cabin'])):

    if(X_train['Cabin'][i] != 0):

        X_train['Cabin'][i] = 1

        

for i in range(len(X_test['Cabin'])):

    if(X_test['Cabin'][i] != 0):

        X_test['Cabin'][i] = 1 





X_train['Cabin'] = X_train['Cabin'].astype('category')

X_test['Cabin'] = X_test['Cabin'].astype('category')



X_train['Cabin'] = X_train['Cabin'].cat.codes

X_test['Cabin'] = X_test['Cabin'].cat.codes



X_train.head(30)
X_test.head()
X_train["Age"]=((X_train["Age"]-X_train["Age"].min())/(X_train["Age"].max()-X_train["Age"].min()))

X_train["Fare"]=((X_train["Fare"]-X_train["Fare"].min())/(X_train["Fare"].max()-X_train["Fare"].min()))



X_test["Age"]=((X_test["Age"]-X_test["Age"].min())/(X_test["Age"].max()-X_test["Age"].min()))

X_test["Fare"]=((X_test["Fare"]-X_test["Fare"].min())/(X_test["Fare"].max()-X_test["Fare"].min()))

X_train.head()
from sklearn import svm

from sklearn.model_selection import GridSearchCV
#clf = svm.SVC()

#clf.fit(X_train,Y_train)

def svc_param_selection(X,y,nfolds):

	Cs = [0.001,0.01,0.1,1,10]

	gammas = [0.001,0.01,0.1,1,10]

	param_grid = {'C': Cs,'gamma':gammas}

	grid_search = GridSearchCV(svm.SVC(),param_grid,cv=nfolds)

	grid_search.fit(X,y)

	print(grid_search.best_params_)

	return grid_search
clf = svc_param_selection(X_train,Y_train,5)
res = clf.predict(X_test)
df3 = pd.DataFrame(columns = ['PassengerId','Survived'])
df3['PassengerId'] = df2['PassengerId']

df3['Survived'] = df3['Survived'].fillna(0)

df3.head()

for i in range(len(X_test)):

    df3['Survived'][i] = res[i]
df3.to_csv('My_submission_svm.csv',index=False)
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=200)

random_forest.fit(X_train, Y_train)
res = random_forest.predict(X_test)
for i in range(len(X_test)):

    df3['Survived'][i] = res[i]
df3.to_csv('My_submission_forest.csv',index=False)
features = pd.DataFrame()

features['feature'] = column_train

features['importance'] = random_forest.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(10, 10))
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()

regressor.fit(X_train,Y_train)
res = regressor.predict(X_test)
for i in range(len(X_test)):

    df3['Survived'][i] = res[i]
df3.to_csv('My_submission_log_regressor.csv',index=False)