import pandas as pd

import numpy as np

import matplotlib as plt

%matplotlib inline
test = pd.read_csv('../input/titanic/test.csv')

train= pd.read_csv('../input/titanic/train.csv')
train.head()
test.head()
train=train.drop(columns=['Name','Ticket','Cabin','Embarked','PassengerId'],axis=1)
test=test.drop(columns=['Name','Ticket','Cabin','Embarked'],axis=1)
train.head()
test.head()
train['Sex']=train['Sex'].map({'male':0,'female':1})
test['Sex']=test['Sex'].map({'male':0,'female':1})
train=train.dropna()
test=test.dropna()
train.head()
test.head()
X=train.drop('Survived',axis=1)

y=train['Survived']
from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(X,y)

y_predict = model.predict(X)



from sklearn.metrics import accuracy_score

accuracy_score(y,y_predict)
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

clf_rf = ensemble.RandomForestClassifier(random_state=1)

parameters = { 

    'n_estimators': [100, 400],

    'criterion' : ['gini', 'entropy'],

    'max_depth' : [2, 4, 6]    

}



from sklearn.model_selection import GridSearchCV, cross_val_score



cv_rf = GridSearchCV(estimator = clf_rf, param_grid = parameters, cv=5, n_jobs=-1)

X_test = test.drop('PassengerId', 1)



clf = cv_rf.fit(X, y)



predictions = clf.predict(X_test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)
