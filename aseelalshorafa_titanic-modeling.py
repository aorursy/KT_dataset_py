import pandas as pd

import matplotlib.pyplot as plt 

import numpy as np
data = pd.read_csv('../input/titanic/train.csv',index_col = "PassengerId")

test = pd.read_csv('../input/titanic/test.csv',index_col = "PassengerId")



#test_y = pd.read_csv('../dataset/titanic/gender_submission.csv',index_col = "PassengerId")

indexs=  test.index
data.sample(2)
data.info()
data.shape
X = data.iloc[:,1:]

y = data.iloc[:,0]
X.columns
X['Ticket'].mode
y
X.sample()
X =X.drop(columns =['Name'])
X.info()
from sklearn.impute import SimpleImputer

imputer_no = SimpleImputer(missing_values= np.nan ,strategy = 'mean')

imputer_no.fit(X[['Pclass','Age','SibSp','Fare','Parch']])

X[['Pclass','Age','SibSp','Fare','Parch']] = imputer_no.transform(X[['Pclass','Age','SibSp','Fare','Parch']])

imputer_cat = SimpleImputer(missing_values= np.nan ,strategy = 'most_frequent')

imputer_cat.fit(X[['Sex','Cabin','Embarked','Ticket']])

X[['Sex','Cabin','Embarked','Ticket']]=imputer_cat.transform(X[['Sex','Cabin','Embarked','Ticket']])
X.info()
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder 
ct = ColumnTransformer(transformers= [('encoder',OneHotEncoder(handle_unknown='ignore'),[1,5,7,8])],remainder = 'passthrough')

X=  ct.fit_transform(X)
from sklearn.model_selection import train_test_split 

train_X,test_X,train_y,test_y = train_test_split(X,y)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score 

for i in range(10,300,10):

    classifier = RandomForestClassifier(n_estimators= i,criterion='gini' )

    classifier.fit(train_X, train_y)

    y_predict = classifier.predict(test_X)

    print('for {} estimators and {}'.format({i},{accuracy_score(y_true=test_y,y_pred=y_predict)}))



test.columns
test.info()
test =test.drop(columns =['Name'])


imputer_no.fit(test[['Pclass','Age','SibSp','Fare','Parch']])

test[['Pclass','Age','SibSp','Fare','Parch']] = imputer_no.transform(test[['Pclass','Age','SibSp','Fare','Parch']])

imputer_cat.fit(test[['Sex','Cabin','Embarked','Ticket']])

test[['Sex','Cabin','Embarked','Ticket']]=imputer_cat.transform(test[['Sex','Cabin','Embarked','Ticket']])

test=  ct.transform(test)

test
classifier = RandomForestClassifier(n_estimators= 150,criterion='gini')

classifier.fit(X, y)

y_predict = classifier.predict(test)

pd.DataFrame(y_predict,index=indexs,columns=['Survived'] ).to_csv('output.csv')
y_predict
#def unique_values(my_col):

   # return my_col.nunique()
#train_X.apply(unique_values,axis = 0)
#from sklearn.compose import ColumnTransformer

#from sklearn.preprocessing import OneHotEncoder 

#ct = ColumnTransformer(transformers= [('encoder',OneHotEncoder(),[2])],remainder = 'passthrough')

#ct = ColumnTransformer(transformers= [('encoder',OneHotEncoder(),[9])],remainder = 'passthrough')

#train_X = ct.fit_transform(train_X)
#pd.DataFrame(train_X)
#train_X.describe()