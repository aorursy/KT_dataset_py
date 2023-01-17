# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.grid_search import GridSearchCV
df=pd.read_csv('../input/train.csv')
df_predict=pd.read_csv('../input/test.csv')
df_predict.head()
df_predict.head()
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Fare']=df['Fare'].fillna(df['Fare'].mean())
df_predict['Age']=df_predict['Age'].fillna(df_predict['Age'].mean())
df_predict['Fare']=df_predict['Fare'].fillna(df_predict['Fare'].mean())
def minor(age):
    if age <= 16:
        return 1
    else:
        return 0

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substrings) != -1:
            return substring
    print (big_string)
    return np.nan
def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
y=list()
z=list()
df['Initial']=0
for i in df:
    df['Initial']=df.Name.str.extract('([A-Za-z]+)\.')
for j in df_predict:
    df_predict['Initial']=df.Name.str.extract('([A-Za-z]+)\.')
df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
df_predict['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
for x in df['Age']:
    y.append(minor(x))
for k in df_predict['Age']:
    z.append(minor(k))    
df['Minor']=y
df_predict['Minor']=z
df_predict.head(20)
parameters=['Survived','Pclass','Sex','Age','Fare','SibSp','Parch','Embarked','Initial','Minor','PassengerId']
trainx=df[parameters]
df_pred=['Pclass','Sex','Age','SibSp','Fare','Parch','Embarked','Initial','Minor','PassengerId']
df_test=df_predict[df_pred]
y=df.Survived
k=trainx
one_hot_encoded_training_predictors = pd.get_dummies(k)
one_hot_encoded_training_predictors_out = pd.get_dummies(df_test)
df_test.head()
df_test['Age']=df_test['Age'].fillna(df_test['Age'].mean())
df_test['Fare']=df_test['Fare'].fillna(df_test['Fare'].mean())
one_hot_encoded_training_predictors_out.head(20)
X = np.array(one_hot_encoded_training_predictors.drop(['Survived','PassengerId'], 1))
training_features = np.array(one_hot_encoded_training_predictors.drop(['Survived','PassengerId'], 1).columns)
#X = preprocessing.scale(X)  --- not needed for XGboost?
y = np.array(one_hot_encoded_training_predictors['Survived'])
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
y_predict= model.predict(np.array(one_hot_encoded_training_predictors_out[training_features]))
y_predict
dfresult = pd.DataFrame(y_predict, one_hot_encoded_training_predictors_out.PassengerId)
dfresult.columns = ['Survived']
dfresult.to_csv('submissions.csv')
print("done.")
df_test.dtypes
X = np.array(one_hot_encoded_training_predictors.drop(['Survived','PassengerId'], 1))
training_features = np.array(one_hot_encoded_training_predictors.drop(['Survived','PassengerId'], 1).columns)
#X = preprocessing.scale(X)  --- not needed for XGboost?
y = np.array(one_hot_encoded_training_predictors['Survived'])
clf = XGBClassifier()
cv = cross_validation.KFold(len(X), n_folds=20, shuffle=True, random_state=1)
scores = cross_validation.cross_val_score(clf, X, y, cv=cv, n_jobs=1, scoring='accuracy')
clf.fit(X,y)
print(scores)
print('Accuracy: %.3f stdev: %.2f' % (np.mean(np.abs(scores)), np.std(scores)))
test2=one_hot_encoded_training_predictors_out[training_features]
test2.isnull().sum()
test2=one_hot_encoded_training_predictors_out[training_features]
z = np.array(test2)
y_predict = clf.predict(z)
#print('The accuracy of the Random Forests is',metrics.accuracy_score(y_predict,one_hot_encoded_training_predictors_out))
dfresult = pd.DataFrame(y_predict, one_hot_encoded_training_predictors_out.PassengerId)
y_predict
dfresult.columns = ['Survived']
dfresult.to_csv('submissions.csv')
print("done.")
