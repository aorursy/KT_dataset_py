import numpy as np 

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
data_train = pd.read_csv('/kaggle/input/udacity-mlcharity-competition/census.csv')

data_test = pd.read_csv('/kaggle/input/udacity-mlcharity-competition/test_census.csv')

goal_test = pd.read_csv('/kaggle/input/udacity-mlcharity-competition/example_submission.csv')
data_train.head()
data_test.head()
goal_test
data_train.income.unique()
income=data_train.income.map({'<=50K': 0, '>50K':1})
features = pd.get_dummies(data_train.drop(['income'],1))
scaler = StandardScaler()

features = scaler.fit_transform(features)
x_train , x_test , y_train , y_test = train_test_split(features,income,test_size=0.2,random_state=0)

logistic= LogisticRegression(random_state=0)

logistic.fit(x_train,y_train)
print('Train score is: ',logistic.score(x_train,y_train))

print('Test score is:',logistic.score(x_test,y_test))
cm = confusion_matrix(y_test,logistic.predict(x_test))

print(logistic.score(x_test,y_test))

pd.DataFrame(cm)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(x_train, y_train)
print(classifier.score(x_train,y_train))

print(classifier.score(x_test,y_test))
cm = confusion_matrix(y_test, classifier.predict(x_test))

print(classifier.score(x_test, y_test))

pd.DataFrame(cm)
x_train , x_test , y_train , y_test = train_test_split(features,income,test_size=0.2,random_state=0)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=9)

classifier.fit(x_train,y_train)
print('Train score is:',classifier.score(x_train,y_train))

print('Test score is :', classifier.score(x_test,y_test))
#cm = confusion_matrix(y_test, classifier.predict(x_test))

#print(classifier.score(x_test, y_test))

#pd.DataFrame(cm)
test = data_test.drop(['Unnamed: 0'] , axis=1)
test.head()
test.fillna(method='ffill', inplace=True)
test = pd.get_dummies(data_test)
test.head()
final_test = test.drop(['Unnamed: 0'] , axis=1)
final_test.fillna(method='ffill', inplace=True)
final_test.head()
scaler= StandardScaler()

final_test = scaler.fit_transform(final_test)
x_train , x_test , y_train , y_test = train_test_split(features,income,test_size=0.2,random_state=0)

logistic= LogisticRegression(random_state=0)

logistic.fit(x_train,y_train)

logistic.predict(x_test)

logistic.predict(final_test)
goal_test.head()
goal_test['id'] = goal_test.iloc[:,0] 

goal_test['income'] = logistic.predict(final_test)
goal_test.head()
goal_test.to_csv('example_submission.csv',index=False,header=1)