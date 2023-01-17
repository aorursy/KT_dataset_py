import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
dataset = pd.read_csv('/kaggle/input/churns/churn_train.csv')

features =dataset.iloc[:, 3:11].values

goals =dataset.iloc[:, -1].values

dataset.head()
dataset.info()
scaler = StandardScaler()

features = scaler.fit_transform(features)



train_set,test_set,train_goal,test_goal =train_test_split(features,goals,test_size=0.2,random_state=0)
logistic = LogisticRegression(random_state=0)

logistic.fit(train_set,train_goal)

ypred_train = logistic.predict(train_set)

ypred_test = logistic.predict(test_set)
print('Train score is :',logistic.score(train_set,train_goal))

print('Test score is:',logistic.score(test_set,test_goal))
cm = confusion_matrix(test_goal,logistic.predict(test_set))

print('Score is:',logistic.score(test_set,test_goal))

pd.DataFrame(cm)
scaler = StandardScaler()

features = scaler.fit_transform(features)



train_set,test_set,train_goal,test_goal =train_test_split(features,goals,test_size=0.2,random_state=0)
from sklearn.neighbors import KNeighborsClassifier 

cl = KNeighborsClassifier(n_neighbors = 3)

cl.fit(train_set,train_goal)
print('Train score is:',cl.score(train_set,train_goal))

print('Test score is:',cl.score(test_set,test_goal))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_goal, cl.predict(test_set))

pd.DataFrame(cm)
# use the bernoulli method because it works fine with (zeros and ones)

from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB()

clf.fit(train_set,train_goal)

print('Train score is :', clf.score(train_set,train_goal))

print('Test score is :', clf.score(test_set,test_goal))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_goal, clf.predict(test_set))

pd.DataFrame(cm)
test_dataset = pd.read_csv('/kaggle/input/churns/churn_test.csv')

test_dataset.head()
#final_features = test_dataset.iloc[:,3:6
test = test_dataset.drop(['Customer ID'],axis=1)
test.head()
final_test = pd.get_dummies(test)

final_test.head()
final_test.info()
final_features = test_dataset.iloc[:,5:8]

#goals= test_dataset.iloc[:,]
scaler = StandardScaler()

final_test = scaler.fit_transform(final_test)
x_train , x_test , y_train , y_test = train_test_split(features,goals,test_size=0.2,random_state=0)
from sklearn.neighbors import KNeighborsClassifier 

clf = KNeighborsClassifier(n_neighbors = 7)

clf.fit(x_train,y_train)

clf.predict(x_test)

clf.predict(features)
goal_data = pd.read_csv('/kaggle/input/churns/sample submission.csv')

goal_data.head()
goal_data['Churn Status'] = clf.predict(final_test)
goal_data.head()