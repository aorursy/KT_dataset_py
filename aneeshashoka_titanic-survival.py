import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

import statistics

data = pd.read_csv('/kaggle/input/titanic/train.csv')

test= pd.read_csv('/kaggle/input/titanic/test.csv')
data.head(5)



data.describe()
test.head(5)
print(data.isnull().sum(), test.isnull().sum())



#EDA

#1_Remove irrelavent variables





data.drop('Name', axis=1, inplace= True)

data.drop('Ticket', axis=1, inplace= True)

data.drop('Cabin', axis=1, inplace= True)

test.drop('Name', axis=1, inplace= True)

test.drop('Fare', axis=1, inplace= True)

data.drop('Fare', axis=1, inplace= True)

test.drop('Ticket', axis=1, inplace= True)

test.drop('Cabin', axis=1, inplace= True)

test.drop('PassengerId', axis=1, inplace= True)

data.drop('PassengerId', axis=1, inplace= True)







    
data.head()

test.head()
m_embarked= statistics.mode(data['Embarked'])

data['Embarked'].fillna(m_embarked, inplace=True)

m_embarked= statistics.mode(test['Embarked'])

test['Embarked'].fillna(m_embarked, inplace=True)
data['Age'].fillna(data['Age'].mean(), inplace= True)



test['Age'].fillna(test['Age'].mean(), inplace= True)
test.isnull().sum()
data.isnull().sum()
data['Sex']= (np.where(data['Sex'].values=='male', 1,0))

test['Sex']= (np.where(test['Sex'].values=='male', 1,0))
data['Embarked']= (np.where(data['Embarked'].values=='S', 1,data['Embarked']))

data['Embarked']= (np.where(data['Embarked'].values=='C', 2,data['Embarked']))

data['Embarked']= (np.where(data['Embarked'].values=='Q', 1,data['Embarked']))

test['Embarked']= (np.where(test['Embarked'].values=='S', 1,test['Embarked']))

test['Embarked']= (np.where(test['Embarked'].values=='C', 2,test['Embarked']))

test['Embarked']= (np.where(test['Embarked'].values=='Q', 1,test['Embarked']))

print(test.head())

print(test.count())
print(data.head())

print(data.count())
X_train, X_test, Y_Train, Y_Test= train_test_split( data.iloc[:,1:8] ,data['Survived'], test_size=0.4, random_state=4)

print(X_train.shape, X_test.shape, Y_Train.shape, Y_Test.shape)
## KNN MODEL with knn= 5



knn= KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, Y_Train)

y_pred = knn.predict(X_test)

print(y_pred[0:5])

print (metrics.accuracy_score(Y_Test,y_pred))

#END OF KNN MODEL WHERE KNN =5
#KNN model...with k=1

knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, Y_Train)

y_predd= knn.predict(X_test)

print(y_pred[0:5])

print(metrics.accuracy_score(Y_Test,y_pred))

#END OF KNN MODEL WHERE KNN =1
# try K=1 through K=25 and record testing accuracy

k_range = range(1, 26)

# We can create Python dictionary using [] or dict()

scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, Y_Train)

    y_pred = knn.predict(X_test)

    scores.append(metrics.accuracy_score(Y_Test, y_pred))

print(scores)
# plot the relationship between K and testing accuracy

plt.plot(k_range, scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Testing Accuracy')

plt.show()
y_predd_test= knn.predict(test)

print(y_predd_test[0:-1])
##LOGICAL REGRESSION



from sklearn.model_selection import train_test_split

X = data.iloc[:, 1:8].values

y = data.iloc[:, 0].values

X_train, X_test,Y_train,Y_test= train_test_split(X,y,test_size=0.2)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,Y_train)
predictions = logmodel.predict(X_test)

print(predictions)
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix,accuracy_score

print(classification_report(Y_test, predictions))

print(confusion_matrix(Y_test, predictions))

print(accuracy_score(Y_test, predictions))
predictions_testLR = logmodel.predict(test)

print(predictions_testLR)