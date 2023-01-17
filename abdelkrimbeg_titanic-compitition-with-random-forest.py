import numpy as np

import pandas as pd







from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

traindata = pd.read_csv('../input/titanic/train.csv')

testdata = pd.read_csv('../input/titanic/test.csv')

combine = [traindata , testdata]



traindata.info()

print('------------------------------------')

testdata.info()

#drop Name, ticket,cabin,passenid





traindatadrop = traindata.drop(['Name','Ticket','Cabin','PassengerId'],axis =1)

testdatadrop = testdata.drop(['Name','Ticket', 'Cabin'],axis =1)

combine = [traindatadrop, testdatadrop]



print(traindatadrop.shape)

print(testdatadrop.shape)

print('---------')

print(traindata.shape)

print(testdata.shape)

#replace male to 0 and female to 1

for data in combine:

    data['Sex'].replace('female',0,inplace=True)

    data['Sex'].replace('male',1,inplace=True)

#completing data for age 



for data in combine:

    mean = traindatadrop["Age"].mean()

    std = testdatadrop["Age"].std()

    is_null = data["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = data["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    data["Age"] = age_slice

    data["Age"] = traindatadrop["Age"].astype(int)

#completion datafor Embarked



common_value = 'S'

for data in combine:

    data['Embarked'] = data['Embarked'].fillna(common_value)

  
#replace data of embarked



for data in combine:

    data['Embarked'].replace('S',0,inplace=True)

    data['Embarked'].replace('C',1,inplace=True)

    data['Embarked'].replace('Q',2,inplace=True)

 
#completion datafor Fare



testdatadrop['Fare'].fillna(testdatadrop['Fare'].dropna().median(), inplace=True)
X_train = traindatadrop.drop("Survived", axis=1)

Y_train = traindatadrop["Survived"]

X_test  = testdatadrop.drop("PassengerId", axis=1).copy()
# Random Forest



random_forest = RandomForestClassifier(n_estimators=95)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print ('score =',acc_random_forest)
submission = pd.DataFrame({"PassengerId": testdata["PassengerId"],"Survived": Y_pred})

submission.to_csv('submission.csv', index=False)