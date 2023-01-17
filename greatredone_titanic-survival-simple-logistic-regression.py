import pandas as pd

import numpy as np

import math 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



train = pd.read_csv('train.csv');

corrmat = train.corr();

#train.info();

test = pd.read_csv('test.csv')

def normalize(inputData):

    temp = inputData.fillna(value = inputData.mean())

    mu = inputData.mean()

    sigma = inputData.std()

    return (temp - mu) / sigma



def denormalize(inputData, mu, sigma):

    return (inputData * sigma) + mu



def sigmoid(z):

    return  1.0 / ( 1 + np.exp(-z))
corrmat
logit = LogisticRegression(max_iter=400)
train.head()
test.head()
print("Unique values of where passengers embarked " + str(train.Embarked.unique()))

print("Unique number of classes passengers where in " + str(train.Pclass.unique()))

print("Number of Siblings & Spouses aboard ship " + str(train.SibSp.unique()))

print("Number of parents & children aboard the Titanic " + str(train.Parch.unique()))

print("Number of men and women on board " + str(train.Sex.unique()))
train.Sex = train.Sex.astype('category')

train.Embarked = train.Embarked.fillna(value = 'S')

train.Embarked = train.Embarked.astype('category')

train = train.drop(['Name','Ticket','Cabin'],axis=1)

train.set_index('PassengerId');

train.info()
# fill missing ages 

mean_age = train.Age.mean()

train.Age = train.Age.fillna(mean_age)

train.Age = normalize(train.Age)

train.Fare = normalize(train.Fare)

cols_to_transform = [ 'Sex','Embarked' ]

train = pd.get_dummies(train, columns = cols_to_transform )

#1, x2, y1, y2, idx1, idx2 = train_test_split(data, labels, indices, test_size=0.2)
train.head()
test.head()
y = train['Survived']

X = train[['Pclass', 'Age','SibSp','Parch','Fare','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S']]



m, n = np.shape(X)

X0 = np.ones(m)

X = np.column_stack([X0, X])



#df.insert(idx, col_name, value)

train.insert(0,'X0', X0)

train2 = train[['X0','Pclass', 'Age','SibSp','Parch','Fare','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S']]
y.head()
train2.isnull().any()
train2.info()
sns.heatmap(corrmat, vmax=.8, linewidths=0.01,square=True,annot=True)

plt.show()
X_train, X_test, y_train, y_test = train_test_split(train2, y, test_size=0.33, random_state=42)
logit.fit(X_train,y_train)



print (" logistic regression score " + str(logit.score(X_train,y_train)))

logit.fit(train2,y)
logit.score(train2,y)
print ("logistic regression score test set " + str(logit.score(X_test,y_test)))
pred = logit.predict(X_test)
logit.coef_[0]

test.head()
#test.Sex = test.Sex.astype('category')

#test.Embarked = test.Embarked.fillna(value = 'S')

#test.Embarked = test.Embarked.astype('category')

#test = test.drop(['Name','Ticket','Cabin'],axis=1)

#test.set_index('PassengerId');

#test.info()

test.isnull().any()
# fill missing ages 

#mean_age = test.Age.mean()

#test.Age = test.Age.fillna(mean_age)

#mean_fare = test.Fare.mean()

#test.Fare = test.Fare.fillna(mean_fare)



#test.Age = normalize(test.Age)

#test.Fare = normalize(test.Fare)

#cols_to_transform = [ 'Sex','Embarked' ]

#test = pd.get_dummies(test, columns = cols_to_transform )

test.info()
Xt = test[['Pclass', 'Age','SibSp','Parch','Fare','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S']]

m, n = np.shape(Xt)

X0 = np.ones(m)

Xt = np.column_stack([X0, Xt])
submit = logit.predict(Xt)
results = pd.DataFrame(submit,index=test['PassengerId'] )
results = results.rename(columns = {0 : 'Survived'})
results.head()
results.to_csv('submit.csv', sep=',')