import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(12, 7))

sns.countplot(x='Survived',data=train,palette='RdBu_r')
plt.figure(figsize=(12, 7))

sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
plt.figure(figsize=(12, 7))

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
train.info()

print("***************************************************************")

test.info()
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=test,palette='winter')
train= train.drop(['PassengerId','Name','Ticket'], axis=1)

test= test.drop(['Name','Ticket'], axis=1)
# Get dummy values for both train and test dataset

# Their are 3 values in embrked: C, Q, S

# drop_first = True: Drop C column as it will be redudant because we can identify the emarked column from S and Q.

embark_train = pd.get_dummies(train['Embarked'],drop_first=True)

emark_test = pd.get_dummies(test['Embarked'], drop_first=True)



# Drop Emarked column

train.drop(['Embarked'],axis=1,inplace=True)

test.drop(['Embarked'],axis=1,inplace=True)



# Concat new embark columns in respective datasets

train = pd.concat([train,embark_train],axis=1)

test = pd.concat([test, emark_test], axis=1)

# Drop Cabin attribute from both the dataset

train.drop("Cabin",axis=1,inplace=True)

test.drop("Cabin", axis=1, inplace=True)
sex_train = pd.get_dummies(train['Sex'],drop_first=True)

sex_test = pd.get_dummies(test['Sex'], drop_first=True)



train.drop(['Sex'],axis=1,inplace=True)

test.drop(['Sex'],axis=1,inplace=True)



train = pd.concat([train,sex_train],axis=1)

test = pd.concat([test, sex_test], axis=1)
test["Fare"].fillna(test["Fare"].median(), inplace=True)

# Function to Impute Age

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
# Apply the above function to our training and testing datasets

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)



train['Age'] = train['Age'].astype(int)

test['Age']    = test['Age'].astype(int)

train.head()
test.head()
X_train = train.drop('Survived', axis=1)

y_train = train['Survived']

X_test = test.drop('PassengerId', axis=1)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

Prediction = lm.predict(X_test)



lm.score(X_train,y_train)
from sklearn.linear_model import LogisticRegression



log = LogisticRegression()



log.fit(X_train, y_train)



Prediction = log.predict(X_test)



log.score(X_train, y_train)
from sklearn.svm import SVC

svc = SVC()



svc.fit(X_train, y_train)



Prediction_SVM = svc.predict(X_test)



svc.score(X_train, y_train)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 1)



knn.fit(X_train, y_train)



Prediction_KNC = knn.predict(X_test)



knn.score(X_train, y_train)
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()



gaussian.fit(X_train, y_train)



Prediction_G = gaussian.predict(X_test)



gaussian.score(X_train, y_train)
from sklearn.ensemble import GradientBoostingClassifier

gradient_boost = GradientBoostingClassifier(n_estimators=100)

gradient_boost.fit(X_train, y_train)



Prediction_GBC = gradient_boost.predict(X_test)



gradient_boost.score(X_train, y_train)
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

RFC_prediction = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": RFC_prediction

    })

submission.to_csv('Result_update.csv', index=False)