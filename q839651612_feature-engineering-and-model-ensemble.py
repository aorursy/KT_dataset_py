import numpy as np 

import pandas as pd 

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from lightgbm import LGBMClassifier

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
titanic_train = pd.read_csv("../input/titanic/train.csv")

titanic_test = pd.read_csv("../input/titanic/test.csv")
# Combine train and test data

y_train = titanic_train['Survived']

titanic_train = titanic_train.drop(['Survived'], axis = 1)

split_ind = len(titanic_train)

x = pd.concat([titanic_train, titanic_test], axis = 0)
# many missing values in Age and Cabin

titanic_train.info()

print('-'*30)

titanic_test.info()
# Cabin -> binary variable

x['Has_Cabin'] = x['Cabin'].apply(lambda x: 1 if type(x) == str else 0)



# Fill Fare and Embarked

x['Fare'] = x['Fare'].fillna(x['Fare'].mean())

x['Embarked'] = x['Embarked'].apply(lambda var: 'S' if type(var) == float else var)



# Extract info from Name

x['Title'] = x['Name'].apply(lambda var: (var.split(',')[1][1:]).split('.')[0])

#x['Last Name'] = x['Name'].apply(lambda var: var.split(',')[0])



# Sex => binary numerical variables(female:1, male:0) 

x['Sex'] = x['Sex'].astype('str')

x['Gender'] = x['Sex'].apply(lambda var: 1 if var == 'female' else 0)



# Create a new vaiable: passenger is alone or not

x['Alone'] = x['SibSp'] + x['Parch']

x['Alone'] = x['Alone'].apply(lambda var: 1 - np.ceil(var / 10))



# Fill the missing value in Age by Title

df = x[['Age', 'Title']]

dic ={}

for row in df.itertuples():

    if not np.isnan(row.Age):

        t = row.Title

        if t in dic:

            dic[t][0] += 1

            dic[t][1] += row.Age

        else:

            dic[t] = [1, row.Age]



loa = list(x['Age'])

lot = list(x['Title'])

for i in range(len(lot)):

    ele = lot[i]

    if np.isnan(loa[i]):

        loa[i] = dic[ele][1] / dic[ele][0]

        

x['Age1'] = loa



# Categorical -> Numerical

x['Embarked'] = x['Embarked'].map({'S': 1, 'C': 2, 'Q':3})

RareTitle = x['Title'].value_counts().index[4:]

x['Title'] = x['Title'].apply(lambda x: 'Rare' if x in RareTitle else x).map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5})



# Drop columns

x = x.drop(['Age', 'Sex', 'Name', 'PassengerId', 'Cabin', 'Ticket'], axis = 1)
X_train = x[:split_ind]

X_test = x[split_ind:]



# Ensemble

RF = RandomForestClassifier(n_estimators=500)

RF.fit(X_train, y_train)

RF_train = RF.predict(X_train)

RF_test = RF.predict(X_test)

print(RF.score(X_train, y_train))

print('-'*30)



GB = GradientBoostingClassifier(n_estimators=500)

GB.fit(X_train, y_train)

GB_train = GB.predict(X_train)

GB_test = GB.predict(X_test)

print(GB.score(X_train, y_train))

print('-'*30)



ET = ExtraTreesClassifier(n_estimators=500)

ET.fit(X_train, y_train)

ET_train = ET.predict(X_train)

ET_test = ET.predict(X_test)

print(ET.score(X_train, y_train))
# Combine first level predictions as second level input

Ensemble_train = pd.DataFrame({'RandomForest': RF_train, 'ExtraTrees': ET_train, 'GradientBoosting': GB_train})

Ensemble_test = pd.DataFrame({'RandomForest': RF_test, 'ExtraTrees': ET_test, 'GradientBoosting': GB_test})



lgb = LGBMClassifier()

lgb.fit(Ensemble_train, y_train)

Y = lgb.predict(Ensemble_test)

lgb.score(Ensemble_train, y_train)
submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": Y

    })

submission.to_csv('titanic.csv', index=False)