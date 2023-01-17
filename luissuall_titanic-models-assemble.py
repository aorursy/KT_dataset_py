import numpy as np # linear algebra

import pandas as pd # data processing

import csv

from sklearn.ensemble import RandomForestClassifier
traindata_raw = pd.read_csv("../input/train.csv")

testdata_raw = pd.read_csv("../input/test.csv")



#Output some data as an example

print("Train data example")

print(traindata_raw.head(3))



print("\n\nTest data example")

print(testdata_raw.head(3))



print(traindata_raw["Name"])
print("Raw data structure:")

print(traindata_raw.dtypes)



print("\n\nBasic statistics:")

print(traindata_raw.describe())
traindata = traindata_raw.copy()



traindata.drop(['PassengerId','Ticket','Cabin'], axis=1, inplace=True)



traindata["Sex"] = traindata['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

embarkeddummies = pd.get_dummies(traindata['Embarked'],prefix='Embarked_at')

traindata = pd.concat([traindata,embarkeddummies],axis=1)



traindata.drop(['Embarked'], axis=1, inplace=True)



print(traindata.dtypes)

print(traindata.head(3))
import re

from collections import Counter



def get_std_titles(names):



    raw_titles = []

    for name in names:

        m = re.search(', (.+?)\. ', name)

        if m:

            title = m.group(1)

        else:

            title = ''

        raw_titles.append(title)



    titles = []

    for title in raw_titles:

        if title in ['Mrs','Ms','Mme']:

            std_title = 'Mrs'

        elif title in ['Miss', 'Mlle']:

            std_title = 'Miss'

        elif title in ['Mr']:

            std_title = 'Mr'

        elif title in ['Master']:

            std_title = 'Master'

        elif title in ['Dr','Rev']:

            std_title = 'Pro'

        elif title in ['Major','Col','Capt']:

            std_title = 'Army'

        elif title in ['Lady', 'the Countess', 'Jonkheer', 'Sir', 'Don', 'Dona']:

            std_title = 'Royal'

        else:

            std_title = 'Other'



        titles.append(std_title)

    

    return titles



titles = get_std_titles(traindata['Name'])



print(Counter(titles))
traindata['Title'] = titles



titledummies = pd.get_dummies(traindata['Title'],prefix='Title')

traindata = pd.concat([traindata,titledummies],axis=1)



traindata.drop(['Title','Name'], axis=1, inplace=True)

print(traindata.info())
traindata['Age'].fillna(traindata['Age'].mean(),inplace=True)

print(traindata.info())
testdata = testdata_raw.copy()



testdata.drop(['PassengerId','Ticket','Cabin'], axis=1, inplace=True)



testdata["Sex"] = testdata['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

embarkeddummies = pd.get_dummies(testdata['Embarked'],prefix='Embarked_at')

testdata = pd.concat([testdata,embarkeddummies],axis=1)



testdata.drop(['Embarked'], axis=1, inplace=True)



titles = get_std_titles(testdata['Name'])

print(Counter(titles))

testdata['Title'] = titles



titledummies = pd.get_dummies(testdata['Title'],prefix='Title')

testdata = pd.concat([testdata,titledummies],axis=1)



testdata.drop(['Name','Title'], axis=1, inplace=True)



testdata['Age'].fillna(traindata['Age'].mean(), inplace=True)

testdata['Fare'].fillna(traindata['Fare'].mean(), inplace=True)



print(testdata.info())
traindata_values = traindata.values

testdata_values = testdata.values



my_rf = RandomForestClassifier(n_estimators=200)

my_rf.fit(traindata_values[0::,1::],traindata_values[0::,0])

rf_predictions = my_rf.predict(testdata_values)
#Support Vector Machine

from sklearn.neural_network import MLPClassifier

my_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,

                    hidden_layer_sizes=(20, 20, 5), random_state=1)

my_mlp.fit(traindata_values[0::,1::],traindata_values[0::,0]) 

mlp_predictions = my_mlp.predict(testdata_values)



#AdaBoost

from sklearn.ensemble import AdaBoostClassifier

my_ada = AdaBoostClassifier(n_estimators=100)

my_ada.fit(traindata_values[0::,1::],traindata_values[0::,0]) 

ada_predictions = my_ada.predict(testdata_values)
rf_result = pd.DataFrame({

        "PassengerId": testdata_raw["PassengerId"],

        "Survived": rf_predictions.astype('int')

    })

rf_result.to_csv("basicRFModel.csv", index=False)



mlp_result = pd.DataFrame({

        "PassengerId": testdata_raw["PassengerId"],

        "Survived": mlp_predictions.astype('int')

    })

mlp_result.to_csv("basicMLPModel.csv", index=False)



ada_result = pd.DataFrame({

        "PassengerId": testdata_raw["PassengerId"],

        "Survived": ada_predictions.astype('int')

    })

ada_result.to_csv("basicADAModel.csv", index=False)
added_predictions = rf_predictions + mlp_predictions + ada_predictions

ensemble_predictions = [0 if x < 2 else 1 for x in added_predictions]



result_ensemble = pd.DataFrame({

        "PassengerId": testdata_raw["PassengerId"],

        "Survived": ensemble_predictions

    })



result_ensemble.to_csv("ensembleModel.csv", index=False)