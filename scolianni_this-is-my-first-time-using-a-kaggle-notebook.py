# The first order of business, as always, is to read in the mother fucking data



import pandas as pd

dfTrain = pd.read_csv('../input/train.csv')

dfTest = pd.read_csv('../input/test.csv')
dfTrain.head()
# Assign default values for each data type

defaultInt = -1

defaultString = 'NaN'

defaultFloat = -1.0



# Create lists by data tpe

intFeatures = ['PassengerId', 'Survived', 'Pclass', 'SibSp', 'Parch']

stringFeatures = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

floatFeatures = ['Age', 'Fare']



# Clean the NaN's

for feature in list(dfTrain):

    if feature in intFeatures:

        dfTrain[feature] = dfTrain[feature].fillna(defaultInt)

    elif feature in stringFeatures:

        dfTrain[feature] = dfTrain[feature].fillna(defaultString)

    elif feature in floatFeatures:

        dfTrain[feature] = dfTrain[feature].fillna(defaultFloat)

    else:

        print('Error: Feature %s not recognized.' % feature)

    

for feature in list(dfTest):

    if feature in intFeatures:

        dfTest[feature] = dfTest[feature].fillna(defaultInt)

    elif feature in stringFeatures:

        dfTest[feature] = dfTest[feature].fillna(defaultString)

    elif feature in floatFeatures:

        dfTest[feature] = dfTest[feature].fillna(defaultFloat)

    else:

        print('Error: Feature %s not recognized.' % feature)
from sklearn.preprocessing import LabelEncoder

dfCombined = pd.concat([dfTrain, dfTest])

for feature in list(dfCombined):

    

    le = LabelEncoder()

    le.fit(dfCombined[feature])

    

    if feature in dfTrain:

        if feature != 'PassengerId':

            dfTrain[feature] = le.transform(dfTrain[feature])

    if feature in dfTest:

        if feature != 'PassengerId':

            dfTest[feature] = le.transform(dfTest[feature])
from sklearn.model_selection import train_test_split



X = dfTrain.drop(['Survived'], axis=1)

y = dfTrain['Survived']



num_test = 0.20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=num_test, random_state=23)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
from sklearn.metrics import accuracy_score



accuracy = accuracy_score(y_test, predictions)



print(accuracy)
# Generate predictions

clf = RandomForestClassifier()

clf.fit(X, y)

dfTestPredictions = clf.predict(dfTest)



# Write predictions to csv file

results = pd.DataFrame({'PassengerId': dfTest['PassengerId'], 'Survived': dfTestPredictions})

results.to_csv('results.csv', index=False)

results.head()