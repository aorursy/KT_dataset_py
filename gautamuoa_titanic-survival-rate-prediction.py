#import

import pandas as pd

import numpy as np

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

import seaborn as sns
#read input

dataTrain = pd.read_csv('../input/train.csv')

dataTest = pd.read_csv('../input/test.csv')

fullDataset = [dataTrain,dataTest]
#feature engineering

for dataset in fullDataset:

    #encode Sex

    dataset['Sex'] = LabelEncoder().fit_transform(dataset['Sex'])

    #encode Embarked

    dataset['Embarked'] = LabelEncoder().fit_transform(dataset['Embarked'].fillna('0'))

    #fill missing data of Cabin

    dataset['Cabin']=dataset['Cabin'].fillna('U')

    #get the first letter from the cabin name

    dataset['Cabin'] = dataset['Cabin'].apply(lambda x: x[0])

    #encode the cabin

    dataset['Cabin'] = LabelEncoder().fit_transform(dataset['Cabin'])

    #impute missing data with mean

    dataset['Age']=dataset['Age'].fillna(dataset['Age'].mean())

    #encode Age

    dataset['Age'].loc[dataset['Age'] < 16] = 0

    dataset['Age'].loc[(dataset['Age'] > 16) & (dataTrain['Age'] <= 32)] = 1

    dataset['Age'].loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48)] = 2

    dataset['Age'].loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64)] = 3

    dataset['Age'].loc[(dataset['Age'] > 64)] = 4

    #impute missing data with mean

    dataset['Fare']=dataset['Fare'].fillna(dataset['Fare'].mean())

    #encode Fare

    dataset['Fare'].loc[dataset['Fare'] < 7.91] = 0

    dataset['Fare'].loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454)] = 1

    dataset['Fare'].loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31)] = 2

    dataset['Fare'].loc[(dataset['Fare'] > 31)] = 3

    #create new column name FamilySize

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    #create new column name IsAlone for alone travellers and inintialize it 0

    dataset['IsAlone'] = 0

    #if familysize is 1 then person is travelling alone

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    #get the title of each passenger

    dataset['Title']=[i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]

    #Replace Title with appropriate value

    dataset['Title'] = dataset['Title'].replace(['Lady','the Countess' ,'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    #encode Title

    dataset['Title'] = LabelEncoder().fit_transform(dataset['Title'])

dataTrain = dataTrain.drop(['Name', 'PassengerId', 'Ticket','SibSp'], axis=1)

dataTest = dataTest.drop(['Name',  'Ticket','SibSp'], axis=1)
X = dataTrain.drop("Survived", axis=1)

Y = dataTrain["Survived"]

# prepare configuration for cross validation test harness

seed = 7

# prepare models which we are going to use

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('Neural Network', MLPClassifier()))

models.append(('Random Forest', RandomForestClassifier(n_estimators=500, max_depth=10,oob_score=True )))

# evaluate each model in turn

results = []

names = []

meanResults = []

scoring = 'accuracy'

print('Algorithm: Accuracy Standard_Deviation')

for name, model in models:

    kfold = model_selection.KFold(n_splits=20, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold) #gives accuracy

    results.append(cv_results)

    names.append(name)    

    print(name, cv_results.mean(), cv_results.std())

    meanResults.append(cv_results.mean())

    

#plot the results

temp = pd.DataFrame({

        "names":names,

        "results":meanResults

        })

temp = temp.sort_values(by=['results'], ascending=False)

g = sns.factorplot(x='results',y='names',kind='bar',data=temp,color='blue',alpha=0.3,           

               size=3,aspect=2).set(title='Performance of each algorithm',ylabel='Algorithms',xlabel='Accuracy')
