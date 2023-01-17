import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



def isboss(name):

    for title in ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']:

        if(title in name):

            return 1

    if('Miss.' in name or 'Mlle.' in name or 'Ms' in name):

        return 2

    if('Master.' in name):

        return 3

    if('Mrs.' in name or 'Mme.' in name):

        return 4

    return 0

def process_data(fileName):

    df = pd.read_csv('../input/'+fileName, header = 0)

    # convert sex to gender float

    df['Gender'] = df.Sex.map({'female':0, 'male':1}).astype(int)

    # fill out age 

    # calculate default age median based on Pclass

    median_ages = np.zeros((2,3))



    for i in range(0,2):

        for j in range(0,3):

            median_ages[i,j] = df[(df['Gender'] == i) & \

                                  (df['Pclass'] == j+1)]['Age'].dropna().median()

        

    df['AgeFill'] = df['Age']       

    for i in range(0, 2):

        for j in range(0, 3):

            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\

                    'AgeFill'] = median_ages[i,j]

    df['FamiliySize'] = df['Parch'] + df['SibSp']

    df['hasFamily'] = df.FamiliySize.map(lambda x : x > 0).astype(int)

    df['AgeFill'] = df['AgeFill']/10

    df['Title'] = df['Name'].map(lambda name : isboss(name)).astype(int)

    df['Embarked'] = df.Embarked.fillna("S").map({'S':2,'C':3,'Q':1}).astype(int)

    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin','Age','PassengerId', 'Parch', 'SibSp','FamiliySize',], axis=1) 

    df.loc[df.Fare.isnull(),'Fare'] = df.Fare.dropna().median()

    df['Fare'] = df['Fare']/df['Embarked']

    #df = df.dropna()

    df.describe()

    # train_data = df.values

    return df
train_data = process_data('train.csv')



test_data = process_data('test.csv')
logReg = LogisticRegression(random_state=1)

# kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

# scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)



# print(scores.mean())



logReg.fit(train_data[['Gender','AgeFill','hasFamily','Title','Fare','Pclass']], train_data["Survived"])



output = logReg.predict(test_data[['Gender','AgeFill','hasFamily','Title','Fare','Pclass']])
orig = pd.read_csv('../input/test.csv',header= 0)



submission = pd.DataFrame({"PassengerId": orig.PassengerId,"Survived": output.astype(int)})



submission.to_csv('titanic.csv', index=False)

        