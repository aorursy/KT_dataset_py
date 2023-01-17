# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

__input__ = '../input'

__output__ = '../output'
def replaceNANavieBayes(data,classlabel):

    from sklearn import tree

    train = data[data[classlabel].notnull()]

    test = data[data[classlabel].isnull()]

    Xtrain = train.drop(classlabel,axis=1)

    ytrain = train[classlabel]

#     print(Xtrain.info())

    model = (tree.DecisionTreeRegressor()).fit(Xtrain,ytrain)

    return model.predict(test.drop(classlabel,axis=1))



def preprocessor(dataset):

    from sklearn import preprocessing

    sexEncoded = (preprocessing.LabelEncoder()).fit(dataset['Sex'])

    embarkedEncoded = (preprocessing.LabelEncoder()).fit((dataset[dataset['Embarked'].notnull()])['Embarked'])

    cabinEncoded = (preprocessing.LabelEncoder()).fit((dataset[dataset['Cabin'].notnull()])['Cabin'])

    #Cleaning

    dataset['Sex'] = sexEncoded.transform(dataset['Sex'])

#     print(dataset.isna().Cabin.value_counts())

#     print(dataset.isna().Age.value_counts())

#     print(dataset.isna().Embarked.value_counts())



    # replacing na Cabin,Age,Embarked

    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].value_counts().index[0])

    dataset['Embarked'] = embarkedEncoded.transform(dataset['Embarked'])

    dataset.loc[dataset['Cabin'].notnull(),'Cabin'] = cabinEncoded.transform(dataset.loc[dataset['Cabin'].notnull(),'Cabin'])

    dataset.loc[dataset['Age'].isna(),'Age'] = replaceNANavieBayes(dataset.drop(['Cabin','Fare'],axis=1),'Age')

    if len(dataset[dataset['Fare'].isna()]) > 0:

        dataset.loc[dataset['Fare'].isna(),'Fare'] = replaceNANavieBayes(dataset.drop('Cabin',axis=1),'Fare')

#     print(dataset.info())

    dataset.loc[dataset['Cabin'].isna(),'Cabin'] = replaceNANavieBayes(dataset,'Cabin')

    

    print(dataset)

    return dataset
dataset = pd.read_csv(__input__+'/train.csv')

dataset = dataset.drop(['PassengerId','Name','Ticket'],axis=1)

print(dataset.info())

dataset = preprocessor(dataset)

# print(len(dataset[dataset['Fare'].isna()]))
from sklearn import tree



X = dataset.drop('Survived',axis=1)

y = dataset['Survived']



decisiontree = tree.DecisionTreeClassifier()



model = decisiontree.fit(X,y)



model
real = pd.read_csv(__input__+'/test.csv')

dataset = real.drop(['PassengerId','Name','Ticket'],axis=1)

print(dataset.info())

test = preprocessor(dataset)



# dataset['Age'].isnull()
predictions = model.predict(test)

print(predictions)

output = pd.DataFrame({

    'PassengerId':real['PassengerId'],

    'Survived':predictions

})

output.to_csv('output.csv', index = False)