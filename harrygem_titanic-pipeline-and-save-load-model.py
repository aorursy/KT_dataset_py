# import necessary libraries

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import dill as pickle
from sklearn.base import BaseEstimator, TransformerMixin
#build your own pipeline, use BaseEstimator and TransformerMinin, then finish the transform() and fit() functions
class PreProcessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, dataset):
        # check below link to understand the data cleansing process.
        # https://www.kaggle.com/startupsci/titanic-data-science-solutions
        dataset = dataset.drop(['Ticket', 'Cabin'], axis=1)
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
                                                         'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

        dataset = dataset.drop(['Name'], axis=1)

        dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
        dataset['Age'] = dataset['Age'].fillna(40).astype(int)

        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

        dataset = dataset.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

        dataset['Embarked'] = dataset['Embarked'].fillna("S")

        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

        dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)

        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

        # make sure to return a narray instead of DataFrame
        return dataset.as_matrix()
    
    def fit(self, dataset, y=None):
        return self
# create the pipeline: preprocessing first, then RF model
pipe = Pipeline([('preprocessing', PreProcessing()), ('RF', RandomForestClassifier())])
pipe
# load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
X_train = df_train.drop(["Survived", "PassengerId"], axis=1)
Y_train = df_train["Survived"]
X_test  = df_test.drop("PassengerId", axis=1).copy()
# fit the pipeline with Training features data and label data
pipe.fit(X_train, Y_train)
#After fit the pipeline, the model is ready, we use predict() to make a prediction, 
#Here X_test is still raw data, but it will be processed by Preprocessing class, then go to the model.
pipe.predict(X_test.loc[0:15])
# Use pickle to save model for next usage.
filename = 'model_v1.pk'
with open('./'+filename, 'wb') as file:
    pickle.dump(pipe, file) 
# Open saved model, and directly make the prediction with new data
with open('./'+filename ,'rb') as f:
    loaded_model = pickle.load(f)
loaded_model.predict(X_test.loc[0:15])
