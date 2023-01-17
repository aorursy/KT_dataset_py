# Titanic dataset

# Humberto Barrantes

# 09-2020


import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

train = pd.read_csv("/kaggle/input/titanic/train.csv")



test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.info()
train.head()
test.info()
test.head()
sns.heatmap(train.isnull(), cbar=False)
sns.heatmap(test.isnull(), cbar=False)
for df in [test, train]:



    # fill embarked with a constant S

    df['Embarked'] = df['Embarked'].fillna('S')

    

    # fill cabin with a constant CX or no cabin

    df['Cabin'] = df['Cabin'].fillna('CX')



    # fill Fare with the mean value

    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

    
from sklearn.impute import KNNImputer



imputer = KNNImputer(n_neighbors=3, weights="uniform")



features = ['Age','SibSp','Parch', 'Fare']



for df in [train, test]:



    imputer.fit(df[features])

    df[features] = imputer.transform(df[features])
import re



other_titles = ['Dr.','Rev.','Col.','Major.','Mlle.','Jonkheer.','Countess.','Capt.','Sir.','Lady.','Don.','Dona.','Mme.','Ms.']



for df in [train, test]:



    df['Title'] = df['Name'].apply(lambda x: re.search("[a-zA-Z]*\.", x).group())



    df['Title'] = df['Title'].apply(lambda x: "Other" if x in other_titles else x)

    '''

    df['Title'] = df['Title'].replace(

        ['Dr.','Rev.','Col.','Major.','Mlle.','Jonkheer.','Countess.','Capt.','Sir.','Lady.','Don.','Dona.','Mme.','Ms.'], 

        'Other'

    )

    '''
for df in [train, test]:

    

    # Family Size is the sum of Siblings and Parents

    df['FamilySize'] = df['SibSp'] + df['Parch']

    

    # If the Passenger didn't have any familiar with him/her, he/she was alone

    df['IsAlone'] = np.where(df['FamilySize'] > 0, 0, 1)

    

    # Did the passenger had a Cabin?

    df['HadCabin'] = np.where(df['Cabin'] != "CX", 1, 0)    

    
train = train.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis = 1)

test = test.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis = 1)
train.head()
from sklearn.preprocessing import OrdinalEncoder



for col in ['Sex', 'Embarked', 'Title']:

    

    encoder = OrdinalEncoder()

    encoder.fit(train[[col]])

    

    train[[col]] = encoder.transform(train[[col]])

    test[[col]] = encoder.transform(test[[col]])
X_train = train.drop(['PassengerId', 'Survived'], axis = 1)

y_train = train['Survived']



X_test = test.drop(['PassengerId'], axis = 1)
X_train.head()
X_test.head()
from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(max_depth=2, random_state=0, )



rfc_model.fit(X_train, y_train)



scores = cross_val_score(rfc_model, X_train, y_train, cv=5)



final_score = rfc_model.score(X_train, y_train)



print(f"Scores: {scores} \nMean: {scores.mean()} \nFinal Score: {final_score}")
predictions = rfc_model.predict(X_test)
submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':predictions})



submission.head()
filename = 'gender_submission.csv'



submission.to_csv(filename,index=False)