# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



#combine = [train_data, test_data]
print("----- train data -----")

print(train_data.isnull().sum())



print("----- test data ------")

print(test_data.isnull().sum())
train_data["Embarked"].mode()
def data_preprocess(data): 

    p_data = data.drop(['Name','PassengerId','Ticket'],axis=1)

    p_data['no_cabin'] = pd.isnull(p_data['Cabin'])

    p_data = p_data.drop(['Cabin'],axis=1)

    

    p_data['Age'] = p_data['Age'].fillna(p_data['Age'].median())

    p_data['Embarked'] = p_data['Embarked'].fillna('S')

    p_data['Fare'] = p_data['Fare'].fillna(p_data['Fare'].mean())

    

    p_data['Sex'] = p_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    p_data['Embarked'] = p_data['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)

    p_data['no_cabin'] = p_data['no_cabin'].map( {False: 0, True: 1} ).astype(int)

    

    return p_data

train_data = data_preprocess(train_data)



train_target = train_data['Survived']

train_features = train_data.drop(['Survived'],axis=1)



train_features, valid_features, train_target, valid_target = train_test_split(train_features,train_target,test_size=0.2,random_state=40)



print(train_features.head())

print("\n\n")

print(train_target.head())
train_features.describe()
test_features = data_preprocess(test_data)



print(test_features.head())
modelr = RandomForestClassifier(n_estimators=10, min_samples_leaf=2)



modelr.fit(train_features, train_target)



modelr.score(train_features, train_target)
modelr.score(valid_features, valid_target)
test_prediction = modelr.predict(test_features)
test_prediction
submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": test_prediction

    })



submission.to_csv("submission.csv", index=False)