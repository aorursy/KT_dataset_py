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
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.head()
train_data['Name'].head()
def get_featuresdata(data):
    data.loc[data['Embarked']==0,['Embarked']] = 'S'
    data['Embarked_Num'] = data['Embarked'].map({'S':1,'C':2,'Q':3}).astype(int)

    data['Sex_Num'] = data['Sex'].map({'male':0,'female':1}).astype(int)

    data['family_Num'] = data['SibSp'] + data['Parch']

    data['Fare_Num']=0
    data.loc[ data['Fare'] <= 7.91, 'Fare_Num'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare_Num'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare_Num']   = 2
    data.loc[ data['Fare'] > 31, 'Fare_Num'] = 3
    data['Fare_Num'] = data['Fare_Num'].astype(int)

    data['Age_Num'] = 0
    data.loc[ data['Age'] <= 16, 'Age_Num'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age_Num'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age_Num'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age_Num'] = 3
    data.loc[ data['Age'] > 64, 'Age_Num'] = 4
    data['Age_Num'] = data['Age_Num'].astype(int)

    data['Cabin_Num'] = 0
    data.loc[data['Cabin'].notnull(),['Cabin_Num']] = 1

    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    data.loc[data['Title'].isnull(),'Title'] = 0
    data['Title_Num'] = 0
    data['Title_Num'] = data['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    
    return data
train_data['Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_data['Title'], train_data['Sex'])
train_data['Title'] = train_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')
pd.crosstab(train_data['Title'], train_data['Sex'])
train_data.loc[train_data['Title'].isnull(),'Title'] = 0
train_data['Title_Num'] = 0
train_data['Title_Num'] = train_data['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
feature_names = ['Cabin_Num','Embarked_Num','Sex_Num','family_Num','Fare_Num','Age_Num','Title_Num']
X = train_data[feature_names]
y = train_data['Survived']

predict_X = get_featuresdata(test_data)
predict_X = predict_X[feature_names]

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X,y)
predict_y = clf.predict(predict_X)
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predict_y
    })
#submission.to_csv('../input/submission.csv', index=False)