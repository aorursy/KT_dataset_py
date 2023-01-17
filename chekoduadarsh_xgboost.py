# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()


test_data.loc[test_data['Sex'] == "male", 'Sex'] = 1

test_data.loc[test_data['Sex'] == "female", 'Sex'] = 0



train_data.loc[train_data['Sex'] == "female", 'Sex'] = 0

train_data.loc[train_data['Sex'] == "male", 'Sex'] = 1



train_data['Age'] = train_data['Age'].fillna(0)

test_data['Age'] = test_data['Age'].fillna(0)

train_data['Age'] = train_data['Age'].fillna(0)

test_data['Age'] = test_data['Age'].fillna(0)

train_data['Cabin'] = train_data['Cabin'].fillna(0)

test_data['Cabin'] = test_data['Cabin'].fillna(0)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

def heatMap(df):

    #Create Correlation df

    corr = df.corr()

    #Plot figsize

    fig, ax = plt.subplots(figsize=(10, 10))

    #Generate Color Map

    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    #Generate Heat Map, allow annotations and place floats in map

    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")

    #Apply xticks

    plt.xticks(range(len(corr.columns)), corr.columns);

    #Apply yticks

    plt.yticks(range(len(corr.columns)), corr.columns)

    #show plot

    plt.show()
features = ["Pclass", "Sex", "SibSp", "Parch","Age","Fare","Survived"]

corr_data = pd.get_dummies(train_data[features])



heatMap(corr_data)
model = XGBClassifier()

y = train_data["Survived"]







features = ["Pclass", "Sex","Age","Fare"]

X = pd.get_dummies(train_data[features])

#X = test_data.dropna()

X_test = pd.get_dummies(test_data[features])



X_test.loc[X_test['Sex'] == "female", 'Sex'] = 0

X_test.loc[X_test['Sex'] == "male", 'Sex'] = 1





model.fit(X, y)


model.score(X, y)
predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")