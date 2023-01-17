import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head(20)
Family_tot = train_data["SibSp"] + train_data["Parch"]

train_data["Family_tot"] = Family_tot

train_data.head()
Family_test = test_data["SibSp"] + test_data["Parch"]

test_data["Family_tot"] = Family_test

sns.heatmap(train_data.corr(), cmap='Blues')
sns.heatmap(test_data.corr(), cmap='Blues')
num_correlation = train_data.select_dtypes(exclude='object').corr()

corr = num_correlation.corr()

print(corr['Survived'].sort_values(ascending=False))
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
train_data['Age'] = train_data[['Age','Pclass']].apply(impute_age,axis=1)
Faremean=test_data.loc[:,"Fare"].mean()

print(Faremean)

test_data['Fare'].fillna(Faremean,inplace = True)
test_data['Age'] = test_data[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(test_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
test_data.info()
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass","Sex","Age", "Family_tot", "Fare"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")