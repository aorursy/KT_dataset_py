import numpy as np

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("../input/titanic/train.csv", index_col='PassengerId')

train_data
for col in train_data.columns:

    print(col, len(train_data[col].unique()))
train_data.drop(columns=['Name', 'Cabin', 'Ticket'], inplace=True)

train_data
total = train_data.isnull().sum()



percent_1 = train_data.isnull().sum()/train_data.isnull().count()*100



percent_2 = (round(percent_1, 1))



missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data
age_mean = train_data.Age.mean()

train_data.Age.fillna(age_mean, inplace=True)



train_data.isna().sum()
train_data['Embarked'].describe()
train_data.Embarked.fillna(train_data.Embarked.mode()[0], inplace=True)

train_data.isna().sum()
train_data.Embarked.replace('C', 0, inplace=True)

train_data.Embarked.replace('S', 1, inplace=True)

train_data.Embarked.replace('Q', 2, inplace=True)



train_data
train_data.Age.dtype
test_data = pd.read_csv("../input/titanic/test.csv", index_col='PassengerId')
data = [train_data, test_data]



for dataset in data:

    dataset['Age'] = dataset['Age'].fillna(0)

    dataset['Age'] = dataset['Age'].astype(int)

    

train_data    
test_data.Age
train_data.info()
train_data.Sex.replace('male', 0, inplace=True)

train_data.Sex.replace('female', 1, inplace=True)



train_data
print(train_data.Sex.value_counts())

print('----------------------')

print(train_data.groupby('Sex').Survived.value_counts())
import seaborn as sns

import matplotlib.pyplot as plt



sns.barplot(x='Sex', y='Survived', data=train_data)
train_data.Embarked.value_counts().plot(kind='bar')

plt.title("Passengers per boarding location");
Survived_Pcalss = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train_data, kind="bar")

Survived_Pcalss = Survived_Pcalss.set_ylabels("survival probability")
test_data = pd.read_csv("../input/titanic/test.csv", index_col='PassengerId')



test_data.drop(columns=['Name', 'Cabin', 'Ticket'], inplace=True)

test_data
test_data.isna().sum()
test_data.Age.fillna(age_mean, inplace=True)

test_data.Fare.fillna(test_data.Fare.mean(), inplace=True)



test_data
test_data.Sex.replace('male', 0, inplace=True)

test_data.Sex.replace('female', 1, inplace=True)



test_data.Embarked.replace('C', 0, inplace=True)

test_data.Embarked.replace('S', 1, inplace=True)

test_data.Embarked.replace('Q', 2, inplace=True)



test_data
X_train = train_data.drop("Survived", axis=1)

y_train = train_data["Survived"]



X_train
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=1)



RFC_model = RandomForestClassifier(criterion='gini', 

                             n_estimators=100,

                             random_state=1,

                             n_jobs=-1)

RFC_model.fit(X_train, y_train)



y_prediction = RFC_model.predict(X_test)



RFC_model.score(X_train, y_train)

acc_RFC = round(RFC_model.score(X_train, y_train) * 100, 2)



acc_RFC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)



y_predict = RFC_model.predict(X_test)

accuracy_score(y_test, y_predict)



print(classification_report(y_test, y_predict))

from sklearn.tree import DecisionTreeClassifier



decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, y_train)  

y_prediction = decision_tree.predict(X_test) 



acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)



acc_decision_tree
y_predict = decision_tree.predict(X_test)

accuracy_score(y_test, y_predict)



print(classification_report(y_test, y_predict))
results = pd.DataFrame({

    'Model': ['Random Forest','Decision Tree'], 'Score': [acc_RFC, acc_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(2)
f_model = DecisionTreeClassifier()

f_model.fit(X_train, y_train)



preds = f_model.predict(test_data)



test_data.shape
test_output = pd.DataFrame({

    'PassengerId': test_data.index, 

    'Survived': preds

})

test_output.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')



submission