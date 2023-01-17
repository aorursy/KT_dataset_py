import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

import seaborn  as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



trainData = train

testData = test 
#Combined test & train dataset so that we don’t have repeat same process for test and train dataset simultaneously.

combined = [trainData, testData]

cols =['PassengerId', 'Name', 'Ticket',  'Cabin', 'Embarked']

trainData = trainData.drop(labels= cols,axis = 1)

testData = testData.drop(labels= cols,axis = 1)
combined = [trainData, testData]
sns.barplot(x='Pclass', y='Survived', data=train)
sns.barplot(x='Sex', y='Survived', data=train)
sns.barplot(x='Embarked', y='Survived', data=train)
grid = sns.FacetGrid(train, col='Survived', row='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
def coeff(model , data):

    coeff_df = pd.DataFrame(data.columns.delete(0))

    coeff_df.columns = ['Feature']

    coeff_df["Correlation"] = pd.Series(model.coef_[0])

    return coeff_df.sort_values(by='Correlation', ascending=False)


for dataset in combined:

    #Before data 

    dataset.info()

    

    #Replace male with 1 and female with 0    

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    

    #As Age & Fare data is missing we will replacing NA data with median value of the same 

    dataset['Age'].fillna(testData['Age'].dropna().median(), inplace=True) 

    dataset['Fare'].fillna(testData['Fare'].dropna().median(), inplace=True)

    

    #After data 

    dataset.info()


#to beter understand the above exaple

trainData['AgeRange'] = pd.cut(trainData['Age'], 4)

Age_and_Survived = trainData[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='AgeRange', ascending=True)

sns.barplot(x='AgeRange', y='Survived', data=Age_and_Survived)

plt.ylabel('Survived %')


#as the age reange is just  to show age range so we will drop that col

trainData.drop(labels=['AgeRange'], axis = 1 , inplace=True)
for dataset in combined: 

    dataset['Age'] = pd.cut(dataset['Age'], 4, labels=[0,1,2,3])

    dataset['Fare'] = pd.cut(dataset['Fare'], 5, labels=[0,1,2,3,4])



    #Creating family size 

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

combined = [trainData, testData]
#Then identifying that is the passenger is alone or not if alone 0 else 1

for dataset in combined:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
#Removing ['Parch', 'SibSp', 'FamilySize'] columns as we have created is Alone column. So we don’t need them anymore.

trainData = trainData.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

testData = testData.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combined = [trainData, testData]
#Preparing train and test data for model building and testing

X_train = trainData.drop("Survived", axis=1)

Y_train = trainData["Survived"]

X_test  = testData

X_train.shape, Y_train.shape, X_test.shape
log_model = LogisticRegression()

log_model.fit(X_train, Y_train)

Y_pred = log_model.predict(X_test)

acc_log_model = round(log_model.score(X_train, Y_train) * 100, 2)

acc_log_model
from IPython.display import Image

Image("../input/coeff/after_coeff.JPG")
Image("../input/coeff/before_coeff.JPG")
coeff(log_model,trainData)
sns.barplot(x = 'Feature', y = 'Correlation' , data = coeff(log_model,trainData))

knn_model = KNeighborsClassifier(n_neighbors = 10)

knn_model.fit(X_train, Y_train)

Y_pred = knn_model.predict(X_test)

acc_knn_model = round(knn_model.score(X_train, Y_train) * 100, 2)

acc_knn_model
tree_model = DecisionTreeClassifier(random_state=123)

tree_model.fit(X_train, Y_train)

Y_pred = tree_model.predict(X_test)

acc_tree_model = round(tree_model.score(X_train, Y_train) * 100, 2)

acc_tree_model
forest_model = RandomForestClassifier(n_estimators=300,random_state=123)

forest_model.fit(X_train, Y_train)

Y_pred = forest_model.predict(X_test)

forest_model.score(X_train, Y_train)

acc_forest_model = round(forest_model.score(X_train, Y_train) * 100, 2)

acc_forest_model
models = pd.DataFrame({

    'Model': [ 'knn_model', 'Logistic Regression', 'Random Forest','Decision Tree'],

    'Score': [ acc_knn_model, acc_log_model,  acc_forest_model, acc_tree_model]})

models.sort_values(by='Score', ascending=False)

finalData = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })



finalData.to_csv('submission.csv', index=False)