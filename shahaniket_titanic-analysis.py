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
test_data.describe(include='all')
train_data.describe()
train_data.isnull().sum()
test_data.isnull().sum()
import seaborn as sns
sns.barplot(x="Sex", y="Survived", data=train_data)
sns.barplot(x="Pclass", y="Survived", data=train_data)
sns.barplot(x="SibSp", y="Survived", data=train_data)
sns.barplot(x="Parch", y="Survived", data=train_data)
data = [train_data, test_data]

for dataset in data:
    mean = train_data["Age"].mean()
    std = test_data["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_data["Age"].astype(int)
sns.barplot(x='Embarked', y='Survived', data=train_data)
test_data.isnull().sum()
train_data.isnull().sum()
test_data = test_data.drop(['Cabin'], axis = 1)
train_data = train_data.drop(['Cabin'], axis = 1)

test_data.columns
test_data = test_data.drop(['Ticket'], axis = 1)
train_data = train_data.drop(['Ticket'], axis = 1)
#fill embarked feature with the majority of the data value
train_data['Embarked'] = train_data['Embarked'].fillna('S')
# #create a combined group of both datasets
# combine = [train_data, test_data]

# #extract a title for each Name in the train and test datasets
# for dataset in combine:
#     dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# pd.crosstab(train_data['Title'], train_data['Sex'])
# #replace various titles with more common names
# for dataset in combine:
#     dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
#     'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
#     dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
#     dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
#     dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
#     dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# #map each of the title groups to a numerical value
# title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
# for dataset in combine:
#     dataset['Title'] = dataset['Title'].map(title_mapping)
#     dataset['Title'] = dataset['Title'].fillna(0)

# train_data.head()
# # fill missing age with mode age group for each title
# mr_age = train_data[train_data["Title"] == 1]["AgeGroup"].mode() #Young Adult
# miss_age = train_data[train_data["Title"] == 2]["AgeGroup"].mode() #Student
# mrs_age = train_data[train_data["Title"] == 3]["AgeGroup"].mode() #Adult
# master_age = train_data[train_data["Title"] == 4]["AgeGroup"].mode() #Baby
# royal_age = train_data[train_data["Title"] == 5]["AgeGroup"].mode() #Adult
# rare_age = train_data[train_data["Title"] == 6]["AgeGroup"].mode() #Adult

# age_title_mapping = {1: "YoungA", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}
# pd.options.mode.chained_assignment = None
# train_data['AgeGroup'] = train_data['AgeGroup'].astype('category')
# for x in range(len(train_data["AgeGroup"])):
#     if train_data["AgeGroup"][x] == "Unknown":
# #         group = age_title_mapping[train_data["Title"][x]]
# #         train_data["Title"][x] = group
# #         train_data["AgeGroup"][x] = age_title_mapping[train_data["Title"][x]]
#         train_data["AgeGroup"][x] = age_title_mapping[train_data["Title"][x]]
# # train_data['AgeGroup'] == 'Unknown'
# # for x in range(len(test_data["AgeGroup"])):
# #     if test_data["AgeGroup"][x] == "Unknown":
# #         test_data["AgeGroup"][x] = age_title_mapping[test_data["Title"][x]]
test_data.isnull().sum()
train_data.isnull().sum()
#drop the name feature since it contains no more useful information.
train_data = train_data.drop(['Name'], axis = 1)
test_data = test_data.drop(['Name'], axis = 1)
#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train_data['Sex'] = train_data['Sex'].map(sex_mapping)
test_data['Sex'] = test_data['Sex'].map(sex_mapping)

train_data.head()
#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train_data['Embarked'] = train_data['Embarked'].map(embarked_mapping)
test_data['Embarked'] = test_data['Embarked'].map(embarked_mapping)

train_data.head()
pd.options.mode.chained_assignment = None
for x in range(len(test_data)):
    if pd.isnull(test_data['Fare'][x]):
        pclass = test_data['Pclass'][x]
        mean = round(test_data[test_data['Pclass']==pclass]['Fare'].mean(), 4)
        test_data['Fare'][x] = mean
test_data.isnull().sum()
#map Fare values into groups of numerical values
train_data['FareBand'] = pd.qcut(train_data['Fare'], 4, labels = [1, 2, 3, 4])
test_data['FareBand'] = pd.qcut(test_data['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
train_data = train_data.drop(['Fare'], axis = 1)
test_data = test_data.drop(['Fare'], axis = 1)
from sklearn.model_selection import train_test_split

X = train_data.drop(['PassengerId', 'Survived'],axis=1)
Y = train_data['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20, random_state=0)
#Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
acc_logreg = round(accuracy_score(y_pred, Y_test)*100, 2)

print (acc_logreg)
#K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

acc_knn = round(accuracy_score(y_pred, Y_test)*100,2)
print (acc_knn)
#Support Vector Classification
from sklearn.svm import SVC

classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

acc_svc = round(accuracy_score(y_pred, Y_test)*100,2)

print (acc_svc)
#Naive Bayes Theorem
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

acc_bayes = round(accuracy_score(y_pred, Y_test)*100,2)

print (acc_bayes)
#Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

acc_decision = round(accuracy_score(y_pred, Y_test)*100,2)

print (acc_decision)
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

acc_rfc = round(accuracy_score(y_pred, Y_test)*100,2)

print (acc_rfc)
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(X_train, Y_train)
y_pred = gbk.predict(X_test)
acc_gbk = round(accuracy_score(y_pred, Y_test) * 100, 2)
print(acc_gbk)
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Decision Tree', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, acc_rfc, acc_bayes,acc_decision, acc_gbk]})
models.sort_values(by='Score', ascending=False)
test_data.columns
from sklearn.ensemble import GradientBoostingClassifier
ids = test_data['PassengerId']
test_predict = test_data.drop(['PassengerId'], axis=1)
gbk = GradientBoostingClassifier()
gbk.fit(X, Y)
predictions = gbk.predict(test_predict)
#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic_submission.csv', index=False)