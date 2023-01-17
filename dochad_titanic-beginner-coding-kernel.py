


#data analysis libraries 
import numpy as np
import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#ignore warnings
import warnings
warnings.filterwarnings('ignore')
#Reading and keep the tables in easy variables
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#Looking at the closer details
train.describe()
#To see the collumn names only
print(train.columns)
#To see it on a graph
train.sample(20)
print(train.dtypes)
print(train.isnull().sum())
sns.barplot(data= train, x="Sex", y="Survived")
print("Percentage of females surviving:", train["Survived"][train.Sex == "female"].value_counts(normalize=True)*100)
print("Number males surviving:", train["Survived"][train.Sex == "male"].value_counts(normalize = True)*100)
train["Age"] = train["Age"].fillna(-0.5)
test["Age"]= test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60,np.inf]
labels = ["Unknown","Small children","Children","Teens","Adolescent","Young Adults","Adults","Elders"]
train["AgeGroups"] = pd.cut(train.Age,labels=labels, bins=bins)
test["AgeGroups"] = pd.cut(test.Age,labels=labels, bins=bins)
sns.barplot(data = train,x= "AgeGroups",y= "Survived")

#pclass Factor
sns.barplot(data= train, x="Pclass", y="Survived")
print("Percentage of High earners surviving:", train["Survived"][train.Pclass == 1].value_counts(normalize=True)*100)
print("Percentage of middle class earners surviving:", train["Survived"][train.Pclass == 2].value_counts(normalize = True)*100)
print("Percentage of low class earners surviving:", train["Survived"][train.Pclass == 3].value_counts(normalize = True)*100)
#Family factors
sns.barplot(data =train, x="SibSp",y="Survived")
print("Survivors with one sibling/Spouse:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)
print("Survivors with two sibling/Spouse:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)
print("survivors with three sibling/Spouse:", train["Survived"][train["SibSp"] == 3].value_counts(normalize = True)[1]*100)
print("Survivors with four sibling/Spouse:", train["Survived"][train["SibSp"] == 4].value_counts(normalize = True)[1]*100)

#Parch Factor
sns.barplot(data = train,x="Parch",y="Survived")
train["CabinBool"] = (train.Cabin.notnull().astype("int"))
test["CabinBool"] =  (train.Cabin.notnull().astype("int"))
print("Percentage of recorded Cabins that survived:",train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of recorded Cabins that didn't survive:",train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)
sns.barplot(data = train,x="CabinBool",y="Survived")
test.describe(include="all")
train = train.drop(["Ticket"],axis = 1)
test = test.drop(['Ticket'], axis = 1)

train = train.drop(["Cabin"],axis = 1)
test = test.drop(["Cabin"],axis = 1)

train = train.drop(["CabinBool"],axis = 1)
test = test.drop(["CabinBool"],axis = 1)
print("Number of people embarking in Southampton (S):",train[train["Embarked"] == "S"].shape[0])

print("Number of people embarking in Cherbourg (C):",train[train["Embarked"] == "C"].shape[0])

print("Number of people embarking in Queenstown (Q):",train[train["Embarked"] == "Q"].shape[0])
#replacing the missing values in the Embarked feature with S
train = train.fillna({"Embarked": "S"})
#create a combined group of both datasets
combine = [train, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])
#Replace the titles with numbers that would let out computer understand
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_map = {"Master":1,"Miss":2,"Mr":3,"Mrs":4,"Rare":5,"Royal":5}
for dataset in combine:
    dataset["Title"] = dataset["Title"].map(title_map)
    dataset["Title"] = dataset["Title"].fillna(0)

train.head()
Master_age = train[train["Title"]==1]["AgeGroups"].mode()#Small children (Somehow, don't ask me)
Ms_age = train[train["Title"]==2]["AgeGroups"].mode() #Teens
Mr_age = train[train["Title"]==3]["AgeGroups"].mode() #Young Adults
Mrs_age = train[train["Title"]==4]["AgeGroups"].mode() #Adults
Rare_age = train[train["Title"]==5]["AgeGroups"].mode() #Adult
Royal_age = train[train["Title"]==6]["AgeGroups"].mode() #Adults
#Demonstrating what I'm doing when using .mode
print(Master_age)
print(Ms_age)
print(Mrs_age)
train.head()
title_age_map = {1: "Small children", 2: "Teens", 3: "Young Adults", 4: "Adults", 5: "Adults", 6: "Adults"}

for x in range(len(train["AgeGroups"])):
    if train["AgeGroups"][x] == "Unknown":
        train["AgeGroups"][x] = title_age_map[train["Title"][x]]

for x in range(len(test["AgeGroups"])):
    if test["AgeGroups"][x] == "Unknown":
        test["AgeGroups"][x] = title_age_map[train["Title"][x]]
Age_map = {"Small children": 1, "Children": 2, "Teens": 3, "Adolescent": 4, "Young Adults": 5, "Adults": 6, "Elders": 7}

train["AgeGroups"] = train["AgeGroups"].map(Age_map)
test["AgeGroups"] = test["AgeGroups"].map(Age_map)
train.head()
#drop the name feature since it contains no more useful information.
train = train.drop(["Name"], axis = 1)
test = test.drop(["Name"], axis = 1)
train = train.drop(["Age"], axis = 1)
test = test.drop(["Age"], axis = 1)
#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()
#map each Embarked value to a numerical value that can be read by the computer. Almost there!
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()
print(train.Fare.max())
print(train.Fare.min())
#fill in missing Fare value in test set based on mean fare for that Pclass 
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)
        
#map Fare values into groups of numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)
#Check train data to see what we ended up with
train.head()
#Check test data
test.head()
from sklearn.model_selection import train_test_split

predict = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predict, target, test_size = 0.22, random_state = 0)
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)
# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)
# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)
# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)
# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)
# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)
# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)
# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)
#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = gbk.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)