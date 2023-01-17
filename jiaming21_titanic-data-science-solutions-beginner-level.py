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
#import train and test CSV files

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
#take a look at the training data

train.describe(include="all")
#show a sample of the training data

train.head()
#get a list of the features within the training dataset

print(train.columns)
#take a look at the test data

test.describe(include="all")
#show a sample of the test data

test.head()
#get a list of the features within the test dataset

print(test.columns)
#count missing data in train.csv

print(pd.isnull(train).sum())
#count missing data in test.csv

print(pd.isnull(test).sum())
#visualize the survival states

f,ax=plt.subplots(1,2,figsize=(13,5))

train['Survived'].value_counts().plot.pie(explode=[0,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=train,ax=ax[1])

ax[1].set_title('Survived')

plt.show()
#draw a bar plot of survival by Pclass

sns.barplot(x="Pclass", y="Survived", data=train)

plt.show()



#print percentage of people by Pclass that survived

#print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

#print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

#print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
#draw a bar plot of survival by sex

sns.barplot(x="Sex", y="Survived", data=train)

plt.show()



#print percentages of females vs. males that survive

#print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

#print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
#sort the ages into logical categories

'''

train["Age"] = train["Age"].fillna(-0.5)

test["Age"] = test["Age"].fillna(-0.5)

bins = [-1, 0, 5, 18, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=train)

plt.show()

'''

#sort the ages into logical categories

#train["Age"] = train["Age"].fillna(-0.5)

#test["Age"] = test["Age"].fillna(-0.5)

bins = [0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=train)

plt.show()
train = train.drop(['AgeGroup'], axis = 1)

test = test.drop(['AgeGroup'], axis = 1)
#draw a bar plot for SibSp vs. survival

sns.barplot(x="SibSp", y="Survived", data=train)

plt.show()



#I won't be printing individual percent values for all of these.

#print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)

#print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)

#print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)
#draw a bar plot for Parch vs. survival

sns.barplot(x="Parch", y="Survived", data=train)

plt.show()
fare_not_survived = train['Fare'][train['Survived'] == 0]

fare_survived = train['Fare'][train['Survived'] == 1]



average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])

average_fare.plot(yerr=std_fare, kind='bar', legend=False)



plt.show()
train["has_Cabin"] = (train["Cabin"].notnull().astype('int'))

test["has_Cabin"] = (test["Cabin"].notnull().astype('int'))



#calculate percentages of CabinBool vs. survived

#print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)



#print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)

#draw a bar plot of CabinBool vs. survival

sns.barplot(x="has_Cabin", y="Survived", data=train)

plt.show()
sns.barplot(x="Embarked", y="Survived", data=train)

#sns.barplot('Embarked', 'Survived', data=train, size=3, aspect=2)

#sns.factorplot('Embarked', 'Survived', data=train, size=3, aspect=2)

#plt.title('Embarked and Survived rate')

plt.show()
#create a combined group of both datasets

combine = [train, test]



#extract a title for each Name in the train and test datasets

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])
#replace various titles with more common names

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',

    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#map title

for dataset in combine:

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)
#to show the new column "Title"

train.head()
#drop Name feature

train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
#map each Sex value to a numerical value

sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)



train.head()
#calculate the correlation between features

train_corr = train.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

train_corr[train_corr['Feature 1'] == 'Age']
#take the median value for Age feature based on 'Pclass' and 'Title'

train['Age'] = train.groupby(['Pclass', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))

test['Age'] = test.groupby(['Pclass', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))
#we can also drop the Ticket feature since it's unlikely to yield any useful information

train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)
#fill in missing Fare value in test set based on mean fare for that Pclass 

'''

for x in range(len(test["Fare"])):

    if pd.isnull(test["Fare"][x]):

        pclass = test["Pclass"][x] #Pclass = 3

        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)

'''     

#map Fare values into groups of numerical values

#train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])

#test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

for dataset in combine:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    

for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



#drop Fare values

train = train.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)
#drop the Cabin feature

train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)
#replacing the missing values in the Embarked feature with S

train = train.fillna({"Embarked": "S"})
#map each Embarked value to a numerical value

embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)



train.head()
#count missing data in train.csv

print(pd.isnull(train).sum())
target = train["Survived"]

predictors = train.drop(['Survived', 'PassengerId'], axis = 1)
from sklearn.model_selection import train_test_split



x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.20, random_state = 0)
predictors.head()
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
# XGBoost

from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_val)

acc_xgb = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_xgb)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier', 'XGBoost'],

    'Score': [acc_svc, acc_knn, acc_logreg, 

              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,

              acc_sgd, acc_gbk, acc_xgb]})

models.sort_values(by='Score', ascending=False)
from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from xgboost import XGBClassifier



X = train.drop(['Survived', 'PassengerId'], axis=1)

Y = train["Survived"]



kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

xyz=[]

accuracy=[]

std=[]

classifiers=['Support Vector Machines', 'K-Nearst Neighbor', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier', 'XGBoost']

models=[SVC(), KNeighborsClassifier(), LogisticRegression(), RandomForestClassifier(), GaussianNB(), 

        Perceptron(), LinearSVC(), DecisionTreeClassifier(), SGDClassifier(), 

        GradientBoostingClassifier(), XGBClassifier()]

for i in models:

    model = i

    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy")

    xyz.append(cv_result.mean())

    std.append(cv_result.std())

    accuracy.append(cv_result)

new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       

new_models_dataframe2
#check missing data

print(pd.isnull(test).sum())
train.head()
test.head()
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = xgb.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)

print("The submission was successfully saved!")