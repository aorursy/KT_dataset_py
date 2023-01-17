# To check data directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#import basic libraries

import numpy as np                 # For Linear Algebra
import pandas as pd                # For data manipulation

import matplotlib.pyplot as plt    # For data visualization
import seaborn as sns              # For data visualization

%matplotlib inline

import warnings                   #ignore warnings
warnings.filterwarnings('ignore')
# Read train and test dataset

train = pd.read_csv("../input/titanic/train.csv")
test  = pd.read_csv("../input/titanic/test.csv")
#make copy of train and test datset

train_df = train.copy()
test_df  = test.copy()

#Combine both datasets for running certain operations together

combine = [train_df, test_df]
#View sample of top 5 records in train dataset

train_df.head()
#Display sample of test dataset

test_df.head()
#display the shape of train and test datset

train_df.shape,test_df.shape
# Satistical summary of numerical variable in the train datset.

train_df.describe()
#Statistical summary of categorical variables in the train dataset.

train_df.describe(include=['object'])
train_df.info()
train_df['Survived'].value_counts().plot.bar()
train_df['Pclass'].value_counts().plot.bar()
train_df['SibSp'].value_counts().plot.bar()
train_df['Parch'].value_counts().plot.bar()
train_df['Sex'].value_counts().plot.bar()
train_df['Embarked'].value_counts(normalize=True).plot.bar()
sns.distplot(train_df['Fare'],color="m", ) 


plt.subplot(121) 
sns.distplot(train_df['Age'],color="m", ) 

plt.subplot(122) 
train_df['Age'].plot.box(figsize=(16,5)) 

plt.show()
#draw a bar plot of survival by sex

sns.barplot(x='Sex',y="Survived",data=train_df)

#print percentages of females vs. males that survive

print("Percentage of females who survived:", train_df["Survived"][train_df["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", train_df["Survived"][train_df["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
#draw a bar plot of survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=train_df)

#print percentage of people by Pclass that survived
print("Percentage of Pclass = 1 who survived:", train_df["Survived"][train_df["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", train_df["Survived"][train_df["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", train_df["Survived"][train_df["Pclass"] == 3].value_counts(normalize = True)[1]*100)
#draw a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=train_df)

# printing individual percent values for all of these.
print("Percentage of SibSp = 0 who survived:", train_df["Survived"][train_df["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:", train_df["Survived"][train_df["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", train_df["Survived"][train_df["SibSp"] == 2].value_counts(normalize = True)[1]*100)
#draw a bar plot for Parch vs. survival
sns.barplot(x="Parch", y="Survived", data=train_df)
plt.show()
#sort the ages into logical categories

train_df["Age"] = train_df["Age"].fillna(-0.5)
test_df["Age"] = test_df["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train_df['AgeGroup'] = pd.cut(train_df["Age"], bins, labels = labels)
test_df['AgeGroup'] = pd.cut(test_df["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train_df)
plt.show()
train_df.head()
pd.isnull(train_df).sum()
pd.isnull(test_df).sum()
train_1 = train_df.drop(['Cabin'], axis = 1)
test_1 = test_df.drop(['Cabin'], axis = 1)
train_1.shape,test_1.shape
train_2 = train_1.drop(['Ticket'], axis = 1)
test_2 = test_1.drop(['Ticket'], axis = 1)
train_2.shape,test_2.shape
#now we need to fill in the missing values in the Embarked feature

print("Number of people embarking in Southampton (S):")
southampton = train_2[train_2["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg = train_2[train_2["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown = train_2[train_2["Embarked"] == "Q"].shape[0]
print(queenstown)
#replacing the missing values in the Embarked feature with S
train_2 = train_2.fillna({"Embarked": "S"})
#create a combined group of both datasets
combine = [train_2, test_2]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_2['Title'], train_2['Sex'])
#replace various titles with more common names

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_2[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#map each of the title groups to a numerical value

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_2.head()
# fill missing age with mode age group for each title
mr_age = train_2[train_2["Title"] == 1]["AgeGroup"].mode() #Young Adult
miss_age = train_2[train_2["Title"] == 2]["AgeGroup"].mode() #Student
mrs_age = train_2[train_2["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = train_2[train_2["Title"] == 4]["AgeGroup"].mode() #Baby
royal_age = train_2[train_2["Title"] == 5]["AgeGroup"].mode() #Adult
rare_age = train_2[train_2["Title"] == 6]["AgeGroup"].mode() #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

#I tried to get this code to work with using .map(), but couldn't.
#I've put down a less elegant, temporary solution for now.
#train = train.fillna({"Age": train["Title"].map(age_title_mapping)})
#test = test.fillna({"Age": test["Title"].map(age_title_mapping)})

for x in range(len(train_2["AgeGroup"])):
    if train_2["AgeGroup"][x] == "Unknown":
        train_2["AgeGroup"][x] = age_title_mapping[train_2["Title"][x]]
        
for x in range(len(test_2["AgeGroup"])):
    if test_2["AgeGroup"][x] == "Unknown":
        test_2["AgeGroup"][x] = age_title_mapping[test_2["Title"][x]]
#map each Age value to a numerical value

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train_2['AgeGroup'] = train_2['AgeGroup'].map(age_mapping)
test_2['AgeGroup'] = test_2['AgeGroup'].map(age_mapping)

train_2.head()

#dropping the Age feature for now, might change
train_3 = train_2.drop(['Age'], axis = 1)
test_3 = test_2.drop(['Age'], axis = 1)
train_3.head()
test_3.head()
#drop the name feature since it contains no more useful information.
train_4 = train_3.drop(['Name'], axis = 1)
test_4 = test_3.drop(['Name'], axis = 1)
#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train_4['Sex'] = train_4['Sex'].map(sex_mapping)
test_4['Sex'] = test_4['Sex'].map(sex_mapping)

train_4.head()
#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train_4['Embarked'] = train_4['Embarked'].map(embarked_mapping)
test_4['Embarked'] = test_4['Embarked'].map(embarked_mapping)

train_4.head()
#fill in missing Fare value in test set based on mean fare for that Pclass 
for x in range(len(test_4["Fare"])):
    if pd.isnull(test_4["Fare"][x]):
        pclass = test_4["Pclass"][x] #Pclass = 3
        test_4["Fare"][x] = round(train_4[train_4["Pclass"] == pclass]["Fare"].mean(), 4)
        
#map Fare values into groups of numerical values
train_4['FareBand'] = pd.qcut(train_4['Fare'], 4, labels = [1, 2, 3, 4])
test_4['FareBand'] = pd.qcut(test_4['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
train_5 = train_4.drop(['Fare'], axis = 1)
test_5 = test_4.drop(['Fare'], axis = 1)

train_5.head()
test_5.head()
from sklearn.model_selection import train_test_split

predictors = train_5.drop(['Survived', 'PassengerId','AgeGroup'], axis=1)
target = train_5["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
predictors.shape,target.shape
x_train.shape, x_val.shape, y_train.shape, y_val.shape
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
ids = test_5['PassengerId']
predictions = gbk.predict(test_5.drop(['PassengerId','AgeGroup'], axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import cross_val_score


skfold = StratifiedKFold (n_splits=10,shuffle=True, random_state= 1)

Log_skf = LogisticRegression()

Log_skf1 = cross_val_score(Log_skf, x_train, y_train, cv=skfold)

print(Log_skf1)

acc_log2 = Log_skf1.mean()*100.0

acc_log2 
skfold = StratifiedKFold (n_splits=5,shuffle=True, random_state= 1)

svc_sk = SVC(gamma='auto')

svc_sk1 = cross_val_score(svc_sk, x_train, y_train, cv=skfold)

print(svc_sk1)

acc_svc2 = svc_sk1.mean()*100.0

acc_svc2 
skfold = StratifiedKFold (n_splits=10,shuffle=True, random_state= 1)

knn_sk = KNeighborsClassifier(n_neighbors = 3)

knn_sk1 = cross_val_score(knn_sk, x_train, y_train, cv=skfold)

print(knn_sk1)

acc_knn2 = knn_sk1.mean()*100.0

acc_knn2
skfold = StratifiedKFold (n_splits=10,shuffle=False, random_state= 1)

rfc_sk = RandomForestClassifier(n_estimators=100,random_state = 1)

rfc_sk1 = cross_val_score(rfc_sk, x_train, y_train, cv=skfold)

print(rfc_sk1)

acc_rfc2 = rfc_sk1.mean()*100.0

acc_rfc2
skfold = StratifiedKFold (n_splits=5,shuffle=False, random_state= 1)

gnb_sk = GaussianNB()

gnb_sk1 = cross_val_score(gnb_sk,x_train, y_train, cv=skfold)

print(gnb_sk1)

acc_gnb2 = gnb_sk1.mean()*100.0

acc_gnb2
skfold = StratifiedKFold (n_splits=5,shuffle=True, random_state= 1)

ptn_sk = Perceptron()

ptn_sk1 = cross_val_score(ptn_sk, x_train, y_train, cv=skfold)

print(ptn_sk1)

acc_ptn2 = ptn_sk1.mean()*100.0

acc_ptn2
skfold = StratifiedKFold (n_splits=5,shuffle=True, random_state= None)

dt_sk = DecisionTreeClassifier(random_state=1)

dt_sk1 = cross_val_score(dt_sk, x_train, y_train, cv=skfold)

print(dt_sk1)

acc_dt2 = dt_sk1.mean()*100.0

acc_dt2
import lightgbm as lgb

skfold = StratifiedKFold (n_splits=10,shuffle=True, random_state= None)

lgb_sk = lgb.LGBMClassifier()

lgb_sk1 = cross_val_score(lgb_sk, x_train, y_train, cv=skfold)

print(lgb_sk1)

acc_lgb2 = lgb_sk1.mean()*100.0

acc_lgb2
models = pd.DataFrame({
    'Model': ['Support Vector Classifier', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron',  
              'Decision Tree','LGBMClassifier'],
    'Mean Accuracy': [acc_svc2, acc_knn2, acc_log2, 
              acc_rfc2, acc_gnb2, acc_ptn2, 
             acc_dt2,acc_lgb2]})
models.sort_values(by='Mean Accuracy', ascending=False)