import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
% matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#first focusing on what the train dataset tells us
#1. The first five observations
train.head(5)
#2. What is the size of the dataframe?
train.shape
#3. What are the data types?
train.info()
#4. How many missing values?
# Looks like we can't fully rely on Age and Cabin. We need to figure out a way on how to predict the values of these fields especially Age.
# Since Embarked only has 2 missing values we fill them in using mean, median, or mode
train.isnull().sum()
#5. Basic Statistics
# 39% of the data survived
# average fare is $32
# the younges passenger is around five months old and the oldest is 80 years old
train.describe()
#6. Number of unique Values
train.nunique()
#Tickets have dupes
#Cabin have dupes
train['Survived_C'] = np.where(train['Survived']==0,'No','Yes')
train.head()
#Breakdown of the passengers by gender
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,4))

train['Sex'].value_counts().plot.pie(autopct='%1.f%%', shadow=True, explode=(.1,0), startangle=90, ax=ax[0]).axis('equal')
ax[0].set_title('Passenger Breakdown Based on Sex')

ax1 = sns.countplot(x='Survived_C', hue='Sex', data=train, ax=ax[1])
total = float(len(train))
for p in ax1.patches:
    height = p.get_height()+1
    ax1.text(p.get_x()+p.get_width()/2.,
            height+3,
            '{:1.0f}%'.format((height/total)*100),
            ha="center")
ax1.set_title('Survival Count and Rate Breakdown')

plt.show()
sns.barplot(x='Sex', y='Survived', data=train).set_title('Survival Rate based on Sex')
plt.show()
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Filling in missing Embarked values
train[train.Embarked.isnull()]
#both are female Pclass1 survivors, paid $80, and travelled alone (although PassengerID 830 looks like have 2 Names registered?)
# Determining how many passengers per observation / per ticket
train['No_of_Passengers_on_Ticket']= train['SibSp'] + train['Parch'] + 1 #+1 for those travelling alone

# Adding a column called 'Group Size' to better segment the observation
# Solo - 1 traveller
# Couple - 2 travellers 
# Mid - 3 to 5 travellers
# Large - 6+ travellers 
train['Group_Size'] = np.where(train.No_of_Passengers_on_Ticket==1, 'Solo',
                                    np.where(train.No_of_Passengers_on_Ticket==2, 'Couple', 
                                             np.where(np.logical_and(train.No_of_Passengers_on_Ticket>2, train.No_of_Passengers_on_Ticket<6),'Mid',
                                                      'Large')))
train[train.Embarked.isnull()]
#We know that the missing values are Female passengers, Pclass 1, fare=$80, both travelling alone
Pclass1 = train[(train['Sex']=='female') & (train['Pclass']==1) & (train['Group_Size']=='Solo')]
#Pclass1[['Embarked', 'Fare']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Fare', ascending=False)
Pclass1[['Embarked', 'Fare']].groupby(['Embarked'], as_index=False).describe()
#The mean closest to $80 is Southampton. It also has the smaller standard deviation 
Pclass1[['Embarked', 'Fare']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Fare', ascending=False)
train.Embarked.fillna('S', inplace=True) #filling in missing Embarked with S
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,4))

train['Embarked'].value_counts().plot.pie(autopct='%1.f%%', shadow=True, explode=(.1,0.1,0), startangle=90, ax=ax[0]).axis('equal')
ax[0].set_title('Passenger Breakdown Based on Port of Embarkation')

sns.countplot(x='Survived_C', hue='Embarked', data=train, ax=ax[1]).set_title('Survival Count based on Port of Embarkation')

plt.show()
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,4))

colors=['forestgreen','steelblue', 'darkorange']

train['Pclass'].value_counts().plot.pie(autopct='%1.1f%%', shadow=True, explode = (0.1,0,0), startangle=90, colors=colors, ax=ax[0]).axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
ax[0].set_title('Proportion of Passengers Per Pclass')

sns.barplot(x='Pclass', y='Survived', ax=ax[1], data=train).set_title('Survival Rate based on Pclass')

plt.show()
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train).set_title('Survival Rate Per Gender and Pclass')
plt.show()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
sns.countplot(x='Embarked', hue='Pclass', data=train).set_title('Survival Count based on Port of Embarkation and Pclass')
plt.show()
data = train.sort_values(['No_of_Passengers_on_Ticket'], ascending=True)

f, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,4))

sns.countplot(x='Group_Size', hue='Survived_C', ax=ax[0], data=data).set_title('Survival Count based on Group Size')
sns.countplot(x='No_of_Passengers_on_Ticket', hue='Survived_C', ax=ax[1], data=data).set_title('Survival Count based on Group Size')
plt.show()
train['Age'].fillna(train.Age.mean(), inplace=True)
train.describe()
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,4))

Gender = ['male', 'female']
Survive = ['Yes', 'No']

for g in Gender:
    survivors = train[(train['Sex']==g) & (train['Survived_C']=='Yes')].Age
    sns.distplot(survivors, hist=False, label=g, ax=ax[0]).set_title('Age Distribution of Survivors')
    deaths = train[(train['Sex']==g) & (train['Survived_C']=='No')].Age        
    sns.distplot(deaths, hist=False, label=g, ax=ax[1]).set_title('Age Distribution of those Who Died')  

plt.show()
f, ax = plt.subplots(nrows=1,ncols=2,figsize=(15,4))

sns.pointplot(y='Age', x='No_of_Passengers_on_Ticket', hue='Survived_C', linestyles=['--','-'], markers=['x','o'], \
              dodge=True, data=train, ax=ax[0]).set_title('Distribution by Age and Passenger Count')
sns.pointplot(y='Age', x='Group_Size', hue='Survived_C', linestyles=['--','-'], markers=['x','o'], \
              dodge=True, data=train, ax=ax[1]).set_title('Distribution by Age and Group Size')

plt.show()
sns.boxplot(y='Age',x='Pclass', data=train).set_title('Age and Pclass')
#Dropping 
train.drop(['Survived_C','Cabin'], axis=1, inplace=True)
train.head()
Sex = pd.get_dummies(train['Sex'], drop_first=True)
Embarked_New = pd.get_dummies(train['Embarked'], drop_first=True)
Pclass_New = pd.get_dummies(train['Pclass'], drop_first=True)
train_n = pd.concat([train,Sex,Embarked_New,Pclass_New],axis=1)
train_n.drop(columns=['Sex','Pclass','Embarked','Name','PassengerId','Ticket','Group_Size'],axis=1, inplace=True)
train_n.rename(columns={'male':'Sex', 'Q':'Queenstown', 'S':'Southampton',2:'Pclass2',3:'Pclass3'}, inplace=True)
train_n.head()
# Preparing the data
X = train_n.drop(['Survived'], axis=1)
y = train_n.Survived
#building the logistic regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)
logmodel = LogisticRegression()
#fitting the model
logmodel.fit(X_train, y_train)
#make predictions
prediction = logmodel.predict(X_test)
# Examining the Prediction
#calculate precision and recall
from sklearn.metrics import classification_report
classification_report(y_test,prediction)
#look at the confusion matrix to justify Precision and Recall
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,prediction)
#calculate the accuracy score - from the confusion matrix
#Number of correct predictions
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)
#we have 77% accuracy, which is still pretty good
#Preparing the data
Sex = pd.get_dummies(test['Sex'], drop_first=True)
Emb = pd.get_dummies(test['Embarked'], drop_first=True)
Pcl = pd.get_dummies(test['Pclass'], drop_first=True)
test_n = pd.concat([test, Sex, Emb, Pcl], axis=1)

#Matching the train data's column labels
test_n.drop(columns=['PassengerId', 'Ticket', 'Pclass', 'Name', 'Sex', 'Cabin', 'Embarked'], axis=1, inplace=True)
test_n['No_of_Passengers_on_Ticket'] = test_n.SibSp + test_n.Parch + 1
test_n.rename(columns={'male':'Sex', 'Q':'Queenstown', 'S':'Southampton',2:'Pclass2',3:'Pclass3'}, inplace=True)
test_n.head()

#checking if the test data has null values
#test_n.isnull().sum()

#Dropping the null values of the test data
X_n_test = test_n.dropna(how='any')
test_n.shape
X_n_test.shape
#X_n_test is still a decent sample size, we still have 79% of the actual test data
331/418
# Preparing the data
X_train1 = train_n.drop(['Survived'], axis=1)
y_train1 = train_n.Survived

#building the logistic regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Train data has a 891 observations, while Testa data has 331 observations
#Using 331/891 = 37% of the Train data as the test size, so that I can compare and plug the Test data later on
X_train, X_test, y_train, y_test = train_test_split(X_train1, y_train1, test_size=.3714, random_state=1)
logmodel = LogisticRegression()

#fitting the model
logmodel.fit(X_train, y_train)
#predict from the train set
prediction = logmodel.predict(X_test)
#Examining the Prediction
#calculate precision and recall
from sklearn.metrics import classification_report
classification_report(y_test,prediction)
#calculate the accuracy score - from the confusion matrix
#Number of correct predictions
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)
#we have 77% accuracy, which is still pretty decent
#predict from the test set
prediction2 = logmodel.predict(X_n_test)
# Examining the Prediction
#calculate precision and recall
from sklearn.metrics import classification_report
classification_report(y_test,prediction2)
#look at the confusion matrix to justify Precision and Recall
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,prediction2)
#calculate the accuracy score - from the confusion matrix
#Number of correct predictions
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction2)
#we have 50% accuracy, it's not good... need to improve the prediction model
X_train2 = train_n.drop(['Survived'], axis=1)
y_train2 = train_n.Survived

X_train, X_test, y_train, y_test = train_test_split(X_train2, y_train2, test_size=.3714, random_state=2)

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#check classification accuracy of KNN with K=16 
#the bigger the K, the less likely an error will occur, 10 is a good start to fine tuning K
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
metrics.accuracy_score(y_test,y_pred)
#from 10 to 33, K=17 has the highest accuracy
#70% classification of accuracy of KNN = 17 using the Train set
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
scores
#average accuracy as an estimate of out-of-sample accuracy - using the Train set
#71% accuracy
scores.mean()
#using the Test set 
y_pred2 = knn.predict(X_n_test)
metrics.accuracy_score(y_test,y_pred2)
#56% classification accuracy of KNN = 17, using the Test set
scores2 = cross_val_score(knn, X_n_test, y_test, cv=10, scoring='accuracy')
scores2
#average accuracy as an estimate of out-of-sample accuracy - using the Train set
#51% accuracy
scores2.mean()
#Looks similar to Logistic Regression 