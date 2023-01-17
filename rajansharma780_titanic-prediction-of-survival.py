%matplotlib inline
import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
#Load the trainig data csv file and make the data frame out of it

train_df = pd.read_csv('../input/train.csv')

#Load the test data csv file and make the data frame out of it

test_df = pd.read_csv('../input/test.csv')
#display the first five rows of train_df

train_df.head()
print("The train data has {} rows and {} columns".format(train_df.shape[0],train_df.shape[1]))
#display the first five rows of test_df

test_df.head()
print("The test data has {} rows and {} columns".format(test_df.shape[0],test_df.shape[1]))
#check column wise null and missing values

train_df.apply(lambda x: sum(x.isnull()))
#display information of train dataframe

train_df.info()
#display 5 number summary of train dataframe

train_df.describe()
print("Percent of missing Age records is {}%".format((177/891)*100))
#display distribution of age column

ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)

train_df["Age"].plot(kind='density', color='teal')

ax.set(xlabel='Age')

plt.xlim(-10,85)

plt.show()
print("Percent of missing Cabin records is {}%".format((687/891)*100))
print("Percent of missing Embarked records is {}%".format((2/891)*100))
#from data set we know C==Cherbourg,Q=Queens,S=Southampton

print("Boarded passengers are grouped by port of Embark(C==Cherbourg,Q=Queens,S=Southampton)")

print(train_df['Embarked'].value_counts())
sns.countplot(train_df['Embarked'])
train_data = train_df.copy()
train_data['Age'].fillna(train_df['Age'].median(),inplace=True)

train_data['Embarked'].fillna(train_df['Embarked'].value_counts().idxmax(),inplace=True)

train_data.drop('Cabin',axis=1,inplace=True)
#check now null values are there in new dataframe

train_data.apply(lambda x: sum(x.isnull()))
#comaprison of Age distibution before and after adjustment

plt.figure(figsize=(15,10))

ax = train_df['Age'].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)

train_df['Age'].plot(kind='density', color='teal')

ax = train_data['Age'].hist(bins=15, density=True, stacked=True, color='orange', alpha=0.5)

train_data['Age'].plot(kind='density', color='orange')

ax.legend(['Age Before Adjustment','Age After Adjustment'])

ax.set(xlabel='Age')

plt.xlim(-10,85)

plt.show()
train_data['TravelAlone'] = np.where((train_data['SibSp']+train_data['Parch'])>0, 0, 1)

train_data.drop('SibSp',axis=1,inplace=True)

train_data.drop('Parch',axis=1,inplace=True)
training = pd.get_dummies(train_data,columns=["Pclass","Embarked","Sex"])
training.drop('Sex_female', axis=1, inplace=True)

training.drop('PassengerId', axis=1, inplace=True)

training.drop('Name', axis=1, inplace=True)

training.drop('Ticket', axis=1, inplace=True)

final_train = training

final_train.head()
test_df.apply(lambda x : sum(x.isnull()))
#display the 5 number summary of test dataframe

test_df.describe()
test_data = test_df.copy()

test_data['Age'].fillna(train_df['Age'].median(),inplace=True)

test_data['Fare'].fillna(train_df['Fare'].median(),inplace=True)

test_data.drop('Cabin',axis=1,inplace=True)
#now check null values are there in test dataframe

test_data.apply(lambda x : sum(x.isnull()))
test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)

test_data.drop('SibSp', axis=1, inplace=True)

test_data.drop('Parch', axis=1, inplace=True)
testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])

testing.drop('Sex_female', axis=1, inplace=True)

testing.drop('PassengerId', axis=1, inplace=True)

testing.drop('Name', axis=1, inplace=True)

testing.drop('Ticket', axis=1, inplace=True)
final_test = testing

final_test.head()
plt.figure(figsize=(15,10))

ax = sns.kdeplot(final_train['Age'][final_train['Survived']==1],color='Teal',shade=True)

sns.kdeplot(final_train['Age'][final_train['Survived']==0],color='lightcoral',shade=True)

plt.legend(['Survived','Died'])

plt.title("Density plot of Age for surviving people and died people")

ax.set(xlabel='Age')

plt.xlim(-10,85)
final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)



final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)
plt.figure(figsize=(15,10))

avg_survival_by_age = final_train[['Age','Survived']].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age',y='Survived',data=avg_survival_by_age)
plt.figure(figsize=(15,10))

ax = sns.kdeplot(final_train['Fare'][final_train['Survived']==1],color='Teal',shade=True)

sns.kdeplot(final_train['Fare'][final_train['Survived']==0],color='lightcoral',shade=True)

plt.legend(['Survived','Died'])

plt.title("Density plot of Fare for surviving people and died people")

ax.set(xlabel='Fare')

plt.xlim(-20,200)
sns.barplot('Pclass','Survived',data=train_df)
sns.barplot('Embarked','Survived',data=train_df)
sns.barplot('TravelAlone','Survived',data=final_train)
sns.barplot('Sex','Survived',data=train_df)
final_train.columns
plt.figure(figsize=(10,10))

sns.heatmap(final_train.corr(),annot=True)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

X = final_train.drop('Survived',axis=1)

y = final_train['Survived']
# use train/test split with different random_state values

# we can change the random_state values that changes the accuracy scores

# the scores change a lot, this is why testing scores is a high-variance estimate

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
# check classification scores of logistic regression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

y_pred_proba = logreg.predict_proba(X_test)[:, 1]

print('Train/Test split results:')

print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))

print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
final_test.info()
test_pred = logreg.predict(final_test)
test_pred.shape