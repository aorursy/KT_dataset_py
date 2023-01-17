import os
print(os.listdir("../input"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# This is how we assign the datasets to variables in python using pandas.
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.head()
test.head()
#To get the number of rows and columns of the dataset
train.shape
test.shape
#Gives us statistical information about the dataset
train.describe()
train.isnull().sum()
test.isnull().sum()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style("whitegrid")
sns.countplot(x='Survived',data=train,palette='viridis')
train.Pclass.value_counts()
train.groupby('Pclass').Survived.value_counts()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
train.Sex.value_counts()
train.groupby('Survived').Sex.value_counts()
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
tab = pd.crosstab(train['Pclass'], train['Sex'])
print (tab)
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
#We are doing this because the test doesn't have the Target column.
train2=train.drop('Survived',axis=1)
#We are combining train and test dataset as it will be easier for us to process the data together.
data = train2.append(test,sort=False)
data.head()
#We drop the PassengerId column as the values from this column wont contribute to our model.
data.drop(['PassengerId'],axis=1,inplace=True)
data['Title'] =data['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
data['Name_Len'] = data['Name'].apply(lambda x: len(x))
data.drop(labels='Name', axis=1, inplace=True)
data.Name_Len = (data.Name_Len/10).astype(np.int64)+1
training_age_n = data.Age.dropna(axis=0)
fx, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].set_title("Age vs frequency")
axes[1].set_title("Age vise Survival rate")
fig1_age = sns.distplot(a=training_age_n, bins=15, ax=axes[0], hist_kws={'rwidth':0.7})

# Creating a new list of survived and dead

pass_survived_age = train[train.Survived == 1].Age
pass_dead_age = train[train.Survived == 0].Age

axes[1].hist([data.Age, pass_survived_age, pass_dead_age], bins=5, range=(0, 100), label=['Total', 'Survived', 'Dead'])
axes[1].legend()
plt.show()
#Null Ages in Training set (177 null values)
train_age_mean = data.Age.mean()
train_age_std = data.Age.std()
train_age_null = data.Age.isnull().sum()
rand_tr_age = np.random.randint(train_age_mean - train_age_std, train_age_mean + train_age_std, size=train_age_null)
data['Age'][np.isnan(data['Age'])] = rand_tr_age
data['Age'] = data['Age'].astype(int) + 1

# Null Ages in Test set (86 null values)
test_age_mean = data.Age.mean()
test_age_std = data.Age.std()
test_age_null = data.Age.isnull().sum()
rand_ts_age = np.random.randint(test_age_mean - test_age_std, test_age_mean + test_age_std, size=test_age_null)
data['Age'][np.isnan(data['Age'])] = rand_ts_age
data['Age'] = data['Age'].astype(int)

data.Age = (data.Age/15).astype(np.int64) + 1
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['isAlone'] =data['FamilySize'].map(lambda x: 1 if x == 1 else 0)
data.drop(labels=['SibSp', 'Parch'], axis=1, inplace=True)
data.head()
# We drop the Cabin column as it has too many null values.
data.drop(['Cabin'],axis=1,inplace=True)
data['Ticket_Len'] = data['Ticket'].apply(lambda x: len(x))
data.drop(labels='Ticket', axis=1, inplace=True)
data['Fare'][np.isnan(data['Fare'])] = data.Fare.mean()
data.Fare = (data.Fare /20).astype(np.int64) + 1
data['Embarked'].isnull().sum()
data['Embarked'] = data['Embarked'].fillna('S')
data.head()
from sklearn.preprocessing import LabelEncoder
lr=LabelEncoder()
data['Sex'] = lr.fit_transform(data['Sex'])
data['Embarked']=lr.fit_transform(data['Embarked'])
data['Title']=lr.fit_transform(data['Title'])
train.shape
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
data.head()
#Splitting the data back
train2=data.iloc[0:891,:]
test2=data.iloc[891:1310,:]
train2.shape
#Splitting the dataset back into the train and test.
X=train2
y=train['Survived']
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X,y)
predictions_log = logmodel.predict(test2)
from sklearn.ensemble import RandomForestClassifier
# n_estimators refers to the number of tress.
rfc=RandomForestClassifier(n_estimators=250)
rfc.fit(X,y)
predictions_rfc=rfc.predict(test2)
predictions_log.shape
#my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions_log})
#my_submission.to_csv('submission.csv', index=False)
my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions_rfc})
my_submission.to_csv('submission.csv', index=False)