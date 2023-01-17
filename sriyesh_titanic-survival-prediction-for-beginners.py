#this is to access data set stored in google drive to use it uncomment the following lines
#!pip install pydrive
#from google.colab import auth
#auth.authenticate_user()

#from pydrive.drive import GoogleDrive
#from pydrive.auth import GoogleAuth
#from oauth2client.client import GoogleCredentials
#gauth = GoogleAuth()
#gauth.credentials = GoogleCredentials.get_application_default()
#drive = GoogleDrive(gauth)

#myfile = drive.CreateFile({'id': 'your_train_dataset_id'})
#myfile2 = drive.CreateFile({'id': 'your_test_dataset_id'})
#myfile.GetContentFile('train.csv')
#myfile2.GetContentFile('test.csv')

# data analysis library
import numpy as np
import pandas as pd

# Visuvalization library
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#ignore warnings
import warnings 
warnings.filterwarnings('ignore')

# import train and test CSV files
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#see the first 3 rows of train
train.head(3)
#Look into the training data
train.describe(include='all')
# get a list of features
print(train.columns)
#see a sample of the dataset to get an idea of variables
train.sample(5)
# See a summary of traning data
train.describe(include = 'all')
# checking of NaN values
print(pd.isnull(train).sum())
#draw a bar plot of survival by sex
sns.barplot(x= 'Sex',y='Survived',data=train)

#print % of female vs. male that survived
print('Percentage of female who survived:',train['Survived'][train['Sex'] == 'female'].value_counts(normalize = True)[1]*100)

print('Percentage of male who survived:',train['Survived'][train['Sex'] == 'male'].value_counts(normalize = True)[1]*100)
#draw a bar plot of survival by Pclass
sns.barplot(x='Pclass', y='Survived',data=train)

# Percentage of people that survived by Pclass
#class 1 = Upper
print('petcentage of Pclass=1 who survived:', train['Survived'][train['Pclass'] == 1].value_counts(normalize=True)[1]*100)

#class 2 = Middle
print('petcentage of Pclass=2 who survived:', train['Survived'][train['Pclass'] == 2].value_counts(normalize=True)[1]*100)

#class 3 = Lower
print('petcentage of Pclass=3 who survived:', train['Survived'][train['Pclass'] == 3].value_counts(normalize=True)[1]*100)
#draw a bar plot for SibSp vs. Survival
sns.barplot(x='SibSp',y='Survived',data=train)

# The percentage of people survived
print("Percentage of SibSp = 0 who suvived:",train['Survived'][train['SibSp']==0].value_counts(normalize=True)[1]*100)

print("Percentage of SibSp = 1 who suvived:",train['Survived'][train['SibSp']==1].value_counts(normalize=True)[1]*100)

print("Percentage of SibSp = 2 who suvived:",train['Survived'][train['SibSp']==2].value_counts(normalize=True)[1]*100)

print("Percentage of SibSp = 3 who suvived:",train['Survived'][train['SibSp']==3].value_counts(normalize=True)[1]*100)

print("Percentage of SibSp = 4 who suvived:",train['Survived'][train['SibSp']==4].value_counts(normalize=True)[1]*100)
#draw a bar plot for parch vs.survival
sns.barplot(x='Parch',y='Survived',data=train)
#sort the ages into logical categories
#train["Age"] = train["Age"].fillna(-0.5)
#test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young adult', 'Adult', 'Elderly']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

# Draw a bar plot for age vs. survival
sns.barplot(x='AgeGroup',y='Survived',data=train)
# to convert the alphanum values to integer for ploting
train['CabinBool'] = train['Cabin'].notnull().astype('int')
test['CabinBool'] = test['Cabin'].notnull().astype('int')

sns.barplot(x='CabinBool', y='Survived', data=train )

# Calculate percentage of CabinBool vs. Survived
print('percentage of CabinBool = 1 who survived:',train['Survived'][train['CabinBool']==1].value_counts(normalize=True)[1]*100)

print('percentage of CabinBool = 0 who survived:',train['Survived'][train['CabinBool']==0].value_counts(normalize=True)[1]*100)

# Draw a bar plot of port of embarkment vs. survival
sns.barplot(x='Embarked', y='Survived', data=train )

# Calculate percentage of CabinBool vs. Survived
print('percentage of Embarked = S who survived:',train['Survived'][train['Embarked']=='S'].value_counts(normalize=True)[1]*100)


print('percentage of Embarked = C who survived:',train['Survived'][train['Embarked']=='C'].value_counts(normalize=True)[1]*100)
      
      
print('percentage of Embarked = Q who survived:',train['Survived'][train['Embarked']=='Q'].value_counts(normalize=True)[1]*100)
test.describe(include='all')
# we can drop the Ticket feature since it's unlikely to yeild any useful information
train.drop(['Ticket'], axis = 1, inplace=True)
test.drop(['Ticket'], axis=1, inplace=True)
 #now we need to fill in the missing value in the embarked feature
print("Number of People embarking in Southampton(S):")
southampton = train[train['Embarked']== 'S'].shape[0]
print(southampton)
  
print("Number of People embarking in Cherbourg(C):")
cherbourg = train[train['Embarked']== 'C'].shape[0]
print(cherbourg)
        
print("Number of People embarking in Queenstown(Q):")
queenstown = train[train['Embarked']== 'Q'].shape[0]
print(queenstown)
#replacing the missing values in the embarked feature with s
train = train.fillna({"Embarked":"S"})
# create a combined group of both dataset
combine = [train, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
  dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
pd.crosstab(train['Title'], train['Sex'])
# Replace various titles with more common names
for dataset in combine:
  dataset['Title'] = dataset['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')
  dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir'],'Royal')
  dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'],'Miss')
  dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()  
  
  
# Map each title to a numerical number
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5, 'Rare':6}
for dataset in combine:
  dataset['Title'] = dataset['Title'].map(title_mapping)
  dataset['Title'] = dataset['Title'].fillna(0)

train.head()
# group by Sex, Pclass, and Title
#combined = train.append(test,ignore_index=True)
grouped = train.groupby(['Sex','Pclass', 'Title'])
grouped2 = test.groupby(['Sex','Pclass', 'Title'])
# view the median Age by the grouped features 
grouped.Age.median()
grouped2.Age.median()
# apply the grouped median value on the Age NaN
train.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))
test.Age = grouped2.Age.apply(lambda x: x.fillna(x.median()))
train.info()
print('-'*40)
test.info()
#drop the name feature
train.drop(['Name'], axis = 1, inplace=True)
test.drop(['Name'],axis = 1, inplace=True)
# map each sex value to anumerical value
sex_mapping = {"male":0,"female":1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()
train.drop('AgeGroup',axis=1,inplace=True)
test.drop('AgeGroup',axis=1,inplace=True)
train.info()
test.info()
# Map each Embarked value to a numerical value
embarked_mapping = {'S':1,'C':2,'Q':3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

#Drop the cabin Feature
train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)
#fill in the missing Fare value in Test set based on mean fare for that Pclass
for x in range(len(test['Fare'])):
  if pd.isnull(test['Fare'][x]):
    pclass = test['Pclass'][x]  #Pclass = 3
    test['Fare'][x] = round(train[train['Pclass'] == pclass]['Fare'].mean(), 4)
    
# maping Fare into Group of numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
train.drop(['Fare'],axis=1,inplace=True)
test.drop(['Fare'],axis=1,inplace=True)
train.head()
test.head()
from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train['Survived']
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
#SVM model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)
#set ids as PassengeId and predict survival
ids = test['PassengerId']
predictions = svc.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({'PassengerId':ids, 'Survived':predictions})
output.to_csv('submission.csv', index=False)
#This give a list of folder and ids in your google drive
file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
for file1 in file_list:
  print('title: %s, id: %s' % (file1['title'], file1['id']))
# Copy the id of the folder where you want to save the submission file
file = drive.CreateFile({'parents':[{u'id': 'Your_folder_id_here'}]}) 
file.SetContentFile("submission.csv")
file.Upload()