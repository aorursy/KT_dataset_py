import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting data
import seaborn as sns #for plotting data

from sklearn.preprocessing import LabelEncoder #to convert categorical variables into numerical values

#for ignoring warnings
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.describe(include = 'all')
train.head()
train.isnull().sum()
print("Percentage of Pclass = 1 who survived:", round(train["Survived"][train["Pclass"] == 1]
                                                      .value_counts(normalize = True)[1]*100),"%")

print("Percentage of Pclass = 2 who survived:", round(train["Survived"][train["Pclass"] == 2]
                                                      .value_counts(normalize = True)[1]*100),"%")

print("Percentage of Pclass = 3 who survived:", round(train["Survived"][train["Pclass"] == 3]
                                                      .value_counts(normalize = True)[1]*100),"%")

sns.catplot(x = 'Pclass' , y = 'Survived',kind = 'point',data = train);



print("Survival % of Male:", round(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100),"%")

print("Survival % of Female:", round(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100),"%")
sns.catplot(x = 'Sex' , y = 'Survived',kind = 'point',data = train);
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
g1=sns.catplot(x = 'SibSp',kind = 'count',data = train,ax = ax1);
g2=sns.catplot(x = 'SibSp' , y = 'Survived',kind = 'bar',data = train,ax = ax2);
plt.close(g1.fig)
plt.close(g2.fig)
plt.show()
for i in range(0,max(train["SibSp"])+1):
    if i in (6,7):
        continue
    else:
        print("Total passengers with", i , "siblings and/or spouse:" ,train["SibSp"].value_counts(sort = False)[i])
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
g1=sns.catplot(x = 'Parch',kind = 'count',data = train,ax = ax1);
g2=sns.catplot(x = 'Parch' , y = 'Survived',kind = 'bar',data = train,ax = ax2);
plt.close(g1.fig)
plt.close(g2.fig)
plt.show()
for i in range(0,max(train["Parch"])+1):
    print("Total passengers with", i , "parent or child:" ,train["Parch"].value_counts(sort = False)[i])
test.head(10)
test.isnull().sum()
# combining both train and test datasets because both have missing Age values
TrainTest = [train, test]

#extract a title for each Name in the train and test datasets
for data in TrainTest:
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])
#To ease the analysis combining the titles into fewer categories
for data in TrainTest:
    data['Title'] = data['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    data['Title'] = data['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

test.head()
#Calculating the average age according to each title
title= ['Mr','Miss','Mrs','Master','Royal','Rare'];
mr_age = round(train[train["Title"] == 'Mr']["Age"].mean()) 
print('Average age of title Mr: ',mr_age)
miss_age = round(train[train["Title"] == 'Miss']["Age"].mean())
print('Average age of title Miss: ',miss_age)
mrs_age = round(train[train["Title"] == 'Mrs']["Age"].mean())
print('Average age of title Mrs: ',mrs_age)
master_age = round(train[train["Title"] == 'Master']["Age"].mean())
print('Average age of title Master: ',master_age)
royal_age = round(train[train["Title"] == 'Royal']["Age"].mean())
print('Average age of title Royal: ',royal_age)
rare_age = round(train[train["Title"] == 'Rare']["Age"].mean())
print('Average age of title Rare: ',rare_age)
avg_age = [mr_age,miss_age,mrs_age,master_age,royal_age,rare_age]
#Filling the missing values in train dataset
n_rows= train.shape[0]   
n_titles= len(title)
for i in range(0, n_rows):
    if np.isnan(train.Age[i])==True:
        for j in range(0, n_titles):
            if train.Title[i] == title[j]:
                train.Age[i] = avg_age[j]

train['Age'].isnull().sum()
#Filling the missing values in test dataset  
n_rows= test.shape[0]   
n_titles= len(title)
for i in range(0, n_rows):
    if np.isnan(test.Age[i])==True:
        for j in range(0, n_titles):
            if test.Title[i] == title[j]:
                test.Age[i] = avg_age[j]

test['Age'].isnull().sum()

#Creating different AgeGroups
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()
print("Number of people embarking in S:",train[train["Embarked"] == "S"].shape[0]) 


print("Number of people embarking in C:",train[train["Embarked"] == "C"].shape[0])


print("Number of people embarking in Q:",train[train["Embarked"] == "Q"].shape[0])

sns.catplot(x='Embarked',kind = 'count',data=train)
plt.show()
train = train.fillna({"Embarked": "S"})
train['Embarked'].isnull().sum()
train.head(10)

test.head(10)
#Age Group
labelEncoder = LabelEncoder()
train.AgeGroup=labelEncoder.fit_transform(train.AgeGroup)
test.AgeGroup=labelEncoder.fit_transform(test.AgeGroup)
#Sex
train.Sex=labelEncoder.fit_transform(train.Sex)
test.Sex=labelEncoder.fit_transform(test.Sex)
train.Embarked=labelEncoder.fit_transform(train.Embarked)
test.Embarked=labelEncoder.fit_transform(test.Embarked)
train=train.drop(['Name','Age','Ticket','Fare','Cabin','Title'],axis=1)
test=test.drop(['Name','Age','Ticket','Fare','Cabin','Title'],axis=1)
train.head()
test.head()