# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/titanic/train.csv')
df.head(5)
# 1 Asking Questions
#2  Data Preprocessing
#3  EDA
#4  Drawing Conclusion
#5  Communicating

#Asking questions
#1 What columns will contribute in analysis
#2 What colums are not useful

#Data Preprocessing
#1 gathering data
#2 Assising Data
# --------a)Incorrect Data type[Pclass,sex,Age,embarked]
#---------b)missing values[Age,cabin,embarked]
#3 cleaning Data

#shape
df.shape
#data type of cols
df.info()
#mathamatical colums
df.describe().T
#check for missing values
df.isnull().sum()
#Handling missing values
#fillna

#dropna
#handling age missing values
df['Age']=df['Age'].fillna(df['Age'].mean())

#handling cabin missing values

df['Cabin']=df['Cabin'].fillna('No Cabin')
df.isnull().sum()
#Handling embarked missing value
df.dropna(subset=['Embarked'],inplace=True)
df.isnull().sum()
#Handling Incorrect Datatype
df['Pclass']=df['Pclass'].astype('category')
df['Sex']=df['Sex'].astype('category')
df['Sex']=df['Sex'].astype('category')
df['Age']=df['Age'].astype('int32')
df['Embarked']=df['Embarked'].astype('category')      
#df['Survived']=df['Survived'].astype('category')
#drop parc
#df.drop('parch',axis=1,inplace=True)
#Eda-Exploratory Data Analysis
#1.Do all univariate analysis
#2 Do bivariate analysis with target column
#3 if possible do multivariate analysis
#4 Find correlation
import seaborn as sns 
import matplotlib.pyplot as plt
df.drop('PassengerId',axis=1,inplace=True)
df.head()
#survive
(df['Survived'].value_counts()/df.shape[0])*100
sns.countplot(df['Survived'])
#pclass
(df['Pclass'].value_counts()/df.shape[0])*100
#sex
(df['Sex'].value_counts()/df.shape[0])*100
#Age col
plt.hist(df['Age'],bins=8)

sns.distplot(df['Age'])
sns.boxplot(df['Age'])
#sibsp
df['SibSp'].value_counts()
#Parch
df['Parch'].value_counts().plot(kind='bar')
#Fare
sns.boxplot(df['Fare'])
sns.distplot(df['Fare'])
#cabin
889-df[df['Cabin']=='No Cabin'].shape[0]
(202/889)*100
# Embarked
(df['Embarked'].value_counts()/df.shape[0])*100
#Bivariate/Multivariate analysis
#1 . Pclass V Survived
sns.heatmap(pd.crosstab(df['Pclass'],df['Survived']))
df.groupby('Pclass').mean()['Survived']*100
#sex vs Survived

sns.heatmap(pd.crosstab(df['Sex'],df['Survived']))
df.groupby('Sex').mean()['Survived']

#Embarked vs Survive
df.groupby('Embarked').mean()['Survived']
df.groupby(['Embarked','Sex']).count()['Survived']
sns.distplot(df[df['Survived']==0]['Age'])
sns.distplot(df[df['Survived']==1]['Age'])
sns.distplot(df[df['Survived']==0]['Fare'])
sns.distplot(df[df['Survived']==1]['Fare'])
#dropping the paasenger id and ticket 
df.drop(column=['Name','Ticket'],inplace=True)

df.head(1)
def process_cabin(value):
    if value=='No Cabin':
        return 0
    else:
        return 1
# Feature Enginnering
df['Cabin']=df['Cabin'].apply(process_cabin)
df['family']=df['SibSp']
df.head()
def process_family(value):
    if value==1:
        return "Alone"
    elif value>1 and value<4:
        return "Small"
    else:
        return "Large"
        
df['family_type']=df['family'].apply(process_family)
df.head()
df.drop(['SibSp','Parch','family'],axis=1,inplace=True)
df.head()
df.groupby('family_type').mean()['Survived']
#Conclusions
# 1 PassengerId doesnt heslp in my analysis.so drop it
#2 More people died(-62) than survived(-38)
#3 Pclass 3(55%) was more populated than Pclass1(24%) and Pclass2(20%)
#4 More males(64%) on board than females(35%)
#5 The range of age is 0 to 80 years.There are quite a few iutliers in data
#6 Fare has too many outliers and data is right skewed data
#7 Around 22% passsengers are traveling in cabin
#8 Most of the passengers onboarded in South Hampton(72%)
#9 Pclass3(~24% survived) was much more dangerous in comparison to Pclass1('~62%' survived) and Pclass2(~42% survived)
#10 In comparision to males(just 18% survived) a lot more females(74% survived) survived
#11 There is an anamoly..somehow people boarding from Chehbourg has survived more in comparison to other 2 cities.
#12 The probability of surviving is higher in smaller and older age group in com to 15-45 age group
#13 Higer Fare 
#14 having a cabin increases your chances  of survival(66% survived with cabin)
#15 Travelling with small families is better in comparision to travelling alone or travellingwith large families.
sns.pairplot(df)
df.corr()['Survived']
df.info()
