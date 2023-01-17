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
df.head()
#1. asking questions
#2. data preprocessing
#3. EDA
#4. Drawing conclusions
#5. Communicating
#1. ASKING QUESTIONS
#what cols will contribute in my analysis
#what cols are not useful
#   DATA PREPROCESSING
#1. Gathering data
#2. Assesing data
#--- a) incorrect data types [Pclass, Sex,Age, Embarked, Survived]
#--- b) Missing values in [Age, Cabin, embarked]


#shape
df.shape

#datatype of cols
df.info()
#check for missing values
df.isnull().sum()
#describe (mathemical summary of numerical cols)
df.describe().T 
#(T - transposed)
#Handling missing values

#fillna
#dropna
# Handling Missing Values

#handling age missing values
# (can replace by mean, median etc)
df['Age']=df['Age'].fillna(df['Age'].mean())
df.isnull().sum()
# in output see age is 0 which means age has no missing values
#handling cabin missing values

(687/891)*100 #(percebtage of missing values)
#(no of missing values/total rows)*100
df['Cabin']=df['Cabin'].fillna('No Cabin')
df.isnull().sum()
#handling embarked missing values
df.dropna(subset=['Embarked'],inplace=True)
#if embarked has missing value drop or del the row
#inplace=true makes permanent changes to the database.
df.isnull().sum()
#handling incorrect data types
# changing the incorrect data type 
df['Pclass']=df['Pclass'].astype('category')
df['Sex']=df['Sex'].astype('category')
df['Embarked']=df['Embarked'].astype('category')
df['Age']=df['Age'].astype('int32')


df['Survived']=df['Survived'].astype('category')
df['SibSp']=df['SibSp'].astype('int32')
df['Parch']=df['Parch'].astype('int32')
df['Fare']=df['Fare'].astype('float32')


#df.drop('abc',axis=1,inpace=True)
#abc is columnname
#axis=1 is column
#axis=0 is row
#inpace=true makes permannet operation
df.info()
#Maintaining a copy of data
df_copy=df.copy()
#can be used in case we make a mistake in original dataframe df
# EDA (Exploratory Data Analysis)

#1. Do all the univariate analysis
#2. Do bivariate analysis with the target column
#3. If possible do multivariate analysis
#4. Find corelation
import seaborn as sns
import matplotlib.pyplot as plt
df['PassengerId'].value_counts()
#each Id is a different category so this is useless
df.drop('PassengerId',axis=1,inplace=True)
df.head()
#Survived column (univariate analysis)(categorical column)

(df['Survived'].value_counts()/df.shape[0])*100
#percentage of people died and survived]

sns.countplot(df['Survived'])
#Pclass column (univariate analysis)(categorical column)

(df['Pclass'].value_counts()/df.shape[0])*100
sns.countplot(df['Pclass'])
# Sex column (categorical column)

(df['Sex'].value_counts()/df.shape[0])*100
sns.countplot(df['Sex'])
#Age column (numerical column)

plt.hist(df['Age'],bins=8)
plt.show()
sns.boxplot(df['Age'])
#SibSp column
df['SibSp'].value_counts()
#without siblings and spouse 606 people are there
#with 1 siblings or spouse 209 people are there
#with 2 siblings or spouse 28 people are there
# etc....
#Parch column
df['Parch'].value_counts()
#without any parent or child 676 people are there
#with 1 parent or child 118 people are there
#with 2 parent or child 80 people are there
# etc....
# Fare column

sns.boxplot(df['Fare'])
sns.distplot(df['Fare'])
#right sqewed data because most of them are towards left side
# tail jis taraf hai us side sqewed data hota hai
#Cabin column

889-df['Cabin'][df['Cabin']=='No Cabin'].shape[0]
(202/889)*100
# Embarked column
(df['Embarked'].value_counts()/df.shape[0])*100

#4. CONCLUSIONS

# a) PassengerId doesnt help in my analysis, so drop it
# b) More people died(~62%) than survived(~38%)
# c) Pclass3(~55%) was more populated than pclass1(~24%) and pclass2(~20%)
# d) There were more males(~64%) than females(~35%)
# e) The range of age is 0-80 years and there a quite a few outliers in data.
# f) Fare column has too many outliers, and data is right sqewed
# g) Around 22% passengers were travelling in cabin
# h) Most of the passengers onboarded from Southampton(~72%),then from Cherbourg(~18%) and then from Queenstown(~8%)