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
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.sample(5)
# 1. Asking Questions
# 2. Data Preprocessing
# 3. EDA
# 4. Drawing Conclusions
# 5. Communicating
# Asking Questions

# 1. What cols will contribute in my analysis
# 2. What cols are not useful
# 3. 

# Data Preprocessing

# 1. Gathering Data [Done]
# 2. Assessing data
# ------ a. Incorrect data types [Pclass, Sex, Age, Embarked, Survived]
# ------ b. Missing values in [Age, Cabin, Embarked]
# 3. Cleaning Data
# Shape of the data

df.shape
# Data type of cols

df.info()
# Mathematical cols

df.describe().T
# Check for missing values

df.isnull().sum()
# Handling Missing values

# handling age missing values

df['Age'] = df['Age'].fillna(df['Age'].mean())
# Handling cabin missing values

df['Cabin'] = df['Cabin'].fillna('No Cabin')
# Handling embarked missing value

df.dropna(subset=['Embarked'],inplace=True)
df.isnull().sum()
# Handling incorrect data types

df['Pclass'] = df['Pclass'].astype('category')
df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
df['Age'] = df['Age'].astype('int32')
#df['Survived'] = df['Survived'].astype('category')
#df['SibSp'] = df['SibSp'].astype('int32')
df['Parch'] = df['Parch'].astype('int32')
#df['Fare'] = df['Fare'].astype('float32')
# Drop parch

df.drop('parch', axis=1,inplace=True)
# Maintain a copy of your data

df_copy = df.copy()
df.info()
# EDA - Exploratory Data Analysis

# 1. Do all the univariate analysis
# 2. Do bivariate analysis with the target col
# 3. If possible do multivariate analysis
# 4. Find correlation
import seaborn as sns
import matplotlib.pyplot as plt
df.drop('PassengerId',axis = 1, inplace=True)
df.head()
# Survived

(df['Survived'].value_counts()/df.shape[0])*100
sns.countplot(df['Survived'])

# More people died(~62%) than survived(~38%)
# Pclass

(df['Pclass'].value_counts()/df.shape[0])*100
# Sex col

(df['Sex'].value_counts()/df.shape[0])*100
# Age col

plt.hist(df['Age'],bins=8)
plt.show()
sns.distplot(df['Age'])
sns.boxplot(df['Age'])
# SibSp

df['SibSp'].value_counts()
# Parch

df['Parch'].value_counts().plot(kind='bar')
# Fare

sns.boxplot(df['Fare'])
sns.distplot(df['Fare'])
889 - df['Cabin'][df['Cabin'] == 'No Cabin'].shape[0]

(202/889)*100
# Embarked

(df['Embarked'].value_counts()/df.shape[0])*100
# Conclusions

# 1. PassengerId doesnt helps in my analysis, so drop it
# 2. More people died(~62%) than survived(~38%)
# 3. Pclass 3(~55%) was more populated than Pclass 1(~24%) and 2(20%)
# 4. More males(~65%) were onboard in comparison to females(~35%)
# 5. The range of age is 0 to 80 years. There are quite a few outliers in data.
# 6. Fare col has too many outliers, and data is right skewed
# 7. Around 22 percent passengers were travelling in Cabin
# 8. Most of the passengers onboarded in Southampton(~72%)
