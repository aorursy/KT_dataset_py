import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn import preprocessing



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
train.isnull().sum()
sns.heatmap(train.isnull(), yticklabels = False, cbar=False, cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived', data=train)
sns.set_style('whitegrid')

sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived', hue='Pclass', data=train, palette='rainbow')
sns.distplot(train['Age'].dropna(), kde = False, color = 'darkred', bins=40)
train['Age'].hist(bins=30, color='darkred', alpha=0.3)
sns.countplot(x='SibSp', data=train)
train['Fare'].hist(bins=40, color='green', figsize=(8,4))
plt.figure(figsize = (12,7))

sns.boxplot(x='Pclass', y='Age', data=train, palette='winter')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        

        if Pclass == 1 :

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
sns.heatmap(train.isnull(), yticklabels = False, cbar=False, cmap='viridis')
train.drop('Cabin', axis=1, inplace=True)
sns.heatmap(train.isnull(), yticklabels = False, cbar=False, cmap='viridis')
train.info()
pd.get_dummies(train['Embarked'], drop_first=True).head()
embark = pd.get_dummies(train['Embarked'], drop_first = True)

sex = pd.get_dummies(train['Sex'], drop_first = True)
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)
train.head()
train = pd.concat([train, sex, embark], axis = 1)
train.head()
from catboost import CatBoostClassifier
# Making Features and Target Seperate from dataset

X = train.drop(['Survived'], axis=1)

Y = train['Survived']



# Taking 80% data for training

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,train_size=0.8,random_state=42)
model_LR = clf =CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)



#now just to make the model to fit the data

clf.fit(xtrain,ytrain,eval_set=(xtest,ytest), early_stopping_rounds=50)
model_LR.score(xtest, ytest)
# Import Student Data 

data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
#Seeing the first 10 Data

data.head()
data.describe()
data.columns
data.nunique()
data['gender'].unique()
#Finding the number of null values 

data.isnull().sum()
# Remove the Irrelevent Columns

student = data.drop(['race/ethnicity','parental level of education'], axis=1)
student.head()
# Finding the correlation of student data

correlation = student.corr()
# Visualizing the correlation of the student data

sns.heatmap(correlation, xticklabels = correlation.columns, yticklabels = correlation.columns, annot = True)
sns.pairplot(student)
sns.relplot(x = 'math score', y = 'reading score', hue = 'gender', data = student)
sns.distplot(student['math score'])
sns.distplot(student['writing score'])
sns.distplot(student['writing score'], bins=5)
sns.catplot(x='math score', kind='box', data = student)