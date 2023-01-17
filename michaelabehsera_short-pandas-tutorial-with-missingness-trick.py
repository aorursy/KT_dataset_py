import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import warnings
train = pd.read_csv('../input/titanic-dataset/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
x = train[['Pclass', 'Name', 'Sex', 'Age', 'Cabin', 'Embarked']]
y = train['Survived']
train.head()
#Checking the tail of the dataset / You can add a number in the function to increase the amount of rows you could see
train.tail(10)
#To check how many columns and rows
train.shape
#To check the mean, standard deviation and other stats of your columns
train.describe()
train.describe(include=['O']) #for categorical data
#To check total null values in a set
train.isnull().sum()
#To check for unique values in a vector
train.Embarked.unique()
type(x)
#Craft new columns based on missinginess (Vastly improves your score with the Titanic dataset)
train['Has_Cabin'] = x['Has_Cabin'] = np.where(train['Cabin'].isnull(), 0, 1)
x.head()
#Calculating the mean
x['Age'].mean()
#Calculating median
x['Age'].median()
#Viewing corralation in the data
x.corr()
#Describe
x.describe()
#Imputation for categorical variables
x = pd.get_dummies(data=x, columns=['Sex'])
x.head()
#Getting a benchmark score for your dependent variable
train.Survived.value_counts() / train.Survived.count() * 100
#Filling a missing value based on the most popular categorical variable
x['Embarked'].fillna('Q', inplace=True)
x.isnull().sum()
#Fill missing values with mean or median
x['Age'].fillna(x['Age'].mean(), inplace=True)
x.isnull().sum()
#Drop a column from dataframe
x.drop(['Cabin'], axis=1, inplace=True)
x.head()
