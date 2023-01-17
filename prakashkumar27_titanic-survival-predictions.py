#Imporing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display
%matplotlib inline
#Read the data
train = pd.read_csv("../input/titanic/train.csv")
train.head()
#Read the columns in the data
train.columns
#Set the index to the passangerid
train = train.set_index('PassengerId')
train.head()
#Get the count of rows and columns
train.shape
# identify datatypes of the each columns
datadict = pd.DataFrame(train.dtypes)
datadict
#Identify the missing value in the dataset
datadict['MissingVal'] = train.isnull().sum()
datadict
# Identify the count for each variable, add the stats to datadict
datadict['Count']=train.count()
datadict
# rename the 0 column as DataType
datadict = datadict.rename(columns={0:'DataType'})
datadict
# get discripte statistcs on "object" datatypes
train.describe(include=['object'])
# get discriptive statistcs on "number" datatypes
train.describe(include=['number'])
#Get the survived values count
train.Survived.value_counts(normalize=True)
#Count plot for the columns that are more influnced
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
sns.countplot('Survived',data=train,ax=axes[0,0])
sns.countplot('Pclass',data=train,ax=axes[0,1])
sns.countplot('Sex',data=train,ax=axes[0,2])
sns.countplot('SibSp',data=train,ax=axes[0,3])
sns.countplot('Parch',data=train,ax=axes[1,0])
sns.countplot('Embarked',data=train,ax=axes[1,1])
sns.distplot(train['Fare'], kde=True,ax=axes[1,2])
sns.distplot(train['Age'].dropna(),kde=True,ax=axes[1,3])
#Data using groupby columns with box plot
figbi, axesbi = plt.subplots(2, 4, figsize=(16, 10))
train.groupby('Pclass')['Survived'].mean().plot(kind='barh',ax=axesbi[0,0],xlim=[0,1])
train.groupby('SibSp')['Survived'].mean().plot(kind='barh',ax=axesbi[0,1],xlim=[0,1])
train.groupby('Parch')['Survived'].mean().plot(kind='barh',ax=axesbi[0,2],xlim=[0,1])
train.groupby('Sex')['Survived'].mean().plot(kind='barh',ax=axesbi[0,3],xlim=[0,1])
train.groupby('Embarked')['Survived'].mean().plot(kind='barh',ax=axesbi[1,0],xlim=[0,1])
sns.boxplot(x="Survived", y="Age", data=train,ax=axesbi[1,1])
sns.boxplot(x="Survived", y="Fare", data=train,ax=axesbi[1,2])
#joint plot for Age and Fare
sns.jointplot(x="Age", y="Fare", data=train);
#Heatmap for the survival ratio, using correlation matrix
f, ax = plt.subplots(figsize=(10, 8))
corr = train.corr()
sns.heatmap(corr,
            mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
train.tail()
