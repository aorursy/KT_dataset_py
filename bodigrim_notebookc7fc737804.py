#invite people for the Kaggle party

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#bring in the six packs

df_train = pd.read_csv('../input/train.csv')
#check the decoration

df_train.columns
df_train['Survived'] = (df_train['Survived'] == 1)

df_train['Survived'].describe()
df_train['PclassNeg'] = 3-df_train['Pclass']

df_train = df_train.drop('Pclass', axis=1)

df_train['PclassNeg'].describe()
sns.distplot(df_train['PclassNeg'],kde=False);
df_train['IsFemale'] = (df_train['Sex'] == "female")

df_train = df_train.drop('Sex', axis=1)

df_train['IsFemale'].describe()
#missing data

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#dealing with missing data

df_train = df_train.drop((missing_data[missing_data['Total'] > 100]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Embarked'].isnull()].index)

df_train.isnull().sum().max() #just checking that there's no missing data missing...
df_train = df_train.drop('Name', axis=1)

df_train = df_train.drop('Ticket', axis=1)

df_train.columns
df_train["Parch"].describe()
sns.distplot(df_train['Parch'], kde=False);
df_train["Fare"].describe()
df_train = df_train.drop(df_train.loc[df_train['Fare']==0].index)

df_train = df_train.drop(df_train.loc[df_train['Fare'] > 400].index)

sns.distplot(df_train['Fare']);
df_train["EmbarkedC"] = df_train["Embarked"]=="C"

df_train["EmbarkedQ"] = df_train["Embarked"]=="Q"

# df_train["EmbarkedS"] = df_train["Embarked"]=="S" is a combination of C and Q

df_train = df_train.drop('Embarked', axis=1)
#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
means = df_train.groupby(['IsFemale', 'PclassNeg', 'Parch', 'EmbarkedC'], as_index=False).mean()

means = means.drop('PassengerId', axis=1)

means = means.drop('SibSp', axis=1)

means = means.drop('EmbarkedQ', axis=1)

means