# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#HW1 - Subtask 1: Analyze by describing data

#Exploring Training Data Set 

dfTrain = pd.read_csv('/kaggle/input/titanic/train.csv')

dfTest = pd.read_csv('/kaggle/input/titanic/test.csv')

dataset = pd.concat([dfTrain, dfTest])

dfTest.head()

dfTrain.head()

dataset.head()



#Q4 mixed data types

dataset.Ticket.unique()



#Q5 which features contain na values

dataset.isna().sum()



#Q6 datatype of each feature

dataset.dtypes



#Splitting into numeric/cetegorical data

num = dataset[['Age', 'SibSp', 'Parch', 'Fare']]

cat = dataset.drop(['Age', 'SibSp', 'Parch', 'Fare'], axis=1)

num.head()

cat.head()



#Q7 Summary Statistics of numerical features

num.describe()



#Q8 Summary statistics of categorical features

cat = cat.astype(str)

cat.describe()
#HW2 - Subtask 2: Analyze by pivoting features

import seaborn as sns

corr = dfTrain.corr()

sns.heatmap(corr)



#Q9 - effect of Pclass on survival

dfTrain[['Pclass', 'Survived']].groupby(['Pclass']).mean()



#Q10 - effect of gender on survival

dfTrain[['Sex', 'Survived']].groupby(['Sex']).mean()
#Q11 Histograms

q11 = dfTrain[['Age', 'Survived']]

q11a = q11[q11.Survived == 0]

q11b = q11[q11.Survived == 1]



import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2)

fig.suptitle('Age Histograms')

ax1.hist(q11a.Age, 10)

ax1.set_title('Ded - Survived = 0')

ax2.hist(q11b.Age, 10)

ax2.set_title('Liv - Survived = 1')

myplt = sns.FacetGrid(dfTrain, col='Survived', row='Pclass')

myplt.map(plt.hist, 'Age', bins=10)

myplt.add_legend();
myplt = sns.FacetGrid(dfTrain, row='Embarked', col='Survived')

myplt.map(sns.barplot, 'Sex', 'Fare', ci=0)

myplt.add_legend();
#Q16 - encoding Gender

mapping = {'male': 0, 'female': 1}

dfTrain['Gender'] = dfTrain.Sex.map(mapping).fillna(dfTrain.Sex)

dfTrain.drop(['Sex'], axis=1, inplace=True)

#dfTrain.head()
#Q17 - Imputing missing age data with KNN

#! pip install -U scikit-learn

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=2)

dfTrain_num = dfTrain[['Age', 'SibSp', 'Parch', 'Fare']][:]

dfTrain_num = pd.DataFrame(data = imputer.fit_transform(dfTrain_num), columns = dfTrain_num.columns, index=dfTrain_num.index)

dfTrain.Age = dfTrain_num.Age

dfTrain.isna().sum()
#Q18 - Impute missing embarked data with most common

from sklearn.impute import SimpleImputer

imp_freq = SimpleImputer(strategy = "most_frequent")

dfTrain = pd.DataFrame(data = imp_freq.fit_transform(dfTrain), columns = dfTrain.columns, index=dfTrain.index)
#Q19 - Impute test fare data with mode

dfTest = pd.DataFrame(data = imp_freq.fit_transform(dfTest), columns = dfTest.columns, index=dfTest.index)
#Q20 - Convert Fare data into ordinal

bins = [-0.001, 7.91, 14.454, 31, 512.329]

labels = [0, 1, 2, 3]

dfTrain['FareBand'] = pd.cut(dfTrain['Fare'], bins = bins, labels = labels)

dfTrain.drop('Fare', axis=1, inplace=True)