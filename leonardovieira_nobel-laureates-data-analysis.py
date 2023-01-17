import os

import pandas as pd

from pandas import DataFrame

import numpy as np



data = pd.read_csv('../input/archive.csv')

data.head()
ChemistryDF = data[(data.Category == 'Chemistry')]

EconomicsDF = data[(data.Category == 'Economics')]

LiteratureDF = data[(data.Category == 'Literature')]

MedicineDF = data[(data.Category == 'Medicine')]

PeaceDF = data[(data.Category == 'Peace')]

PhysicsDF = data[(data.Category == 'Physics')]
print(ChemistryDF['Birth Country'].value_counts())
import seaborn.apionly as sns

%matplotlib inline

import matplotlib.pyplot as plt



plt.figure(figsize=(10,12))

ChemestryGraph = sns.countplot(y="Birth Country", data=ChemistryDF,

              order=ChemistryDF['Birth Country'].value_counts().index,

              palette='GnBu_d')

plt.show()
print(EconomicsDF['Birth Country'].value_counts())
plt.figure(figsize=(10,12))

EconomicsGraph = sns.countplot(y="Birth Country", data=EconomicsDF,

              order=EconomicsDF['Birth Country'].value_counts().index,

              palette='GnBu_d')

plt.show()
print(LiteratureDF['Birth Country'].value_counts())
plt.figure(figsize=(10,12))

LiteratureGraph = sns.countplot(y="Birth Country", data=LiteratureDF,

              order=LiteratureDF['Birth Country'].value_counts().index,

              palette='GnBu_d')

plt.show()
print(MedicineDF['Birth Country'].value_counts())
plt.figure(figsize=(10,12))

MedicineGraph = sns.countplot(y="Birth Country", data=MedicineDF,

              order=MedicineDF['Birth Country'].value_counts().index,

              palette='GnBu_d')

plt.show()
print(PeaceDF['Birth Country'].value_counts())
plt.figure(figsize=(10,12))

PeaceGraph = sns.countplot(y="Birth Country", data=PeaceDF,

              order=PeaceDF['Birth Country'].value_counts().index,

              palette='GnBu_d')

plt.show()
print(PhysicsDF['Birth Country'].value_counts())
plt.figure(figsize=(10,12))

PhysicsGraph = sns.countplot(y="Birth Country", data=PhysicsDF,

              order=PhysicsDF['Birth Country'].value_counts().index,

              palette='GnBu_d')

plt.show()
from collections import Counter

import nltk



top_N = 16



stopwords = nltk.corpus.stopwords.words('english')

# RegEx for stopwords

RE_stopwords = r'\b(?:{})\b'.format('|'.join(stopwords))

# replace '|'-->' ' and drop all stopwords

words = (data.Motivation

           .str.lower()

           .replace([r'\|', RE_stopwords], [' ', ''], regex=True)

           .str.cat(sep=' ')

           .split()

)



# generate DF out of Counter

rslt = pd.DataFrame(Counter(words).most_common(top_N),

                    columns=['Word', 'Frequency'])

rslt = rslt[rslt.Word != '"'].reset_index()

del rslt['index']

rslt = rslt.set_index('Word')

print(rslt)

# plot

rslt.plot.bar(rot=0, figsize=(17,8), width=0.8)
from matplotlib.pyplot import pie, axis, show



ChemistryGender = ChemistryDF['Sex'].value_counts()

print(ChemistryGender)



pie(ChemistryGender, labels=ChemistryGender.index, autopct='%1.1f%%');

show()
import warnings

warnings.filterwarnings('ignore')

ChemistryDF['Birth Date'] = ChemistryDF['Birth Date'].str[0:4]

ChemistryDF['Birth Date'] = ChemistryDF['Birth Date'].replace(to_replace="nan", value=0)

ChemistryDF['Birth Date'] = ChemistryDF['Birth Date'].apply(pd.to_numeric)

ChemistryDF["Age"] = ChemistryDF["Year"] - ChemistryDF["Birth Date"]
bins = [0, 19, 29, 39, 49, 59, 69, 79, 89, 100]

groupNames = ['Teens', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s']

ChemistryDF['Age Categorical'] = pd.cut(ChemistryDF['Age'], bins, labels=groupNames)
ChemistryAge = ChemistryDF['Age Categorical'].value_counts()

print(ChemistryAge)



pie(ChemistryAge, labels=ChemistryAge.index, autopct='%1.1f%%');

show()
EconomicsGender = EconomicsDF['Sex'].value_counts()

print(EconomicsGender)



pie(EconomicsGender, labels=EconomicsGender.index, autopct='%1.1f%%');

show()
EconomicsDF['Birth Date'] = EconomicsDF['Birth Date'].str[0:4]

EconomicsDF['Birth Date'] = EconomicsDF['Birth Date'].replace(to_replace="nan", value=0)

EconomicsDF['Birth Date'] = EconomicsDF['Birth Date'].apply(pd.to_numeric)

EconomicsDF["Age"] = EconomicsDF["Year"] - EconomicsDF["Birth Date"]
bins = [0, 19, 29, 39, 49, 59, 69, 79, 89, 100]

groupNames = ['Teens', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s']

EconomicsDF['Age Categorical'] = pd.cut(EconomicsDF['Age'], bins, labels=groupNames)



EconomicsAge = EconomicsDF['Age Categorical'].value_counts()

print(EconomicsAge)



pie(EconomicsAge, labels=EconomicsAge.index, autopct='%1.1f%%');

show()
LiteratureGender = LiteratureDF['Sex'].value_counts()

print(LiteratureGender)



pie(LiteratureGender, labels=LiteratureGender.index, autopct='%1.1f%%');

show()
LiteratureDF['Birth Date'] = LiteratureDF['Birth Date'].str[0:4]

LiteratureDF['Birth Date'] = LiteratureDF['Birth Date'].replace(to_replace="nan", value=0)

LiteratureDF['Birth Date'] = LiteratureDF['Birth Date'].apply(pd.to_numeric)

LiteratureDF["Age"] = LiteratureDF["Year"] - LiteratureDF["Birth Date"]
bins = [0, 19, 29, 39, 49, 59, 69, 79, 89, 100]

groupNames = ['Teens', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s']

LiteratureDF['Age Categorical'] = pd.cut(LiteratureDF['Age'], bins, labels=groupNames)



LiteratureAge = LiteratureDF['Age Categorical'].value_counts()

print(LiteratureAge)



pie(LiteratureAge, labels=LiteratureAge.index, autopct='%1.1f%%');

show()
MedicineGender = MedicineDF['Sex'].value_counts()

print(MedicineGender)



pie(MedicineGender, labels=MedicineGender.index, autopct='%1.1f%%');

show()
MedicineDF['Birth Date'] = MedicineDF['Birth Date'].str[0:4]

MedicineDF['Birth Date'] = MedicineDF['Birth Date'].replace(to_replace="nan", value=0)

MedicineDF['Birth Date'] = MedicineDF['Birth Date'].apply(pd.to_numeric)

MedicineDF["Age"] = MedicineDF["Year"] - MedicineDF["Birth Date"]
bins = [0, 19, 29, 39, 49, 59, 69, 79, 89, 100]

groupNames = ['Teens', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s']

MedicineDF['Age Categorical'] = pd.cut(MedicineDF['Age'], bins, labels=groupNames)



MedicineAge = MedicineDF['Age Categorical'].value_counts()

print(MedicineAge)



pie(MedicineAge, labels=MedicineAge.index, autopct='%1.1f%%');

show()
PeaceGender = PeaceDF['Sex'].value_counts()

print(PeaceGender)



pie(PeaceGender, labels=PeaceGender.index, autopct='%1.1f%%');

show()
PeaceDF['Birth Date'] = PeaceDF['Birth Date'].str[0:4]

PeaceDF['Birth Date'] = PeaceDF['Birth Date'].replace(to_replace="nan", value=0)

PeaceDF['Birth Date'] = PeaceDF['Birth Date'].apply(pd.to_numeric)

PeaceDF["Age"] = PeaceDF["Year"] - PeaceDF["Birth Date"]
bins = [0, 19, 29, 39, 49, 59, 69, 79, 89, 100]

groupNames = ['Teens', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s']

PeaceDF['Age Categorical'] = pd.cut(PeaceDF['Age'], bins, labels=groupNames)



PeaceAge = PeaceDF['Age Categorical'].value_counts()

print(PeaceAge)



pie(PeaceAge, labels=PeaceAge.index, autopct='%1.1f%%');

show()
PhysicsGender = PhysicsDF['Sex'].value_counts()

print(PhysicsGender)



pie(PhysicsGender, labels=PhysicsGender.index, autopct='%1.1f%%');

show()
PhysicsDF['Birth Date'] = PhysicsDF['Birth Date'].str[0:4]

PhysicsDF['Birth Date'] = PhysicsDF['Birth Date'].replace(to_replace="nan", value=0)

PhysicsDF['Birth Date'] = PhysicsDF['Birth Date'].apply(pd.to_numeric)

PhysicsDF["Age"] = PhysicsDF["Year"] - PhysicsDF["Birth Date"]
bins = [0, 19, 29, 39, 49, 59, 69, 79, 89, 100]

groupNames = ['Teens', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s']

PhysicsDF['Age Categorical'] = pd.cut(PhysicsDF['Age'], bins, labels=groupNames)



PhysicsAge = PhysicsDF['Age Categorical'].value_counts()

print(PhysicsAge)



pie(PhysicsAge, labels=PhysicsAge.index, autopct='%1.1f%%');

show()
c = data['Organization Name'].value_counts()

plt.figure(figsize=(5,12))

UniversitiesGraph = sns.countplot(y="Organization Name", data=data,

              order=c.nlargest(50).index,

              palette='Reds')

plt.show()