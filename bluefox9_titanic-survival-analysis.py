import pandas as pd

import numpy as np

import os
data_1 = pd.read_csv("../input/train.csv")

data_1.head()
data_1.columns
data_1.info()
import seaborn as sns
#data_1["Sex"].hist(bins=10)

sns.catplot(x='Sex',data=data_1,kind='count')
data_1["Sex"].value_counts()
sns.catplot('Pclass',data=data_1,kind='count')
data_1.Pclass.value_counts()
sns.catplot(x='Pclass',hue='Sex',data=data_1,kind='count')
def male_female_child(passenger):



    age,sex = passenger

    

    if age > 10:

        return 'child'

    else:

        return sex

    

data_1['person'] = data_1[['Age','Sex']].apply(male_female_child,axis=1)

data_1.head(10)
sns.catplot(x='person',hue='Sex',data=data_1,kind='count')
pd.crosstab(data_1["Pclass"],data_1["person"],margins=True)
#pd.crosstab(index=data_1['Pcalss'],columns=data_1['person'])
data_1['Age'].hist(bins=70)
data_1['person'].value_counts()
fig = sns.FacetGrid(data_1,hue="Sex",aspect=4)

fig.map(sns.kdeplot,'Age',shade='True')

oldset = data_1['Age'].max()

fig.set(xlim=(0,oldset))

fig.add_legend()
fig = sns.FacetGrid(data_1,aspect=4)

fig.map(sns.kdeplot,'Age',shade='True')

oldset = data_1['Age'].max()

fig.set(xlim=(0,oldset))

fig.add_legend()
fig = sns.FacetGrid(data_1,hue="person",aspect=5)

fig.map(sns.kdeplot,'Age',shade='True') # KDE

oldset = data_1['Age'].max() # max limit by the oldest passengers

fig.set(xlim=(0,oldset))

fig.add_legend()
fig = sns.FacetGrid(data_1,hue="Pclass",aspect=4)

fig.map(sns.kdeplot,'Age',shade='True')

oldset = data_1['Age'].max()

fig.set(xlim=(0,oldset))

fig.add_legend()
data_1['Cabin'].value_counts()
deck = data_1['Cabin'].dropna()
levels = []



for level in deck:

    levels.append(level[0])



    levels.sort()

cabin_df = pd.DataFrame(levels)

cabin_df.columns = ['Cabin']

sns.factorplot('Cabin',data=cabin_df,palette='winter_d',kind='count')
levels = []



for level in deck:

    levels.append(level[0])



    levels.sort()

cabin_df = pd.DataFrame(levels)

cabin_df.columns = ['Cabin']

sns.catplot('Cabin',data=cabin_df,kind='count')
data_1.head()
data_1.groupby('Embarked').agg(['count', 'size', 'nunique']).stack()
sns.catplot(data_1['Embarked'],data=data_1,hue='person',kind='count')
by = data_1.groupby(by=['Embarked','person'])
by = data_1.groupby(by=['Embarked','Pclass','person'])
by['PassengerId'].count()
by.count()
data_1['Alone'] = data_1.Parch + data_1.SibSp

data_1['Alone']
data_1.head()
data_1['Alone'].loc[3]
data_1.head()
data_1['Alone'].value_counts()
sns.catplot(x='Alone',hue='Sex',data=data_1,kind='count')
sns.catplot(x='Alone',hue='person',data=data_1,kind='count')
data_1['Survivor'] = data_1.Survived.map({0:"no",1:"yes"})



sns.catplot('Survivor',data=data_1,kind='count')

data_1["Survivor"].value_counts()
sns.catplot('Survivor',data=data_1,kind='count',hue='person')
sns.factorplot('Pclass','Survived',data=data_1)
sns.factorplot('Pclass','Survived',data=data_1,hue='person')
sns.lmplot('Age','Survived',data=data_1)
sns.lmplot('Age','Survived',data=data_1,hue='Pclass')
sns.lmplot('Age','Survived',data=data_1,hue='Sex')
sns.lmplot('Age','Survived',data=data_1,hue='Pclass')