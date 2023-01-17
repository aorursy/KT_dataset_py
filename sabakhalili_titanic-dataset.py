import pandas as pd

from pandas import Series,DataFrame
titanic=pd.read_csv("../input/titanic/train.csv")
titanic.head(9)
titanic.info()
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.catplot('Sex',data=titanic,kind='count')
sns.catplot('Pclass',data=titanic,kind='count',hue='Sex')
def person_mfc(passenger):

    age,sex = passenger

    if age < 16:

        return 'child'

    else:

        return sex



    

titanic['person'] = titanic[['Age','Sex']].apply(person_mfc,axis=1)
titanic.head()
sns.catplot('Pclass',data=titanic,hue='person',kind='count')
titanic['Age'].hist(bins=70)

titanic['Age'].mean()
titanic['person'].value_counts()
fig = sns.FacetGrid(titanic, hue="Sex",aspect=4)



fig.map(sns.kdeplot,'Age',shade= True)





oldest = titanic['Age'].max()



fig.set(xlim=(0,oldest))



fig.add_legend()
fig = sns.FacetGrid(titanic, hue="person",aspect=4)

fig.map(sns.kdeplot,'Age',shade= True)

oldest = titanic['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()
titanic.head()
deck = titanic['Cabin'].dropna()
deck.head()
levels = []



for level in deck:

    levels.append(level[0])    



cabin_df = DataFrame(levels)

cabin_df.columns = ['Cabin']

sns.catplot('Cabin',data=cabin_df,palette='winter_d',kind='count',order=['A','B','C','D','E','T'])

cabin_df = cabin_df[cabin_df.Cabin != 'T']



sns.catplot('Cabin',data=cabin_df,palette='summer',kind='count',order=['A','B','C','D','E'])
titanic.head()
sns.catplot('Embarked',data=titanic,hue='Pclass',order=['C','Q','S'],kind='count')
titanic['Alone'] =  titanic.Parch + titanic.SibSp

titanic['Alone']
titanic['Alone'].loc[titanic['Alone'] >0] = 'With Family'

titanic['Alone'].loc[titanic['Alone'] == 0] = 'Alone'
titanic.head()
sns.catplot('Alone',data=titanic,palette='Blues',kind='count')