import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from pandas import Series,DataFrame



titanic_df = pd.read_csv('../input/titanic/train.csv')

%matplotlib inline

titanic_df.head()
titanic_df.info()
#titanic_df.describe()
sns.countplot('Sex',data=titanic_df)
sns.countplot('Sex',data=titanic_df,hue='Pclass')
def male_female_child(passenger):

    age,sex = passenger

    

    if age < 16 :

        return 'child'

    else:

        return sex
titanic_df['Person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)
#titanic_df[0:10]
sns.countplot('Pclass',data=titanic_df,hue='Person')
titanic_df['Age'].hist(bins=70)
titanic_df['Age'].mean()
titanic_df['Person'].value_counts()
fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)



oldest = titanic_df['Age'].max()



fig.set(xlim=(0,oldest))



fig.add_legend()
fig = sns.FacetGrid(titanic_df,hue='Person',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)



oldest = titanic_df['Age'].max()



fig.set(xlim=(0,oldest))



fig.add_legend()
fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)



oldest = titanic_df['Age'].max()



fig.set(xlim=(0,oldest))





fig.add_legend()
titanic_df['Pclass'].value_counts()
titanic_df.head(5)
deck = titanic_df['Cabin'].dropna()
deck.head()
lvl = []



for lvls in deck:

    lvl.append(lvls[0])



    cabin_df = DataFrame(lvl)

    cabin_df.columns = ['Cabin']

    sns.countplot('Cabin',data=cabin_df,palette='winter_d',order=['A','B','C','D','E','F','G','T'])

    
cabin_df = cabin_df[cabin_df.Cabin != 'T']

sns.countplot('Cabin',data=cabin_df,palette='summer',order=['A','B','C','D','E','F','G'])
sns.countplot('Embarked',data=titanic_df,hue='Pclass',order=['C','Q','S'])
titanic_df['Alone'] = titanic_df['SibSp'] + titanic_df['Parch']
titanic_df['Alone']
titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'ALone'
titanic_df.head()
sns.countplot('Alone',data=titanic_df,palette='summer')
titanic_df['Survivor'] = titanic_df.Survived.map({0:'no',1:'yes'})
#titanic_df.head()
sns.countplot('Survivor',data=titanic_df,palette="Set1")
sns.factorplot('Pclass','Survived',hue='Person',data=titanic_df)