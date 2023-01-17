# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd 

from pandas import Series, DataFrame



titanic_df = pd.read_csv('../input/train.csv')
titanic_df.head(3)


titanic_df.info()


import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.catplot("Sex",kind="count",data=titanic_df)
sns.catplot("Sex",kind="count",hue="Pclass",data=titanic_df)
sns.catplot("Pclass",kind="count",hue="Sex",data=titanic_df)
def male_female_child(passenger):

        age,sex = passenger

        

        if age < 16:

            return 'child'

        else:

            return sex
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)
titanic_df[0:10]
sns.catplot('Pclass', kind ='count', hue ='person',data=titanic_df)
titanic_df['Age'].hist(bins=70)
titanic_df['Age'].mean()

titanic_df['person'].value_counts()
fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)



fig.map(sns.kdeplot, 'Age', shade=True)



oldest = titanic_df['Age'].max()



fig.set(xlim=(0,oldest))



fig.add_legend()
fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)



fig.map(sns.kdeplot, 'Age', shade=True)



oldest = titanic_df['Age'].max()



fig.set(xlim=(0,oldest))



fig.add_legend()
fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)



fig.map(sns.kdeplot, 'Age', shade=True)



oldest = titanic_df['Age'].max()



fig.set(xlim=(0,oldest))



fig.add_legend()
deck = titanic_df['Cabin'].dropna()
deck.head()
levels =[]

for level in deck:

    levels.append(level[0])

    

cabin_df = DataFrame(levels)



cabin_df.columns = ['Cabin']

cabin_df = cabin_df.sort_values(['Cabin'])

sns.catplot('Cabin',kind='count',data=cabin_df,palette='winter_d')

cabin_df = cabin_df[cabin_df.Cabin != 'T']

sns.catplot('Cabin',kind='count',data=cabin_df,palette='cubehelix')
titanic_df.head(3)


sns.catplot(x="Age", y="Embarked",data=titanic_df,hue="Sex",col="Pclass", col_wrap=3,height=4, aspect=1, dodge=True, palette="Set3",kind="violin", order=['C','Q','S'])

titanic_df.head(3)
titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch
titanic_df['Alone'].head()
titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With Family'



titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'

titanic_df.head(3)
sns.catplot('Alone',kind='count',data=titanic_df,palette ='summer')
titanic_df['Survivor'] = titanic_df.Survived.map({0:'no',1:'yes'})
sns.catplot('Survivor',data=titanic_df,kind='count',palette='Set1')
sns.catplot('Pclass','Survived', kind = "point",data=titanic_df)
sns.catplot('Pclass','Survived', hue='person', kind = "point",data=titanic_df)
sns.lmplot('Age','Survived',data=titanic_df)
sns.lmplot('Age','Survived',hue='Pclass',palette='winter',data=titanic_df)
generations = [10,20,40,60,80]

sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)
sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)
titanic_df.head(3)
titanic_df['Deck'] = cabin_df

sns.catplot('Survivor',col='Deck',col_wrap=4,data=titanic_df[titanic_df.Deck.notnull()],kind="count",height=3.3, aspect=.9,palette='rocket')

sns.catplot('Alone',kind="count",hue='Survivor',data=titanic_df,palette='rocket')