import numpy as np
import pandas as pd 
from pandas import Series, DataFrame 
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
t_df = pd.read_csv('../input/train.csv')
t_df
t_df.head()
t_df.info()
t_df['Age'].hist(bins=70)
sns.factorplot('Sex', data=t_df, kind="count")
t_df.groupby('Sex')[['Survived']].mean()
sns.factorplot('Pclass',data=t_df,hue='Sex', kind='count')
#t_df.groupby(['Sex', 'Pclass'])['Survived'].aggregate('mean').unstack()
t_df.pivot_table('Survived', index= 'Sex', columns= 'Pclass')
Age = pd.cut(t_df['Age'], [0,18,80])
t_df.pivot_table('Survived', index= [Age, 'Sex'], columns = 'Pclass')

t_df.pivot_table(index='Sex', columns='Pclass',
                aggfunc={'Survived':sum, 'Fare':'mean'})
def male_female_child(passenger):
    Age,Sex = passenger
    if Age < 16:
        return 'child'
    else:
        return Sex
t_df['person'] = t_df[['Age','Sex']].apply(male_female_child,axis=1)
t_df[0:10]
sns.factorplot('Pclass',data=t_df,hue='person', kind='count')
t_df['person'].value_counts()
t_df.pivot_table('Survived', index='person', columns='Pclass', margins= True, margins_name="%survival")
deck = t_df['Cabin'].dropna()
deck.head()

levels = []

for level in deck:
    levels.append(level[0])  #prendi la prima lettera  

cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.factorplot('Cabin',data=cabin_df,palette='winter_d', kind='count')
cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.factorplot('Cabin',data=cabin_df,palette='summer', kind='count')
sns.factorplot('Embarked',data=t_df,hue='Pclass', kind='count')
t_df.pivot_table('Survived', index= 'Embarked', columns= 'Pclass')
t_df['Alone'] =  t_df.Parch + t_df.SibSp
t_df['Alone']

t_df['Alone'].loc[t_df['Alone'] >0] = 'With Family'
t_df['Alone'].loc[t_df['Alone'] == 0] = 'Alone'

sns.factorplot('Alone',data=t_df,palette='Blues', kind='count')
t_df["Survivor"] = t_df.Survived.map({0: "no", 1: "yes"})

sns.factorplot('Survivor',data=t_df,palette='Set1', kind='count')
t_df.pivot_table('Survived', index='Alone', columns='Pclass', margins= True, margins_name="%survival")
t_df.pivot_table('Survived', index='Alone', columns='Embarked', margins= True, margins_name="%survival")
