# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
titanic_df=pd.read_csv('../input/train.csv')
Y=pd.read_csv('../input/test.csv')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
titanic_df.head()
#as it has some null values we replace them.
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
titanic_df["Age"].fillna(Y["Age"].mean(), inplace=True)
titanic_df["Fare"].fillna(titanic_df["Fare"].mean(), inplace=True)
titanic_df.head()
titanic_df.info()
#who were the passengers on the titanic
sns.factorplot('Sex',data=titanic_df,kind="count")
sns.factorplot('Sex',data=titanic_df,hue='Pclass',kind='count')
sns.factorplot('Pclass',data=titanic_df,hue='Sex',kind='count')
def male_female_child(passenger):
    age,sex = passenger
    
    if age < 16:
        return 'child'
    else:
        return sex
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)
sns.factorplot('Pclass',data=titanic_df,kind='count',hue='person')
titanic_df['Age'].hist(bins=40)
titanic_df['person'].value_counts()
fig=sns.FacetGrid(titanic_df,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest=titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
fig=sns.FacetGrid(titanic_df,hue='person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest=titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
deck=titanic_df['Cabin'].dropna()
deck.head()
#classify deck levels
from pandas import Series,DataFrame
levels=[]
for level in deck:
    levels.append(level[0])

cabin_df=DataFrame(levels)
cabin_df.columns=['Cabin']
sns.factorplot('Cabin',data=cabin_df,palette='winter_d',kind='count')
#drop T
cabin_df=cabin_df[cabin_df.Cabin != 'T']
sns.factorplot('Cabin',data=cabin_df,palette='summer',kind='count')
#where do the people came from
sns.factorplot('Embarked',data=titanic_df,hue='Pclass',order=['C','Q','S'],kind='count')
#alone/family
#addition implies that if sum is 0 they don't have any family onboard
titanic_df['Alone']= titanic_df.SibSp + titanic_df.Parch
titanic_df['Alone'].head()
titanic_df["Alone"].loc[titanic_df['Alone'] > 0] = 'with family'

titanic_df["Alone"].loc[titanic_df['Alone'] == 0] = 'Alone'
titanic_df.head()
sns.factorplot('Alone',data=titanic_df,palette='Blues',kind='count')
titanic_df['Survivor'] = titanic_df.Survived.map({0:'No',1:'Yes'})
sns.factorplot('Survivor',data=titanic_df,palette='Set1',kind='count')
#survival of class vs person
sns.factorplot('Pclass','Survived',hue='person',data=titanic_df)
sns.lmplot('Age','Survived',data=titanic_df)
fig=sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='summer_d')
fig.add_legend()
generations=[10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)
sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)
#plot of if a person is alone vs with family 
sns.lmplot('Age','Survived',hue='Alone',data=titanic_df,palette='winter',x_bins=generations)
