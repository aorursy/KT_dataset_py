

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series,DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
titanic_df=pd.read_csv('../input/train.csv')
titanic_df.head()
titanic_df.info()
titanic_df.isnull().sum()
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
sns.countplot(x='Sex', data=titanic_df)
sns.countplot(x='Pclass', data=titanic_df, hue='Sex')
def male_female_child(passenger):
    age,sex = passenger
    
    if age<10:
       return 'child'
    else :
       return sex
titanic_df['person']=titanic_df[['Age','Sex']].apply(male_female_child,axis=1)
titanic_df.head(10)
sns.countplot(x='Pclass', data=titanic_df, hue='person')
titanic_df['Age'].hist(bins=40)
titanic_df['Age'].mean()
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
fig=sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest=titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
deck=titanic_df['Cabin'].dropna()
deck.head()
levels=[]

for level in deck:
    levels.append(level[0])
    
cabin_df=DataFrame(levels)
cabin_df.columns=['Cabin']

sns.countplot('Cabin',data=cabin_df,palette='summer')
sns.countplot('Embarked',data=titanic_df,hue='Pclass')
#who was alone and who was with family
titanic_df['Alone']=titanic_df.SibSp + titanic_df.Parch
titanic_df['Alone'].loc[titanic_df['Alone']>0]='With family'
titanic_df['Alone'].loc[titanic_df['Alone']==0]='Alone'
titanic_df.head()
sns.countplot('Alone',data=titanic_df,palette='Blues')
titanic_df['Survivor']=titanic_df.Survived.map({0:'no',1:'yes'})
sns.countplot('Survivor', data=titanic_df, palette='Set1')
sns.factorplot('Pclass','Survived',data=titanic_df)
sns.factorplot('Pclass','Survived',hue='person',data=titanic_df,palette='Set2')
sns.lmplot('Age','Survived',data=titanic_df)
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df)
generations=[10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter', x_bins=generations)
sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter', x_bins=generations)
sns.countplot('Survivor',hue='Alone', data=titanic_df, palette='summer')
sns.factorplot('Sex','Survived',hue='Alone', data=titanic_df, palette='winter')
sns.countplot('Survivor',hue='Embarked', data=titanic_df, palette='Blues')
titanic_ndf = titanic_df.dropna()

def Deck(Cabin):

        if Cabin[0] == 'A':

          return 'A'

        elif Cabin[0] == 'B':

         return 'B'

        elif Cabin[0] == 'C':

         return 'C'

        elif Cabin[0] == 'D':

         return 'D'

        elif Cabin[0] == 'E':

         return 'E'

        elif Cabin[0] == 'F':

         return 'F'

        elif Cabin[0] == 'G':

         return 'G'

        else:

         return np.NaN

titanic_ndf['Deck'] = titanic_ndf['Cabin'].apply(Deck)
titanic_ndf.head()
sns.factorplot('Deck','Survived',hue='person',data=titanic_ndf,palette='Blues')
def Class(Fare):

        if Fare <=25:

          return 'E'

        elif Fare >25 and Fare<=50:

         return 'M'

        elif Fare > 50:

         return 'P'

        

titanic_df['Class'] = titanic_df['Fare'].apply(Class)
titanic_df.head(5)
sns.factorplot('Class','Survived',hue='person',data=titanic_df,palette='winter')
