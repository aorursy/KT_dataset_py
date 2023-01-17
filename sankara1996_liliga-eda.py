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
df1=pd.read_csv('../input/Laliga1.csv')
df1.head()
dfn=df1.replace('-',0)

dfn.tail()
import matplotlib.pyplot as plt

import seaborn as sns
sns.countplot('BestPosition',data=dfn)

plt.legend()
dfn

pd.set_option('display.max_rows',1000)

pd.set_option('display.max_column',1000)
dfn['Debuttem']=dfn['Debut'].apply(lambda x: str(x).split('-'))
def format_year(x):

    if len(x)>1:

        if int(x[0])<1999:

            return [x[0],int(x[1])+1900]

        else:

            return [x[0],int(x[1])+2000]

    else:

        return x
dfn['Debut_temp']=dfn['Debuttem'].apply(format_year)
dfn.head()
def check_debut(x):

    if len(x)==1:

        return int(x[0])>=1930 and int(x[0])<=1980

    else:

        return ((int(x[0])>=1930 and int(x[0])<=1980)or(int(x[1])>=1930 and int(x[1])<=1980))
a=dfn[dfn['Debut_temp'].apply(check_debut)]['Team']
print(a)

print('No.of teams that debut b/w 1930-1980 is',a.count())
cols=['Pos', 'Seasons', 'Points', 'GamesPlayed', 'GamesWon',

       'GamesDrawn', 'GamesLost', 'GoalsFor', 'GoalsAgainst', 'Champion',

       'Runner-up', 'Third', 'Fourth', 'Fifth', 'Sixth', 'T','BestPosition']
dfn[cols]=dfn[cols].apply(pd.to_numeric)
dfn.info()
dfn['rank']=dfn['Points'].rank(method='dense',ascending=False)
dfn[['Team','rank']].head()
dfn['Winning %']=(dfn['GamesWon']/dfn['GamesPlayed'])*100
dfn[['Team','Winning %']].head()
dfn.info()
dfn['Winning %']=dfn['Winning %'].fillna(0)
sns.scatterplot('BestPosition','Winning %',data=dfn)
dfn['Goaldif']=(dfn['GoalsFor']-dfn['GoalsAgainst'])
dfn[['Team','Goaldif']].head()
import statsmodels.api as sm

import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LinearRegression

from statsmodels.stats.anova import anova_lm
z=smf.ols(formula='BestPosition ~ Goaldif+GamesPlayed',data=dfn).fit()
z.summary()