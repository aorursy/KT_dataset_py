# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re #regular expression

from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv', index_col = 'Unnamed: 0')
data.info()
data.dtypes
data.head()
df=data.copy()
df.rename(columns = {'Club Logo':'Logo'},inplace=True)

df.rename(columns = {'International Reputation':'Rep'},inplace=True)

df.rename(columns = {'Preferred Foot':'PrefFoot'},inplace=True)

df.rename(columns = {'Weak Foot':'WeakFoot'},inplace=True)

df.rename(columns = {'Skill Moves':'SkillMoves'},inplace=True)

df.rename(columns = {'Work Rate':'WorkRate'},inplace=True)

df.rename(columns = {'Body Type':'BodyType'},inplace=True)

df.rename(columns = {'Real Face':'RealFace'},inplace=True)

df.rename(columns = {'Jersey Number':'JerseyNumber'},inplace=True)

df.rename(columns = {'Loaned From':'LoanedFrom'},inplace=True)

df.rename(columns = {'Contract Valid Until':'Contract'},inplace=True)

df.rename(columns = {'Release Clause':'ReleaseClause'},inplace=True)

df.head()
def money(x):

    if type(x) is float:

        return x    

    if 'K' in [x]:

        y = 1000        

    else:

        y = 1000000

        x = x.replace('M','').replace('K','').replace('€','')

        return float(x) * y
df['Wage'] = df['Wage'].apply(money)

df['Value'] = df['Value'].apply(money)

df['ReleaseClause'] = df['ReleaseClause'].apply(money)

df.head()
df.columns
df.shape
position_count = df['Position'].value_counts(dropna=False)

position_count
count = Counter(df['Position'])

list_of_positions = []

for i in count:

    list_of_positions.append([i , count[i]])

positions = pd.DataFrame(list_of_positions,columns=('Position','No')).sort_values(['No'],ascending=False)

plt.figure(figsize=(15,10))

sns.barplot(x=positions['Position'],y=positions['No'])
df.describe()
df.corr()
df.boxplot(column='Overall',by='Rep')
df.boxplot('Potential',by='Rep')
rep5 = df.Rep ==5

rep4 = df.Rep ==4

rep3 = df.Rep ==3

rep2 = df.Rep ==2

rep1 = df.Rep ==1

r5=df[rep5]

r4=df[rep4]

r3=df[rep3]

r2=df[rep2]

r1=df[rep1]
a = r2[(r2['Potential'] > 90) & (r2['Age'] <= 20)]

b = r1[(r1['Potential'] > 90) & (r1['Age'] <= 20)]

wonderkid = pd.concat([a,b],axis = 0)

wonderkid
filter1 = df['Potential'] > 85

filter2 = df['Value'] < 5000000

filter3 = df['Age'] < 20

transfer_list = df[filter1 & filter2 & filter3]

tl = transfer_list.set_index(['Position','Nationality']).sort_values(['Position','Value'])

tl
transfer_list = transfer_list.sort_values(['Value'],ascending=False)

plt.figure(figsize=(20,10))

sns.barplot(x='Name',y='Value',data=transfer_list)

plt.xticks(rotation= 45)

plt.show()
df.head()
strikers.columns
#strikers.Special.value_counts()
st = df['Position'] == 'ST' 

rf = df['Position'] == 'RF'

lf = df['Position'] == 'LF'

strikers = df[st | rf | lf]
