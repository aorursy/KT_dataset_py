import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import datetime as dt



import sklearn

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree

from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

from math import isnan
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv(r"/kaggle/input/cricket/Cricket.csv",encoding='latin1')

df.head()
df.shape
df_dub = df.copy()

# Checking for duplicates and dropping the entire duplicate row if any

df_dub.drop_duplicates(subset=None, inplace=True)
df_dub.shape
df.shape
df.info()
df.describe()
(df.isnull().sum() * 100 / len(df)).value_counts(ascending=False)
df.isnull().sum().value_counts(ascending=False)
(df.isnull().sum(axis=1) * 100 / len(df)).value_counts(ascending=False)
df.isnull().sum(axis=1).value_counts(ascending=False)
df.head()
df[['Strt','End']] = df.Span.str.split("-",expand=True) 
df[['Strt','End']]=df[['Strt','End']].astype(int)

df['Exp']=df['End']-df['Strt']

df=df.drop(['Strt','End','Span'], axis = 1) 

df.head()
#Match Played

plt.figure(figsize = (30,5))

mat = df[['Player','Mat']].sort_values('Mat', ascending = False)

ax = sns.barplot(x='Player', y='Mat', data= mat)

ax.set(xlabel = '', ylabel= 'Match Played')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

mat_top10 = df[['Player','Mat']].sort_values('Mat', ascending = False).head(10)

ax = sns.barplot(x='Player', y='Mat', data= mat_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Match Played')

plt.xticks(rotation=90)

plt.show()
#Inns

plt.figure(figsize = (30,5))

inns = df[['Player','Inns']].sort_values('Inns', ascending = False)

ax = sns.barplot(x='Player', y='Inns', data= inns)

ax.set(xlabel = '', ylabel= 'Innings Played')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

inns_top10 = df[['Player','Inns']].sort_values('Inns', ascending = False).head(10)

ax = sns.barplot(x='Player', y='Inns', data= inns_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Innings Played')

plt.xticks(rotation=90)

plt.show()
#NO

plt.figure(figsize = (30,5))

no = df[['Player','NO']].sort_values('NO', ascending = False)

ax = sns.barplot(x='Player', y='NO', data= no)

ax.set(xlabel = '', ylabel= 'Not Out')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

inns_top10 = df[['Player','NO']].sort_values('NO', ascending = False).head(10)

ax = sns.barplot(x='Player', y='NO', data= inns_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Not Out')

plt.xticks(rotation=90)

plt.show()
#Runs

plt.figure(figsize = (30,5))

run = df[['Player','Runs']].sort_values('Runs', ascending = False)

ax = sns.barplot(x='Player', y='Runs', data= run)

ax.set(xlabel = '', ylabel= 'Runs Scored')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

runs_top10 = df[['Player','Runs']].sort_values('Runs', ascending = False).head(10)

ax = sns.barplot(x='Player', y='Runs', data= runs_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Runs Scored')

plt.xticks(rotation=90)

plt.show()
#HS

df.HS=df.HS.str.extract('(\d+)')

df.HS=df.HS.astype(int)

plt.figure(figsize = (30,5))

hs = df[['Player','HS']].sort_values('HS', ascending = False)

ax = sns.barplot(x='Player', y='HS', data= hs)

ax.set(xlabel = '', ylabel= 'Highest Score')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

hs_top10 = df[['Player','HS']].sort_values('HS', ascending = False).head(10)

ax = sns.barplot(x='Player', y='HS', data= hs_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Highest Score')

plt.xticks(rotation=90)

plt.show()
#Ave

plt.figure(figsize = (30,5))

ave = df[['Player','Ave']].sort_values('Ave', ascending = False)

ax = sns.barplot(x='Player', y='Ave', data= ave)

ax.set(xlabel = '', ylabel= 'Averages')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

avg_top10 = df[['Player','Ave']].sort_values('Ave', ascending = False).head(10)

ax = sns.barplot(x='Player', y='Ave', data= avg_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Averages')

plt.xticks(rotation=90)

plt.show()
#BF

plt.figure(figsize = (30,5))

bf = df[['Player','BF']].sort_values('BF', ascending = False)

ax = sns.barplot(x='Player', y='BF', data= bf)

ax.set(xlabel = '', ylabel= 'Best Form')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

bf_top10 = df[['Player','BF']].sort_values('BF', ascending = False).head(10)

ax = sns.barplot(x='Player', y='BF', data= bf_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Best Form')

plt.xticks(rotation=90)

plt.show()
#SR 

plt.figure(figsize = (30,5))

sr = df[['Player','SR']].sort_values('SR', ascending = False)

ax = sns.barplot(x='Player', y='SR', data= sr)

ax.set(xlabel = '', ylabel= 'SR')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

sr_top10 = df[['Player','SR']].sort_values('SR', ascending = False).head(10)

ax = sns.barplot(x='Player', y='SR', data= sr_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'SR')

plt.xticks(rotation=90)

plt.show()
#100

plt.figure(figsize = (30,5))

r100 = df[['Player','100']].sort_values('100', ascending = False)

ax = sns.barplot(x='Player', y='100', data= r100)

ax.set(xlabel = '', ylabel= "100's Scored" )

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

r100_top10 = df[['Player','100']].sort_values('100', ascending = False).head(10)

ax = sns.barplot(x='Player', y='100', data= r100_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= "100's Scored")

plt.xticks(rotation=90)

plt.show()
#50

plt.figure(figsize = (30,5))

r50 = df[['Player','50']].sort_values('50', ascending = False)

ax = sns.barplot(x='Player', y='50', data= r50)

ax.set(xlabel = '', ylabel= "50s Scored")

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

r50_top10 = df[['Player','50']].sort_values('50', ascending = False).head(10)

ax = sns.barplot(x='Player', y='50', data= r50_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= "50's Scored")

plt.xticks(rotation=90)

plt.show()
#0

plt.figure(figsize = (30,5))

r0 = df[['Player','0']].sort_values('0', ascending = False)

ax = sns.barplot(x='Player', y='0', data= r0)

ax.set(xlabel = '', ylabel= "Os Scored")

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

r0_top10 = df[['Player','0']].sort_values('0', ascending = False).head(10)

ax = sns.barplot(x='Player', y='0', data= r0_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= "Os Scored")

plt.xticks(rotation=90)

plt.show()
#Exp

plt.figure(figsize = (30,5))

exp = df[['Player','Exp']].sort_values('Exp', ascending = False)

ax = sns.barplot(x='Player', y='Exp', data= exp)

ax.set(xlabel = '', ylabel= 'Experience')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize = (10,5))

exp_top10 = df[['Player','Exp']].sort_values('Exp', ascending = False).head(10)

ax = sns.barplot(x='Player', y='Exp', data= exp_top10)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

ax.set(xlabel = '', ylabel= 'Exprience')

plt.xticks(rotation=90)

plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (10, 10))

sns.heatmap(df.corr(), annot = True, cmap="rainbow")

plt.savefig('Correlation')

plt.show()
sns.pairplot(df,corner=True,diag_kind="kde")

plt.show()
df.describe()
f, axes = plt.subplots(4,3, figsize=(16, 8))

s=sns.violinplot(y=df.Exp,ax=axes[0, 0])

axes[0, 0].set_title('Exp')

s=sns.violinplot(y=df.Mat,ax=axes[0, 1])

axes[0, 1].set_title('Mat')

s=sns.violinplot(y=df.Inns,ax=axes[0, 2])

axes[0, 2].set_title('Inns')



s=sns.violinplot(y=df.NO,ax=axes[1, 0])

axes[1, 0].set_title('NO')

s=sns.violinplot(y=df.Runs,ax=axes[1, 1])

axes[1, 1].set_title('Runs')

s=sns.violinplot(y=df.HS,ax=axes[1, 2])

axes[1, 2].set_title('HS')



s=sns.violinplot(y=df.Ave,ax=axes[2, 0])

axes[2, 0].set_title('Ave')

s=sns.violinplot(y=df.SR,ax=axes[2, 1])

axes[2, 1].set_title('SR')

s=sns.violinplot(y=df['100'],ax=axes[2, 2])

axes[2, 2].set_title('100')

s=sns.violinplot(y=df.BF,ax=axes[3, 0])

axes[3, 0].set_title('BF')

s=sns.violinplot(y=df['50'],ax=axes[3, 1])

axes[3, 1].set_title('50s')

s=sns.violinplot(y=df['0'],ax=axes[3, 2])

axes[3, 2].set_title('0s')

plt.show()
plt.figure(figsize = (30,10))

features=[ 'Mat', 'Inns', 'NO', 'Runs', 'HS', 'Ave', 'BF', 'SR', '100','50', '0', 'Exp']

for i in enumerate(features):

    plt.subplot(3,4,i[0]+1)

    sns.distplot(df[i[1]])
Q3 = df.Mat.quantile(0.99)

Q1 = df.Mat.quantile(0.01)

df['Mat'][df['Mat']<=Q1]=Q1

df['Mat'][df['Mat']>=Q3]=Q3
Q3 = df.Inns.quantile(0.99)

Q1 = df.Inns.quantile(0.01)

df['Inns'][df['Inns']<=Q1]=Q1

df['Inns'][df['Inns']>=Q3]=Q3
Q3 = df.NO.quantile(0.99)

Q1 = df.NO.quantile(0.01)

df['NO'][df['NO']<=Q1]=Q1

df['NO'][df['NO']>=Q3]=Q3
Q3 = df.Runs.quantile(0.99)

Q1 = df.Runs.quantile(0.01)

df['Runs'][df['Runs']<=Q1]=Q1

df['Runs'][df['Runs']>=Q3]=Q3
Q3 = df.HS.quantile(0.99)

Q1 = df.HS.quantile(0.01)

df['HS'][df['HS']<=Q1]=Q1

df['HS'][df['HS']>=Q3]=Q3
Q3 = df.Ave.quantile(0.99)

Q1 = df.Ave.quantile(0.01)

df['Ave'][df['Ave']<=Q1]=Q1

df['Ave'][df['Ave']>=Q3]=Q3
Q3 = df.BF.quantile(0.99)

Q1 = df.BF.quantile(0.01)

df['BF'][df['BF']<=Q1]=Q1

df['BF'][df['BF']>=Q3]=Q3
Q3 = df.SR.quantile(0.99)

Q1 = df.SR.quantile(0.01)

df['SR'][df['SR']<=Q1]=Q1

df['SR'][df['SR']>=Q3]=Q3
Q3 = df.Exp.quantile(0.99)

Q1 = df.Exp.quantile(0.01)

df['Exp'][df['Exp']<=Q1]=Q1

df['Exp'][df['Exp']>=Q3]=Q3
Q3 = df['100'].quantile(0.99)

Q1 = df['100'].quantile(0.01)

df['100'][df['100']<=Q1]=Q1

df['100'][df['100']>=Q3]=Q3
Q3 = df['50'].quantile(0.99)

Q1 = df['50'].quantile(0.01)

df['50'][df['50']<=Q1]=Q1

df['50'][df['50']>=Q3]=Q3
Q3 = df['0'].quantile(0.99)

Q1 = df['0'].quantile(0.01)

df['0'][df['0']<=Q1]=Q1

df['0'][df['0']>=Q3]=Q3
f, axes = plt.subplots(4,3, figsize=(16, 8))

s=sns.violinplot(y=df.Exp,ax=axes[0, 0])

axes[0, 0].set_title('Exp')

s=sns.violinplot(y=df.Mat,ax=axes[0, 1])

axes[0, 1].set_title('Mat')

s=sns.violinplot(y=df.Inns,ax=axes[0, 2])

axes[0, 2].set_title('Inns')



s=sns.violinplot(y=df.NO,ax=axes[1, 0])

axes[1, 0].set_title('NO')

s=sns.violinplot(y=df.Runs,ax=axes[1, 1])

axes[1, 1].set_title('Runs')

s=sns.violinplot(y=df.HS,ax=axes[1, 2])

axes[1, 2].set_title('HS')



s=sns.violinplot(y=df.Ave,ax=axes[2, 0])

axes[2, 0].set_title('Ave')

s=sns.violinplot(y=df.SR,ax=axes[2, 1])

axes[2, 1].set_title('SR')

s=sns.violinplot(y=df['100'],ax=axes[2, 2])

axes[2, 2].set_title('100')

s=sns.violinplot(y=df.BF,ax=axes[3, 0])

axes[3, 0].set_title('BF')

s=sns.violinplot(y=df['50'],ax=axes[3, 1])

axes[3, 1].set_title('50s')

s=sns.violinplot(y=df['0'],ax=axes[3, 2])

axes[3, 2].set_title('0s')

plt.show()
# Dropping Player field as final dataframe will only contain data columns



df_drop = df.copy()

player = df_drop.pop('Player')
df_drop.head()