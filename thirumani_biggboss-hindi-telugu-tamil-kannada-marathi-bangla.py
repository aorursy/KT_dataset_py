import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 50)



import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
bigg_boss = pd.read_csv('/kaggle/input/Bigg_Boss_India.csv', encoding = "ISO-8859-1")

nRow, nCol = bigg_boss.shape

print(f'There are {nRow} rows and {nCol} columns')
bigg_boss.head(5)
bigg_boss.tail(10).T
bigg_boss.sample(10)
bigg_boss.info()
bigg_boss.describe()
# Unique values in each column

for col in bigg_boss.columns:

    print("Number of unique values in", col,"-", bigg_boss[col].nunique())
# Number of seasons in all Indian languages

print(bigg_boss.groupby('Language')['Season Number'].nunique().sum())



# 35 seasons happened (including current seasons)
# Number of seasons in each Indian language

print(bigg_boss.groupby('Language')['Season Number'].nunique().nlargest(10))
# Total number of Bigg Boss housemates

fig = plt.figure(figsize=(10,4))

ax = sns.countplot(x='Language', data=bigg_boss)

ax.set_title('Bigg Boss Series - Indian Language')

for t in ax.patches:

    if (np.isnan(float(t.get_height()))):

        ax.annotate(0, (t.get_x(), 0))

    else:

        ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))
# Number of normal entries and wild card entries

print(bigg_boss['Wild Card'].value_counts(), "\n")

print(round(bigg_boss['Wild Card'].value_counts(normalize=True)*100))

sns.countplot(x='Wild Card', data=bigg_boss)
# Common people has many professions, so clubbing them into one category

bigg_boss.loc[bigg_boss['Profession'].str.contains('Commoner'),'Profession']='Commoner'
# Participant's Profession

print(bigg_boss['Profession'].value_counts())

fig = plt.figure(figsize=(25,8))

sns.countplot(x='Profession', data=bigg_boss)

plt.xticks(rotation=90)
# Broadcastor

fig = plt.figure(figsize=(20,5))

ax = sns.countplot(x='Broadcasted By', data=bigg_boss, palette='RdBu')

ax.set_title('Bigg Boss Series - Indian Broadcastor & Total Number of Housemates')

for t in ax.patches:

    if (np.isnan(float(t.get_height()))):

        ax.annotate(0, (t.get_x(), 0))

    else:

        ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))
bigg_boss.groupby('Host Name')['Season Number'].nunique().nlargest(25)
# Housemate's Gender

print(bigg_boss['Gender'].value_counts())
# Maximum TRP of Bigg Boss Hindi/India seasons

print("Maximum TRP",bigg_boss['Average TRP'].max(), "\n")

print(bigg_boss.loc[bigg_boss['Average TRP']==bigg_boss['Average TRP'].max()][["Language","Season Number"]].head(1).to_string(index=False))
# Longest season of Bigg Boss Hindi/India seasons

print("Longest season",bigg_boss['Season Length'].max(), "days \n")

print(bigg_boss.loc[bigg_boss['Season Length']==bigg_boss['Season Length'].max()][["Language","Season Number"]].head(1).to_string(index=False))
# All BB Winners

bigg_boss.loc[bigg_boss.Winner==1]
# Profession of BB Season Winners

bigg_boss.loc[bigg_boss.Winner==1,'Profession'].value_counts()
# Gender of Season title Winners

print(bigg_boss.loc[bigg_boss.Winner==1,'Gender'].value_counts(),'\n')



# In percentage

print(round(bigg_boss.loc[bigg_boss.Winner==1,'Gender'].value_counts(normalize=True)*100))
# Entry type of the Season Winners

bigg_boss.loc[bigg_boss.Winner==1,'Wild Card'].value_counts()
# No re-entered contestant won Bigg Boss title

bigg_boss.loc[bigg_boss.Winner==1,'Number of re-entries'].value_counts()
# Number of eliminations or evictions faced by the Bigg Boss competition winners

bigg_boss.loc[bigg_boss.Winner==1,'Number of Evictions Faced'].value_counts().sort_index()



# Number of eliminations faced - Number of Winners
# Bigg Boss winners Number of times elected as Captain

bigg_boss.loc[bigg_boss.Winner==1,'Number of times elected as Captain'].value_counts().sort_index()



# Number of times elected as Captain   - Number of winners
import pandas_profiling

pandas_profiling.ProfileReport(bigg_boss)