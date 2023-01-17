import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../input/Pokemon.csv')
df.head()
df.shape

df.info()
df.isnull().sum()
df.describe()
df.query('Total == 780')
df.query('HP == 255')
df.query('Attack == 190')
df.query('Defense == 230')
df.query('Speed == 180')
df['Type 1'].value_counts()
plt.figure(figsize = (10,5))
sns.set(style = 'white')
p = sns.countplot(x = 'Type 1', data = df, hue = 'Generation',palette = 'Paired')
_ = plt.setp(p.get_xticklabels(), rotation = 60)
_ = plt.xlabel('Types of Pokemon')
_ = plt.legend(loc = 6, bbox_to_anchor = (1, 0.5), title = 'Generation')
df['Generation'].value_counts()
plt.figure(figsize = (10,6))
sns.countplot(x = 'Generation',data = df, hue = 'Type 1',order = [1,2,3,4,5,6], palette = 'Paired')
_ = plt.legend(loc = 6, bbox_to_anchor = (1, 0.5), title = 'Types of Pokemon')
sns.pairplot(df,vars = ['HP', 'Attack','Defense', 'Speed'],diag_kind = 'kde')
sns.jointplot(x = 'Attack', y = 'Defense', data = df, kind = 'hex',color = 'c')
sns.jointplot(x = "Sp. Atk", y = 'Sp. Def', data = df, kind = 'reg', color = 'y')
sns.jointplot(x = 'HP', y = 'Total', data = df, kind = 'kde', color = '#FF1300')
df.Legendary.value_counts()
df_legendary = df[df['Legendary'] == True]
df_legendary.head()
df_legendary.reset_index(drop = True)
df_legendary.describe()
df_legendary.Generation.value_counts()
sns.countplot(x = 'Generation', data = df_legendary, palette = 'Paired')
df_legendary['Type 1'].value_counts()
p = sns.countplot(x = 'Type 1', data = df_legendary, palette = 'Paired')
_ = plt.xlabel('Types of Pokemon')
_ = plt.setp(p.get_xticklabels(), rotation = 60)
sns.set()
for i in ['Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']:
    plt.figure(i)
    sns.distplot(df[i], hist = False, rug = True)
# scroll down to view the distplots 
