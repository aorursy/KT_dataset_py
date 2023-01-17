import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes = True)

%matplotlib inline

import os

print(os.listdir("../input"))
battle_dataset = pd.read_csv('../input/battles.csv')
char_death = pd.read_csv("../input/character-deaths.csv")
char_pred = pd.read_csv("../input/character-predictions.csv")
battle_dataset
battle_dataset.info()
battle_dataset.isnull().sum()
battle_dataset['attacker_size'].mean()
battle_dataset['defender_size'].mean()
new_battle = battle_dataset[['defender_size','attacker_size','attacker_outcome']].dropna()
new_battle.reset_index(inplace=True)
new_battle = new_battle.iloc[:,1:]
new_battle
sns.pairplot(new_battle, hue='attacker_outcome')
battle_dataset[battle_dataset['attacker_outcome'] == 'loss']
battle_dataset.info()
sns.countplot(battle_dataset['year'])
battle_dataset.groupby('year')['attacker_outcome'].value_counts().plot(kind = 'bar')
battle_dataset.groupby('attacker_outcome')['year'].value_counts().plot(kind = 'bar')
plt.figure(figsize=(12,6))

sns.countplot(battle_dataset['attacker_king'])
plt.figure(figsize=(15,6))

sns.countplot(battle_dataset['defender_king'])
battle_dataset.groupby('attacker_king')['attacker_outcome'].value_counts().plot(kind = 'bar')
battle_dataset.groupby('attacker_king')['attacker_outcome'].value_counts()
battle_dataset['attacker_king'].value_counts()
battle_dataset.groupby('attacker_king')['defender_king'].value_counts().plot(kind = 'bar')
battle_dataset.groupby(['attacker_king','defender_king'])['attacker_outcome'].value_counts().plot(kind = 'bar')
battle_dataset.groupby(['attacker_king','defender_king'])['attacker_outcome'].value_counts()
battle_dataset['battle_type'].value_counts().plot(kind = 'bar')
battle_dataset.groupby('battle_type')['attacker_outcome'].value_counts().plot(kind = 'bar')
battle_dataset.groupby(['attacker_king','battle_type'])['attacker_outcome'].value_counts()
battle_dataset.groupby('summer')['attacker_outcome'].value_counts()
battle_dataset['region'].value_counts().plot(kind = 'bar')
battle_dataset['location'].value_counts().plot(kind = 'bar')
battle_dataset.groupby('attacker_outcome')['major_capture'].sum().plot(kind = 'bar')
battle_dataset.groupby('attacker_outcome')['major_death'].sum().plot(kind = 'bar')
battle_dataset.groupby('attacker_outcome')['major_death'].sum()
char_death.head()
char_death.info()
char_death.isnull().sum()
battle_dataset.groupby('year')[['major_death','major_capture']].sum().plot(kind = 'bar')
battle_dataset.groupby('attacker_outcome')[['major_death','major_capture']].sum().plot(kind = 'bar')
battle_dataset.groupby(['attacker_king','year'])[['attacker_1','attacker_2','attacker_3','attacker_4']].count()
a = battle_dataset.groupby(['defender_king','year'])[['defender_1','defender_2','defender_3','defender_4']].count()
a.index
sns.heatmap(a)
sns.heatmap(battle_dataset.groupby(['attacker_king','year'])[['attacker_1','attacker_2','attacker_3','attacker_4']].count())
char_death.head()
### Use below code and add similar commands in lambda to replace duplicate Allegiances with House name

char_death['Allegiances'] = char_death['Allegiances'].apply(lambda x : 'House Martell' if(x == 'Martell') else 'House Stark' if(x=='Stark') else 'House Targaryen' if(x=='Targaryen') else 'House Tully' if(x=='Tully') else 'House Tyrell' if(x=='Tyrell') else x)
char_death['Gender'].value_counts().plot(kind = 'bar')
char_death['Allegiances'].value_counts().plot(kind = 'bar')
char_death[char_death['Death Year'].notnull()]['Allegiances'].value_counts().plot(kind = 'bar')
plt.figure(figsize=(12,5))

char_death[char_death['Death Year'].notnull()].groupby('Death Year')['Allegiances'].value_counts().plot(kind = 'bar')
char_death[char_death['Death Year'].notnull()]['Death Year'].value_counts()
char_death.groupby('Gender')['Allegiances'].value_counts().plot(kind = 'bar')
char_death.groupby('Allegiances')['Gender'].value_counts().plot(kind = 'bar')