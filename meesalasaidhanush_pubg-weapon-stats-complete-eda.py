# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv(r'/kaggle/input/pubg-weapon-stats/pubg-weapon-stats.csv')
df.head()
df.shape
df.isnull().any()
## null values are related to other melee weapons like crossbow,pan,sickle..etc. so we will drop it
df.dropna(axis=0,inplace=True)
df.shape
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
## weapon type
plt.figure(figsize=(15,10))
sns.countplot(df['Weapon Type'])
plt.show()
## bullet type
plt.figure(figsize=(12,7))
sns.countplot(df['Bullet Type'])
plt.show()
## magaizne capacity
plt.figure(figsize = (10,10))
sns.countplot (df['Magazine Capacity'], palette="inferno")
plt.show()
dum=df.groupby('Weapon Type')['Damage'].mean().reset_index()
dum.head()
sns.barplot(y=df['Weapon Type'],x=df['Damage'],data=dum)
most_damaging_weapon = df[df["Weapon Type"] == "Assault Rifle"][["Damage", "Weapon Name"]].sort_values("Damage", ascending  = False)
most_damaging_weapon
sns.barplot(x=most_damaging_weapon['Damage'],y=most_damaging_weapon['Weapon Name'])
## range of the weapon
plt.figure(figsize=(10,10))
sns.countplot(df['Range'])
plt.show()
bs=df.groupby('Weapon Type')['Bullet Speed'].mean().reset_index()
bs.head()
## inspecting the bullet speed
plt.figure(figsize=(13,10))
sns.barplot(x=df['Weapon Type'],y=df['Bullet Speed'],data=bs)
plt.show()
## inspecting shots on chest to kill
plt.figure(figsize=(13,7))
sns.barplot(y=df['Shots to Kill (Chest)'],x=df['Weapon Type'],data=df)
plt.show()
most_damaging_weapon_sniper = df[df["Weapon Type"] == "Sniper Rifle"][["Damage", "Weapon Name"]].sort_values("Damage", ascending  = False)
most_damaging_weapon_sniper
sns.barplot(x=most_damaging_weapon_sniper['Damage'],y=most_damaging_weapon_sniper['Weapon Name'],data=df)
## inspecting shots on head to kill
plt.figure(figsize=(13,7))
sns.barplot(y=df['Shots to Kill (Head)'],x=df['Weapon Type'],data=df,palette="rainbow")
plt.show()
