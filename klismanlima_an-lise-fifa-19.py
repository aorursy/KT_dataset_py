# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importando o dataset para um dataframe

df = pd.read_csv("/kaggle/input/fifa19/data.csv")
#Início da análise do Dataframe

df.head()
#Mostrando todas as colunas

pd.set_option('display.max_columns', 500)

df.head()
df.describe()
fig= plt.figure(figsize=(15,5))

sns.distplot(df['Age'])

plt.xlabel("Distribuição de jogadores por idade")
df.nlargest(5, 'Age')
fig= plt.figure(figsize=(18,5))

jogadores_pais = df['Nationality'].value_counts().reset_index()

jogadores_pais

sns.barplot(x="index", y="Nationality", data=jogadores_pais.head(15))
top5jog = df.nlargest(5, 'Overall')

top5jog[['Name', 'Nationality', 'Overall', 'Club']]
messi = top5jog.loc[[0]]



labels=np.array(['Agility', 'ShotPower', 'Dribbling', 'Stamina', 'Marking'])

stats=messi.loc[0,labels].values



#Configurações

angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

stats=np.concatenate((stats,[stats[0]]))

angles=np.concatenate((angles,[angles[0]]))

#Tamanho plot

fig=plt.figure(figsize=(8,8))

ax = fig.add_subplot(111, polar=True)   # Set polar axis

ax.plot(angles, stats, 'o-', linewidth=2) 

ax.fill(angles, stats, alpha=0.25) 

ax.set_thetagrids(angles * 180/np.pi, labels) 

ax.set_title(messi.loc[0,"Name"], fontsize=20) 



ax.grid(True)
ronaldo = top5jog.loc[[1]]



labels=np.array(['Agility', 'ShotPower', 'Dribbling', 'Stamina', 'Marking'])

stats=ronaldo.loc[1,labels].values



#Configurações

angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

stats=np.concatenate((stats,[stats[0]]))

angles=np.concatenate((angles,[angles[0]]))

#Tamanho plot

fig=plt.figure(figsize=(8,8))

ax = fig.add_subplot(111, polar=True)   # Set polar axis

ax.plot(angles, stats, 'o-', linewidth=2) 

ax.fill(angles, stats, alpha=0.25) 

ax.set_thetagrids(angles * 180/np.pi, labels) 

ax.set_title(ronaldo.loc[1,"Name"], fontsize=20) 



ax.grid(True)

neymar = top5jog.loc[[2]]



labels=np.array(['Agility', 'ShotPower', 'Dribbling', 'Stamina', 'Marking'])

stats=neymar.loc[2,labels].values



#Configurações

angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

stats=np.concatenate((stats,[stats[0]]))

angles=np.concatenate((angles,[angles[0]]))

#Tamanho plot

fig=plt.figure(figsize=(8,8))

ax = fig.add_subplot(111, polar=True)   # Set polar axis

ax.plot(angles, stats, 'o-', linewidth=2) 

ax.fill(angles, stats, alpha=0.25) 

ax.set_thetagrids(angles * 180/np.pi, labels) 

ax.set_title(neymar.loc[2,"Name"], fontsize=20) 



ax.grid(True)
