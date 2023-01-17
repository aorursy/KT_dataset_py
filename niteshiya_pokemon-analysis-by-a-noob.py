import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pokemon=pd.read_csv('../input/pokemon.csv')
pokemon.drop_duplicates('#',keep='first',inplace=True)
pokemon = pokemon.replace(np.nan, 'None', regex=True)
pokemon.head()
pokemon.describe()
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(pokemon['Generation'])
pokemon.Generation.value_counts()
(pokemon['Generation'].value_counts().sort_index()).plot.bar()
plt.show()
color=['gold','silver']
leg=pokemon[pokemon['Legendary']==True]
nonleg=pokemon[pokemon['Legendary']==False]
plt.pie([leg["Name"].count(),nonleg['Name'].count()],colors=color,labels=['Legendary','Non Legendary'])
plt.show()
sns.catplot(
y='Type 1',
data=pokemon,
kind='count',
)
sns.catplot(
y='Type 2',
data=pokemon,
kind='count',
)
sns.heatmap(
    pokemon.groupby(['Type 1', 'Type 2']).size().unstack(),
    linewidths=1,
    annot=True,
)

sns.heatmap(
    pokemon[['Attack','Defense','HP','Sp. Atk','Sp. Def','Speed','Generation','Legendary']].corr(),
    linewidths=1,
    annot=True,
)

