import pandas as pd

import seaborn as sns

import matplotlib.pyplot as pl
poke = pd.read_csv('../input/pokemonGO.csv')
poke.head()
poke = poke.drop(['Image URL','Pokemon No.'],1)
sns.jointplot(x="Max CP", y="Max HP", data=poke);
sns.boxplot(data=poke);
sns.swarmplot(x="Max CP", y="Max HP", data=poke);
sns.barplot(x="Max CP", y="Max HP", data=poke);
sns.pointplot(x="Max CP", y="Max HP", data=poke);