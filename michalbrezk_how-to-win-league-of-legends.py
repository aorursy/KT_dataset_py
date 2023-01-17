import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

pd.options.display.max_columns = 500

plt.rcParams.update({'figure.figsize': (12, 5), 'figure.dpi': 100})

sns.set(style="whitegrid")
df = pd.read_csv('/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
df.info()
df.head()
df.isnull().sum() > 0
df.drop('gameId', axis=1, inplace=True)
plt.figure(figsize=(5,3))

sns.countplot(x=df.blueWins, palette=sns.color_palette(['r','b']))

plt.xticks([0,1], ['Red', 'Blue'])

plt.xlabel('')

plt.ylabel('')
# dropping all the columns starting with "red"

df = df.loc[:,~df.columns.str.startswith('red')]
plt.figure(figsize=(15,15))

corrmat = df.corr()

corrmat = np.tril(corrmat)

corrmat[corrmat==0] = None

corrmat = corrmat.round(1)

labels = df.select_dtypes(include='number').columns.values

f, ax = plt.subplots(figsize=(15, 8))

sns.heatmap(corrmat, annot=True, vmax=0.8,vmin=-0.8, cmap='seismic_r', xticklabels=labels,yticklabels=labels, cbar=False)

plt.legend('')



plt.show()
g = sns.PairGrid(data = df, vars=['blueWardsPlaced', 'blueKills', 'blueDeaths', 'blueAssists', 'blueAvgLevel', 'blueTotalExperience'], hue='blueWins', palette=sns.color_palette(['r', 'b']), hue_kws={"marker": ["D", "o"], "alpha": [0.3, 0.3]})

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)

g.add_legend()

plt.show()
plt.figure(figsize=(15,6))

dfw = df.corr()['blueWins'].drop(['blueWins', 'blueDeaths'])

dfw = dfw.sort_values(ascending=False)



pal = sns.color_palette("Greens_d", len(dfw))

rank = dfw.argsort().argsort() 



sns.barplot(y=dfw.index, x=dfw, palette=np.array(pal[::-1])[rank])



plt.show()

dfw.apply(lambda x: round(round(20/dfw.max()*x)/2, 1))