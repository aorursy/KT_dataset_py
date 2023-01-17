import pandas as pd

import numpy as np

import urllib

from colorama import Fore, Back, Style

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.image as mpimg



y_ = Fore.YELLOW

r_ = Fore.RED

g_ = Fore.GREEN

b_ = Fore.BLUE

m_ = Fore.MAGENTA
df_full = pd.read_csv('../input/pokemon/Pokemon.csv')

df_full.drop(columns=['#'],axis=1, inplace=True)
df_full.head(8).T
df_full.describe().T
df_full.info()
cols = df_full.columns

for col in cols: 

    print(f'{y_}Unique values in    {r_}{col} : {r_}{df_full[col].nunique()}')
print(f'{y_}Shape of dataframe: {b_}{df_full.shape} {y_}and total null values: {b_}{df_full.isna().sum().sum()}')
import missingno as msno

msno.bar(df_full,(8,6),color='red')

plt.title('MISSING VALUES',fontsize=14)
df_full['Legendary_map'] = df_full["Legendary"].astype(int)

df = df_full.copy()

df.drop(columns = ['Type 2'], axis=1, inplace=True)
df_full.info()
cols = [ 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']

sns.pairplot(df,hue='Legendary_map',vars=cols,corner=True,plot_kws=dict(linewidth=0, alpha=1),height=4)
corr = df.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(10, 10))

    ax = sns.heatmap(corr,mask=mask,square=True,linewidths=.8,cmap="viridis",annot=True)
sns.set_style('dark')

plt.figure(figsize=(16,6))

sns.countplot(df_full['Type 1'],

             palette=['#7ec63c','#f0560f','#3ba7fa','#aab31f','#d5cec8','#9e58a0','#f8bc16','#dabe6a','#f9bef8','#944526','#ef4681','#c1a961','#6f72bd','#7ddbf7','#7059d8','#584537','#9ea0af','#8fa3ec'])
sns.set_style('dark')

plt.figure(figsize=(16,12))

plt.subplot(211)

sns.countplot(df_full['Type 1'], hue=df_full['Generation'],palette='hot',linewidth=0,alpha=1)

plt.subplot(212)

sns.countplot(df_full['Type 1'], hue= df_full['Legendary'], palette='hot_r', linewidth=0, alpha=1)
plt.figure(figsize=(12,6))

sns.distplot(df_full['Total'],color='red',hist_kws={'alpha':1,"linewidth": 4}, kde_kws={"color": "black", "lw": 2, "label": "KDE"})

plt.title('Distribution of Total', fontdict = {'size': 12})

# plt.xlabel('Percentage of correct answers', size = 12)
print(f"{y_}The Pokemons with highest TOTAL are \n{r_}{df.loc[df['Total'] == df_full['Total'].max()]['Name']}")
print(f"{y_}The Pokemon with lowest TOTAL is \n{b_}{df.loc[df['Total'] == df_full['Total'].min()]['Name']}")
cols = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']

for i,c in enumerate(cols): 



    sns.set_style('whitegrid')

    g = sns.FacetGrid(df_full, col="Generation", hue='Legendary_map', height=2.5, aspect=0.8,palette=['red','black'])

    g.set_axis_labels("Total", c)

    g.map(sns.scatterplot, 'Total', c)

print(f"{y_}The Pokemon with highest HP is \n{b_}{df.loc[df['HP'] == df_full['HP'].max()]['Name']}")

print(f"\n{y_}The Pokemon with lowest HP is \n{b_}{df.loc[df['HP'] == df_full['HP'].min()]['Name']}")
f = urllib.request.urlopen("https://pokestop.io/img/pokemon/blissey-256x256.png")

f1 = urllib.request.urlopen("https://img.rankedboost.com/wp-content/plugins/pokemon-sword-shield/assets/pokemon-images-regular/Shedinja.png")

img = mpimg.imread(f)

img1 = mpimg.imread(f1)



plt.figure(figsize=(10,10))

plt.subplot(121)

imgplot = plt.imshow(img)

plt.title('Highest HP')

plt.axis('off')





plt.subplot(122)

imgplot1 = plt.imshow(img1)

plt.axis('off')

plt.title('Lowest HP')



plt.show()
print(f"{y_}The Pokemon with highest Attack is \n{b_}{df.loc[df['Attack'] == df_full['Attack'].max()]['Name']}")

print(f"\n{y_}The Pokemon with lowest Attack is \n{b_}{df.loc[df['Attack'] == df_full['Attack'].min()]['Name']}")
f = urllib.request.urlopen("https://in.portal-pokemon.com/play/resources/pokedex/img/pm/dc96945bf5cb7f776f0272bf17ebf0d4593a5849.png")

f1 = urllib.request.urlopen("https://www.serebii.net/swordshield/pokemon/113.png")

f2 = urllib.request.urlopen("https://www.serebii.net/swordshield/pokemon/440.png")



img = mpimg.imread(f)

img1 = mpimg.imread(f1)

img2 = mpimg.imread(f2)



plt.figure(figsize=(15,15))

plt.subplot(131)

imgplot = plt.imshow(img)

plt.title('Highest Attack')

plt.axis('off')





plt.subplot(132)

imgplot1 = plt.imshow(img1)

plt.axis('off')

plt.title('Lowest Attack')



plt.subplot(133)

imgplot1 = plt.imshow(img2)

plt.axis('off')

plt.title('Lowest Attack')



plt.show()
print(f"{y_}The Pokemon with highest Defense is \n{b_}{df.loc[df['Defense'] == df_full['Defense'].max()]['Name']}")

print(f"\n{y_}The Pokemon with lowest Defense is \n{b_}{df.loc[df['Defense'] == df_full['Defense'].min()]['Name']}")
f = urllib.request.urlopen("https://sg.portal-pokemon.com/play/resources/pokedex/img/pm/f6e89e59cf6c2de593179ff7c2825403fdd494e7.png")

f1 = urllib.request.urlopen("https://www.serebii.net/swordshield/pokemon/213.png")

f2 = urllib.request.urlopen("https://i.pinimg.com/originals/a6/72/1e/a6721e086b20846b79050bff722a56c5.png")



img = mpimg.imread(f)

img1 = mpimg.imread(f1)

img2 = mpimg.imread(f2)



plt.figure(figsize=(15,15))

plt.subplot(131)

imgplot = plt.imshow(img)

plt.title('Highest Defense')

plt.axis('off')





plt.subplot(132)

imgplot1 = plt.imshow(img1)

plt.axis('off')

plt.title('Highest Defense')



plt.subplot(133)

imgplot1 = plt.imshow(img2)

plt.axis('off')

plt.title('Highest Defense')



plt.show()