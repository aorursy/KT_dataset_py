import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_style('whitegrid')
pokemon = pd.read_csv('../input/Pokemon.csv')
pokemon.head(6)
print(list(pokemon))
print(len(list(pokemon)))
print(pokemon.info())
print(pokemon.describe())
print(pokemon.isnull().sum())
pokemon.rename(columns={'Type 1': 'TypeA','Type 2':'TypeB', 'Sp. Atk':'Sp.Atk','Sp. Def':'Sp.Def'}, inplace=True)
pokemon.drop('#', axis=1,inplace=True)
print(pokemon.groupby(['TypeA']).size())
sns.set_context('poster',font_scale=1.1)
plt.figure(figsize=(12,9))
sns.heatmap(pokemon.isnull(),yticklabels=False,cbar=False, cmap='viridis')
plt.figure(figsize=(9,5))
ax = sns.countplot(x='Generation',data=pokemon,palette='viridis')
ax.axes.set_title("Pokemon Generations",fontsize=18)
ax.set_xlabel("Generation", fontsize=16)
ax.set_ylabel("Total", fontsize=16)
sns.set_style('whitegrid')
sns.lmplot('Defense','Attack',data=pokemon, hue='Generation',
           palette='Spectral',size=8,aspect=1.4,fit_reg=False)
plt.figure(figsize=(9,5))
a = sns.countplot(y="TypeA", data=pokemon, palette="Blues_d",
              order=pokemon.TypeA.value_counts().iloc[:7].index)
a.axes.set_title("Top 7",fontsize=18)
a.set_xlabel("Total",fontsize=16)
a.set_ylabel("TypeA", fontsize=16)
plt.figure(figsize=(15,8))
ax = sns.countplot(x='TypeB',data=pokemon,palette='viridis', order=pokemon.TypeB.value_counts().index)
ax.axes.set_title("Type B Pokemons",fontsize=20)
ax.set_xlabel("Type",fontsize=16)
ax.set_ylabel("Total",fontsize=16)
for item in ax.get_xticklabels():
    item.set_rotation(60)
x = pokemon['Total']

bins = np.arange(150, 800, 12)
plt.figure(figsize=(14,8))
sns.set_context('poster')
ax = sns.distplot(x, kde=False, bins = bins, color = 'darkred',hist_kws={"alpha":0.7})
ax.axes.set_title("Total power of pokemons",fontsize=25)
ax.set_xlabel("Total")
ax.set_ylabel("Pokemons")
plt.figure(figsize=(10, 7.5))
sns.boxplot(x='Legendary',y='HP',data=pokemon,palette='winter')
sns.set_style('whitegrid')
sns.lmplot('Total','Attack',data=pokemon, hue='Legendary',
           palette='coolwarm',height=8,aspect=1.3,fit_reg=False)
plt.figure(figsize=(12,8))
sns.set_context('poster')
ax = sns.regplot(x="Legendary", y="Total", data=pokemon, x_jitter=.07, fit_reg=False,color="darkseagreen",)
ax.axes.set_title("Normal VS Legendary",fontsize=25)
ax.set_xlabel("N vs L")
ax.set_ylabel("Total")
sns.set_style('whitegrid')
sns.set_context('poster',font_scale=1.1)
sns.lmplot(x="HP", y="Attack", col="Legendary",data=pokemon,
           palette='coolwarm',height=8,aspect=1.4,fit_reg=True)
plt.figure(figsize=(12,8))
sns.set_context('poster')
ax = sns.regplot(x="Generation", y="Total", data=pokemon, x_jitter=.07, fit_reg=False,color="darkseagreen",)
ax.axes.set_title("Power per generation",fontsize=25)
ax.set_xlabel("Generations")
ax.set_ylabel("Total")

















































































