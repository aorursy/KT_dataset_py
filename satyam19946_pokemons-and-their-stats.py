import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "darkgrid")
pkmn = pd.read_csv("../input/Pokemon.csv")
print(pkmn.head(5))
sns.lmplot(x = "Attack", y = "Defense",data = pkmn, fit_reg= False, hue = "Generation")
boxdf = pkmn.drop(['#','Total','Generation','Legendary'],axis = 1)
sns.boxplot(data = boxdf)
pkmn['Type 1'].value_counts().plot.bar()
plt.title("Number of Pokemons of Type 1")
pkmn['Type 2'].value_counts().plot.bar()
plt.title("Number of Pokemons of Type 2")
pkmn.Generation.loc[pkmn.Legendary.apply(lambda g: g == True)].value_counts().plot.bar()
plt.xlabel("Generation")
#Pokemon with the highest health
pkmn.loc[pkmn.HP.apply(lambda g: g == max(pkmn.HP))]
#Pokemon with the highest attack
pkmn.loc[pkmn.Attack.apply(lambda g: g == max(pkmn.Attack))]
#Pokemon with the highest Defence
pkmn.loc[pkmn.Defense.apply(lambda g: g == max(pkmn.Defense))]
#Pokemon with the highest average of HP,Attack,Defense
Indexofhighestavg = (((pkmn.Defense+pkmn.Attack+pkmn.Speed+pkmn['Sp. Atk']+pkmn['Sp. Def']+pkmn.Speed)/6).sort_values(ascending = False).index)[0:10]
pkmn.iloc[Indexofhighestavg]