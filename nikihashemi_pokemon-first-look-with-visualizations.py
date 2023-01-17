#Import Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Load Pokemon dataset
pokemon = pd.read_csv('../input/Pokemon.csv')

#First look at the data
pokemon.head()
type_one_count = pokemon['Type 1'].value_counts()
type_one_count.plot.bar()
plt.title('Types of Pokemon')
plt.xlabel('Type')
plt.ylabel('Number')
plt.show()

stats_by_generation = pokemon.groupby('Generation').mean()[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]
stats_by_generation.plot.line()
plt.title('Stats by Generation')
plt.xlabel('Generation')
plt.ylabel('Number')
plt.show()
pokemon2 = pokemon.drop(['Total','#','Generation','Legendary'],1)
sns.boxplot(data = pokemon2)
plt.title('Combined Box Plots')
plt.xticks(rotation = 90)
plt.show()
dist_hp = pokemon['HP']
sns.distplot(dist_hp)
plt.title('Distribution of Hit Points')
plt.show()
dist_attack = pokemon['Attack']
sns.distplot(dist_attack)
plt.title('Distribution of Attack')
plt.show()
dist_defense = pokemon['Defense']
sns.distplot(dist_defense)
plt.title('Distribution of Defense')
plt.show()
dist_spattack = pokemon['Sp. Atk']
sns.distplot(dist_spattack)
plt.title('Distribution of Special Attack')
plt.show()
dist_spdefense = pokemon['Sp. Def']
sns.distplot(dist_spdefense)
plt.title('Distribution of Special Defense')
plt.show()
dist_speed = pokemon['Speed']
sns.distplot(dist_speed)
plt.title('Distribution of Speed')
plt.show()