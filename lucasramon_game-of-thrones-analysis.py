import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter  
battles = pd.read_csv("../input/battles.csv")

character_deaths = pd.read_csv("../input/character-deaths.csv")

character_predictions = pd.read_csv("../input/character-predictions.csv")
a = Counter(character_deaths['Book Intro Chapter'])

b = Counter(character_deaths['Death Chapter'])

c = Counter(character_deaths['Book of Death'])

d = Counter(battles['attacker_king'])

e = Counter(battles['defender_king'])

f = Counter(battles['region'])

g = Counter(battles['battle_type'])

h = character_predictions['alive']

i = character_predictions['popularity']
plt.bar(a.keys(), a.values())

plt.title('Book Intro Chapter')

plt.xlabel('Chapter')

plt.ylabel('Characters')

plt.show()
plt.bar(b.keys(), b.values())

plt.title('Death Chapter')

plt.xlabel('Chapter')

plt.ylabel('Character deaths')

plt.show()
plt.bar(c.keys(), c.values())

plt.title('Book of Death')

plt.xlabel('Book')

plt.ylabel('Character deaths')

plt.show()
character_deaths.groupby(['Allegiances']).sum()[["Nobility"]].plot.bar()

plt.xlabel('Allegiances')

plt.ylabel('Notable deaths')

plt.show()
plt.pie([float(v) for v in d.values()], labels=[k for k in d.keys()],autopct='%1.1f%%', startangle=90)

plt.axis('equal')

plt.title('Attacker king',  loc='left')

plt.show()
plt.pie([float(v) for v in e.values()], labels=[k for k in e.keys()],autopct='%1.1f%%', startangle=90)

plt.axis('equal')

plt.title('Defender king', loc='left')

plt.show()
plt.pie([float(v) for v in f.values()], labels=[k for k in f.keys()],autopct='%1.1f%%')

plt.axis('equal')

plt.title('Region', loc='left')

plt.show()
plt.pie([float(v) for v in g.values()], labels=[k for k in g.keys()],autopct='%1.1f%%', startangle=90)

plt.axis('equal')

plt.title('Battle type', loc='left')

plt.show()
plt.boxplot(h.values)

plt.title('Alive')

plt.show()
plt.boxplot(i.values)

plt.title('Popularity')

plt.show()