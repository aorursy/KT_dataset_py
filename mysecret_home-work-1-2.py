import numpy as np 

import pandas as pd 



df = pd.read_csv('../input/pokemon/Pokemon.csv')
df.head()
df.info()
df.groupby(['Type 1']).size()
import matplotlib.pyplot as plt
x = np.linspace(0, 10, 30)

y = np.sin(x)



plt.plot(x, y, 'o', color='red');

plt.show()
a = df['Attack']

b = df['Defense']



plt.plot(a, b, '^', color='blue');

plt.show()
# attack = df.iloc[:,6:7]

# defense = df.iloc[:,7:8]

# plt.plot(attack, defense, 'o', c ='red');

# plt.show()



attack = df.loc[:,['Attack']]

defense = df.loc[:,['Defense']]

plt.plot(attack, defense, 'o', color='red');

plt.show()
rng = np.random.RandomState(0)

x = rng.randn(100)

y = rng.randn(100)



colors = rng.rand(100)

sizes = 1000 * rng.rand(100)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.4,

            cmap='viridis')
a = df['Attack']

b = df['Defense']





plt.plot(a, b, 'o',alpha=0.4);

plt.show()
# Create a figure

fig = plt.figure(figsize=(10, 6))



# Ask, out of a 1x1 grid, the first axes.

ax = fig.add_subplot(1, 1, 1)



ax.plot(a, b, 'o',alpha=0.4);



ax.set_xlabel('ATTACK')

ax.set_ylabel('DEFENSE')



ax.set_title('Pokemon Attack VS. Defense', fontdict={'size':16})



ax.grid(True)



df.Attack.describe()
eachGen = df.loc[:,'Generation']

plt.scatter(a, b, c=eachGen)



plt.plot
c = df['HP']

d = df['Speed']



plt.plot(c, d, 'v',alpha=0.4);

plt.show()



# Create a figure

fig = plt.figure(figsize=(10, 6))



# Ask, out of a 1x1 grid, the first axes.

ax1 = fig.add_subplot(1, 2, 1)



ax1.plot(a, b, 'o',alpha=0.4);



ax1.set_xlabel('ATTACK')

ax1.set_ylabel('DEFENSE')



ax1.set_title('Pokemon Attack VS. Defense', fontdict={'size':16})



ax1.grid(True)



# Ask, out of a 1x1 grid, the first axes.

ax2 = fig.add_subplot(1, 2, 2)



ax2.plot(c, d, 'v',alpha=0.4);



ax2.set_xlabel('HP')

ax2.set_ylabel('Speed')



ax2.set_title('Pokemon HP VS. Speed', fontdict={'size':16})



ax2.grid(True)
import seaborn as sns
sns.set(style = 'darkgrid')

sns.set(palette= 'deep')
sns.relplot(x = 'Attack', y = 'Defense', data = df)
sns.relplot(x = 'Attack', y = 'Defense', data = df, col = 'Generation', col_wrap = 3)
sns.relplot(x = 'Attack', y = 'Defense', data = df, col = 'Generation', col_wrap = 3, hue='Type 1')
sns.relplot(x = 'Sp. Atk', y = 'Sp. Def', data = df, hue='Type 2')
sns.pairplot(df, height = 5, vars = (['Defense', 'Attack','Speed']))
sns.pairplot(df, hue='Generation',height = 5, vars = (['Defense', 'Attack','Speed']))
sns.pairplot(df, hue='Generation',height = 5, vars = (['Total', 'Sp. Atk','Speed']))
sns.catplot(x = 'Generation', y = 'Total', kind = 'box', data = df, hue = 'Generation')
sns.catplot(x = 'Generation', y = 'Speed', kind = 'box', data = df, hue = 'Generation')