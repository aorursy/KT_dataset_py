import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('../input/pokemon/Pokemon.csv')
df.head()
df.info()
df['Type 1'].value_counts()

# df.groupby(['Type 1']).size()
import matplotlib.pyplot as plt
a = df['Attack']

b = df['Defense']



plt.plot(a,b,'x',color='red')

plt.show()
rng = np.random.RandomState(0)

x = rng.rand(100)

y = rng.rand(100)



colors = rng.rand(100)

sizes = 1000 * rng.rand(100)
# creat scatter graph

plt.scatter(x,y,c=colors,s=sizes,alpha=0.3,cmap='viridis')
a = df['Attack']



print(df.Attack.mean)

b = df['Defense']

df.Attack.describe()
fig = plt.figure(figsize=(10,6))



ax = fig.add_subplot(1,1,1)



ax.plot(a,b,'o',alpha=0.4)



ax.set_xlabel('ATTACK')

ax.set_ylabel('DEFENSE')



ax.set_title('Pokemon Attack Vs. Defense',fontdict={'size':16})



ax.grid(True)
spdDef = df['Sp. Def']

spdAtk = df['Sp. Atk']



fig = plt.figure(figsize=(10,6))



# row , column , what subplot of graph

ax = fig.add_subplot(1,1,1)



ax.plot(spdDef,spdAtk,'o',alpha=0.4)



ax.set_xlabel('Speed Def')

ax.set_ylabel('Speed Atk')



ax.set_title('Pokemon Speed Attack Vs. Speed Defense',fontdict={'size':16})



ax.grid(True)
import seaborn as sns
sns.set(style= 'darkgrid',palette='deep')
# relation plot

sns.relplot(x = 'Attack' , y = 'Defense', data =df)
sns.relplot(x = 'Attack' , y = 'Defense', data =df , col = 'Generation' , col_wrap = 3)
sns.relplot(x = 'Attack' , y = 'Defense', data =df , col = 'Generation' , col_wrap = 3 ,hue='Type 1')
sns.relplot(x = 'Attack' , y = 'Defense', data =df , col = 'Generation' , col_wrap = 3 ,hue='Type 2')
# plot graph compare between vars

sns.pairplot(df , height = 5 , vars = (['Defense','Attack','Speed']))
sns.pairplot(df ,hue='Generation', height = 5 , vars = (['Defense','Attack','Speed']))
sns.pairplot(df ,hue='Generation', height = 5 , vars = (['Defense','HP','Attack']))
sns.catplot(x='Generation' , y = 'Speed' , kind = 'box' , data = df , hue = 'Generation')
sns.catplot(x='Type 1' , y = 'Total' , kind = 'box' , data = df , hue = 'Type 1')