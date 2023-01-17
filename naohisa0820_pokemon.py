# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import

import warnings 

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()
#読み込み

#pokemon = pd.read_csv("Pokemon.csv",index_col = 0)

pokemon = pd.read_csv("../input/Pokemon.csv",index_col = 0)
#データ数、カラム数

pokemon.shape
pokemon.head()
#data type

pokemon.dtypes
#columns

#カラム種類

print(len(pokemon.columns),"種類")

print("カラム",pokemon.columns)



#Legendary = 伝説

#各種統計量

pokemon.describe()
#Total

sns.distplot(pokemon.Total,kde = True)

plt.ylabel("frequency")

plt.title("Total distribution")



#山が二つできている　ある値が山を作っている？(2種類あり)
#HP

sns.distplot(pokemon.HP,kde=True)

plt.ylabel("frequency")

plt.title("HP distribution")
#Attack

sns.distplot(pokemon.Attack,kde=True)

plt.ylabel("frequency")

plt.title("Attack distribution")
#Defense

sns.distplot(pokemon.Defense,kde=True)

plt.ylabel("frequency")

plt.title("Defense distribution")
#Sp. Atk

sns.distplot(pokemon["Sp. Atk"],kde=True)

plt.ylabel("frequency")

plt.title("Sp. Atk distribution")
#Sp. Def

sns.distplot(pokemon["Sp. Def"],kde=True)

plt.ylabel("frequency")

plt.title("Sp. Def distribution")
#Speed

sns.distplot(pokemon["Speed"],kde=True)

plt.ylabel("frequency")

plt.title("Speed distribution")
#タイプ別数

pokemon["Type 1"].value_counts().plot(kind="barh")

#sns.countplot(pokemon["Type 1"])
print(len(pokemon["Type 1"].value_counts()))

print(pokemon["Type 1"].value_counts())
#タイプ別数

pokemon["Type 2"].value_counts().plot(kind="barh")
print(len(pokemon["Type 2"].value_counts()))

print(pokemon["Type 2"].value_counts())
#Total

plt.figure(figsize=(12,8))

sns.boxplot(x=pokemon["Type 1"],y=pokemon["Total"],data=pokemon)
#HP

plt.figure(figsize=(12,8))

sns.boxplot(x=pokemon["Type 1"],y=pokemon["HP"],data=pokemon)
#Attack

plt.figure(figsize=(12,8))

sns.boxplot(x=pokemon["Type 1"],y=pokemon["Attack"],data=pokemon)
#Defense

plt.figure(figsize=(12,8))

sns.boxplot(x=pokemon["Type 1"],y=pokemon["Defense"],data=pokemon)
#Sp. Atk

plt.figure(figsize=(12,8))

sns.boxplot(x=pokemon["Type 1"],y=pokemon["Sp. Atk"],data=pokemon)
#Sp. Def

plt.figure(figsize=(12,8))

sns.boxplot(x=pokemon["Type 1"],y=pokemon["Sp. Def"],data=pokemon)
#Speed

plt.figure(figsize=(12,8))

sns.boxplot(x=pokemon["Type 1"],y=pokemon["Speed"],data=pokemon)
legend_poke = pokemon[pokemon["Legendary"]==True]

nlegend_poke = pokemon[pokemon["Legendary"]==False]
print(len(legend_poke))

print(len(nlegend_poke))
legend_poke.describe()
nlegend_poke.describe()
legend_poke["Type 1"].value_counts().plot(kind="barh")
plt.figure(figsize=(18,8))



plt.subplot(2,4,1)

sns.distplot(legend_poke["Total"],kde=True,color="red",label="legend")

sns.distplot(nlegend_poke["Total"],kde=True,color="green",label="N-ledend")

plt.legend()



plt.subplot(2,4,2)

sns.distplot(legend_poke["Attack"],kde=True,color="red",label="legend")

sns.distplot(nlegend_poke["Attack"],kde=True,color="green",label="N-ledend")

plt.legend()



plt.subplot(2,4,3)

sns.distplot(legend_poke["Attack"],kde=True,color="red",label="legend")

sns.distplot(nlegend_poke["Attack"],kde=True,color="green",label="N-ledend")

plt.legend()



plt.subplot(2,4,4)

sns.distplot(legend_poke["Defense"],kde=True,color="red",label="legend")

sns.distplot(nlegend_poke["Defense"],kde=True,color="green",label="N-ledend")

plt.legend()



plt.subplot(2,4,5)

sns.distplot(legend_poke["Sp. Atk"],kde=True,color="red",label="legend")

sns.distplot(nlegend_poke["Sp. Atk"],kde=True,color="green",label="N-ledend")

plt.legend()



plt.subplot(2,4,6)

sns.distplot(legend_poke["Sp. Def"],kde=True,color="red",label="legend")

sns.distplot(nlegend_poke["Sp. Def"],kde=True,color="green",label="N-ledend")

plt.legend()



plt.subplot(2,4,7)

sns.distplot(legend_poke["Speed"],kde=True,color="red",label="legend")

sns.distplot(nlegend_poke["Speed"],kde=True,color="green",label="N-ledend")

plt.legend()
#Total

plt.figure(figsize=(12,8))

sns.boxplot(x=legend_poke["Type 1"],y=legend_poke["Total"],data=legend_poke)



#HP

plt.figure(figsize=(12,8))

sns.boxplot(x=legend_poke["Type 1"],y=legend_poke["HP"],data=legend_poke)



#Attack

plt.figure(figsize=(12,8))

sns.boxplot(x=legend_poke["Type 1"],y=legend_poke["Attack"],data=legend_poke)



#Defense

plt.figure(figsize=(12,8))

sns.boxplot(x=legend_poke["Type 1"],y=legend_poke["Defense"],data=legend_poke)



#Sp. Atk

plt.figure(figsize=(12,8))

sns.boxplot(x=legend_poke["Type 1"],y=legend_poke["Sp. Atk"],data=legend_poke)



#Sp. Def

plt.figure(figsize=(12,8))

sns.boxplot(x=legend_poke["Type 1"],y=legend_poke["Sp. Def"],data=legend_poke)



#Speed

plt.figure(figsize=(12,8))

sns.boxplot(x=legend_poke["Type 1"],y=legend_poke["Speed"],data=legend_poke)
generation = []

gene_num = []



for i in range(1,7):

    generation.append(pokemon[pokemon["Generation"] == i])

    

for i in range(0,6):

    print(i+1,"世代ポケモン数: ",len(generation[i]))

    gene_num.append(len(generation[i]))



num = np.arange(1,7)



plt.bar(num,gene_num)#x軸 y軸