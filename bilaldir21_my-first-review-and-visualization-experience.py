import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for data visulizations

import seaborn as sns  # to draw a heat map



import os

print(os.listdir("../input"))

data=pd.read_csv('../input/Pokemon.csv')  #import data



data.info()
data.columns
data.shape
data.head()
data.drop('#',axis=1,inplace=True)
data.head(10)
# Histogram

# bins = number of bar in figure

data.plot(kind = 'hist',bins = 50)

plt.show()
data.describe()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True, linewidths=.5,linecolor='w',cmap="YlGnBu", fmt= '.2f',ax=ax)

plt.show()
plt.plot(data.Attack,data.Defense,color='orange')

plt.xlabel('Attack')         # Name of X

plt.ylabel('Defense')        # Name of Y

plt.title('Attack-Defense Scatter Plot') #Name of Title

plt.gcf().set_size_inches((10, 10)) #To enlarge the size of the image.

plt.show() #do not forget that when you drawing plot
# Scatter Plot 

# x = Attack, y = Defense

data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'blue')

plt.xlabel('Attack')         # Name of X

plt.ylabel('Defense')        # Name of Y

plt.title('Attack-Defense Scatter Plot') #Name of Title

plt.gcf().set_size_inches((10, 10)) #To enlarge the size of the image.

plt.show() #do not forget that when you drawing plot
data
#sort by type

Water= data[data["Type 1"]=="Water"]

Steel= data[data["Type 1"]=="Steel"]

Rock= data[data["Type 1"]=="Rock"]

Psychic= data[data["Type 1"]=="Psychic"]

Poison= data[data["Type 1"]=="Water"]

Normal= data[data["Type 1"]=="Normal"]

Ice= data[data["Type 1"]=="Ice"]

Ground= data[data["Type 1"]=="Ground"]

Grass= data[data["Type 1"]=="Grass"]

Ghost= data[data["Type 1"]=="Ghost"]

Fire= data[data["Type 1"]=="Fire"]

Fighting= data[data["Type 1"]=="Fighting"]

Fairy= data[data["Type 1"]=="Fairy"]

Electric= data[data["Type 1"]=="Electric"]

Dragon= data[data["Type 1"]=="Dragon"]

Dark= data[data["Type 1"]=="Dark"]

Bug= data[data["Type 1"]=="Bug"]

plt.scatter(Water.Attack,Water.Defense,color="red",label="Water")

plt.scatter(Steel.Attack,Steel.Defense,color="blue",label="Steel")

plt.scatter(Rock.Attack,Rock.Defense,color="green",label="Rock")

plt.scatter(Psychic.Attack,Psychic.Defense,color="red",label="Psychic")

plt.scatter(Poison.Attack,Poison.Defense,color="c",label="Poison")

plt.scatter(Normal.Attack,Normal.Defense,color="m",label="Normal")

plt.scatter(Ice.Attack,Ice.Defense,color="w",label="Ice")

plt.scatter(Ground.Attack,Ground.Defense,color="pink",label="Ground")

plt.scatter(Grass.Attack,Grass.Defense,color="orange",label="Grass")

plt.scatter(Ghost.Attack,Ghost.Defense,color="brown",label="Ghost")

plt.scatter(Fire.Attack,Fire.Defense,color="olive",label="Fire")

plt.scatter(Fighting.Attack,Fighting.Defense,color="teal",label="Fighting")

plt.scatter(Fairy.Attack,Fairy.Defense,color="tan",label="Fairy")

plt.scatter(Electric.Attack,Electric.Defense,color="tomato",label="Electric")

plt.scatter(Dragon.Attack,Dragon.Defense,color="yellow",label="Dragon")

plt.scatter(Dark.Attack,Dark.Defense,color="cyan",label="Dark")

plt.scatter(Bug.Attack,Bug.Defense,color="indigo",label="Bug")

plt.legend()

plt.xlabel("Attack")

plt.ylabel("Defense")

plt.title("scatter plot")

plt.gcf().set_size_inches((10, 10))

plt.show()
plt.hist(Water.Attack,bins= 40,color="red")

plt.hist(Steel.Attack,bins= 40,color="blue")

plt.hist(Rock.Attack,bins= 40,color="green")

plt.hist(Psychic.Attack,bins= 40,color="gold")

plt.hist(Poison.Attack,bins= 40,color="c")

plt.hist(Normal.Attack,bins= 40,color="m")

plt.hist(Ice.Attack,bins=40,color="w")

plt.hist(Ground.Attack,bins= 40,color="pink")

plt.hist(Grass.Attack,bins= 40,color="orange")

plt.hist(Ghost.Attack,bins= 40,color="brown")

plt.hist(Fire.Attack,bins= 40,color="olive")

plt.hist(Fighting.Attack,bins= 40,color="teal")

plt.hist(Fairy.Attack,bins= 40,color="tan")

plt.hist(Electric.Attack,bins= 40,color="tomato")

plt.hist(Dragon.Attack,bins= 40,color="yellow")

plt.hist(Dark.Attack,bins= 40,color="cyan")

plt.hist(Bug.Attack,bins= 40,color="indigo")

plt.xlabel("values")

plt.ylabel("Sayı")

plt.title("Histogram Dağılımı")

plt.gcf().set_size_inches((10, 10))

plt.show()
plt.plot(Water.HP,Water.Speed,color="red",label= "Water")

plt.plot(Steel.HP,Steel.Speed,color="blue",label= "Steel")

plt.plot(Rock.HP,Rock.Speed,color="green",label= "Rock")

plt.plot(Psychic.HP,Psychic.Speed,color="gold",label= "Psychic")

plt.plot(Poison.HP,Poison.Speed,color="c",label= "Poison")

plt.plot(Normal.HP,Normal.Speed,color="m",label= "Normal")

plt.plot(Ice.HP,Ice.Speed,color="w",label= "Ice")

plt.plot(Ground.HP,Ground.Speed,color="pink",label= "Ground")

plt.plot(Grass.HP,Grass.Speed,color="orange",label= "Grass")

plt.plot(Ghost.HP,Ghost.Speed,color="brown",label= "Ghost")

plt.plot(Fire.HP,Fire.Speed,color="olive",label= "Fire")

plt.plot(Fighting.HP,Fighting.Speed,color="teal",label= "Fighting")

plt.plot(Fairy.HP,Fairy.Speed,color="tan",label= "Fairy")

plt.plot(Electric.HP,Electric.Speed,color="tomato",label= "Electric")

plt.plot(Dragon.HP,Dragon.Speed,color="yellow",label= "Dragon")

plt.plot(Dark.HP,Dark.Speed,color="cyan",label= "Dark")

plt.plot(Bug.HP,Bug.Speed,color="indigo",label= "Bug")

plt.gcf().set_size_inches((15, 15))

plt.legend()

plt.xlabel("HP")

plt.ylabel("Speed")

plt.show()