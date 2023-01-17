# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/yugioh-trading-cards-dataset/card_data.csv")

data=data.drop_duplicates()

data.head()
data.shape
traps=data[data["Type"]=="Trap Card"]

traps[0:10]
traps[10:20]
data[data["Name"].str.contains("Solem")]
data[data["Name"].str.contains("Blue-Eyes")]
data[data["Name"].str.contains("Kaiba")]
data[data["Name"].str.contains("Goblin Attack Force")]
data[data["Name"].str.contains("Slate")]
data[data["Name"].str.contains("Gemini Elf")]
data[data["Name"].str.contains("Penguin Soldier")]
monstacardo=data[data["Type"].str.contains("Monst")]

monstacardo.head()
onetribute=monstacardo[(monstacardo["Level"]>=5) & (monstacardo["Level"]<7)]

len(onetribute)
onetribute.sort_values(by="ATK",ascending=False)[0:20]
onetribute.sort_values(by="ATK",ascending=False)[20:30]
onetribute.sort_values(by="ATK",ascending=False)[30:40]
twotribute=monstacardo[(monstacardo["Level"]>=7) & (monstacardo["Level"]<9)]

len(twotribute)
twotribute.sort_values(by="ATK",ascending=False)[0:20]
twotribute.sort_values(by="ATK",ascending=False)[20:30]
twotribute.sort_values(by="ATK",ascending=True)[0:20]
moretribute=monstacardo[monstacardo["Level"]>=9]

len(moretribute)
moretribute[0:10]
moretribute[moretribute["Name"].str.contains("Obeli")]
moretribute[moretribute["Name"].str.contains("Sli")]
moretribute[moretribute["Name"].str.contains("The Wing")]
data.sort_values(by=["ATK","DEF"],ascending=False)[0:30]
data.sort_values(by=["DEF","ATK"],ascending=False)[0:10]
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(20,20))

sns.relplot(data=monstacardo,x="ATK",y="DEF",hue="Attribute")
plt.figure(figsize=(24,24))

sns.relplot(data=monstacardo,x="ATK",y="DEF",hue="Type")
plt.figure(figsize=(24,24))

sns.relplot(data=monstacardo,x="ATK",y="DEF",hue="Race")
plt.figure(figsize=(24,24))

sns.relplot(data=monstacardo,x="ATK",y="DEF",hue="Level")
monstacardo.head()
x=monstacardo[["ATK","DEF"]]

y=monstacardo["Level"]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=320)



from tpot import TPOTRegressor

tpot = TPOTRegressor(generations=1, population_size=50, verbosity=2,max_time_mins=35)

tpot.fit(x_train,y_train)

print(tpot.score(x_test, y_test))

tpot.export('tpot_result.py')
tpot.predict([[1200,400]]) 
tpot.predict([[2500,1400]]) 
for i in range(12,30):

    level=tpot.predict([[2500,i*100]])[0] 

    print("Defense: ",i*100,"  ",level)
from sklearn.tree import DecisionTreeRegressor

dtreg=DecisionTreeRegressor()

dtreg.fit(x_train,y_train)

print(dtreg.score(x_test, y_test))
for i in range(12,30):

    level=dtreg.predict([[2500,i*100]])[0] 

    print("Defense: ",i*100,"  ",level)
from sklearn.tree import DecisionTreeRegressor

dtregmax2=DecisionTreeRegressor(max_depth=3)

dtregmax2.fit(x_train,y_train)

print(dtregmax2.score(x_test, y_test))



for i in range(12,30):

    level=dtreg.predict([[2500,i*100]])[0] 

    print("Defense: ",i*100,"  ",level)
from sklearn import tree

from sklearn import tree



tree.plot_tree(dtreg,filled=True)  

plt.show()
plt.figure(figsize=(30,30))

tree.plot_tree(dtregmax2,filled=True,)

plt.show()
monstacardo["atkdef"]=monstacardo["ATK"]+monstacardo["DEF"]

x=monstacardo[["ATK","DEF","atkdef"]]

y=monstacardo["Level"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=320)

from tpot import TPOTRegressor

tpot = TPOTRegressor(generations=1, population_size=50, verbosity=2,max_time_mins=35)

tpot.fit(x_train,y_train)

print(tpot.score(x_test, y_test))

tpot.export('tpot_result.py')