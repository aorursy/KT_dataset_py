import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
from collections import OrderedDict,defaultdict

import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("../input/winemag-data-130k-v2.csv")
data.info()
data.head(10)
data.index.name = "index"
data = data.drop("Unnamed: 0",axis=1)
data.head(1)
plt.figure(figsize=(25,10))
sns.countplot(data.country,order = data.country.value_counts().index[:10])
plt.title("Counts of the Evaluation of Countries")
plt.xlabel("Countries")
plt.ylabel("Counts")
plt.show()
plt.figure(figsize=(25,10))
ax = sns.countplot(data.taster_name,order = data.taster_name.value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.title("Counts of the Tasters Works")
plt.xlabel("Taster Name")
plt.ylabel("Counts")
plt.show()
dictionary = {}

for country in data.country.unique():
    filt = data.country == country
    point = data[filt].points.mean()
    dictionary[country]=point

dictionary = OrderedDict(sorted(dictionary.items(), key=itemgetter(1),reverse=True))

plt.figure(figsize=(50,7))
sns.barplot([k for k,v in dictionary.items()],[v for k,v in dictionary.items()])
plt.title("Average Scores of Countries")
plt.xlabel("Cities")
plt.ylabel("Points")
plt.show()
dictionaryPrice = {}

for country in data.country.unique():
    filt = data.country == country
    price = data[filt].price.mean()
    dictionaryPrice[country]=price
    
dictionaryPrice = OrderedDict(sorted(dictionaryPrice.items(), key=itemgetter(1),reverse=True))

plt.figure(figsize=(75,7))
sns.barplot([k for k,v in dictionaryPrice.items()],[v for k,v in dictionaryPrice.items()])
plt.title("Average Wine Prices of Countries")
plt.xlabel("Cities")
plt.ylabel("Price")
plt.show()
filt = data[[i for i in data.country=="Turkey"]].province
dataTurkey = data.iloc[filt.index]

dictionaryTurkey = {}

for province in dataTurkey.province.unique():
    filt = data.province == province
    point = data[filt].points.mean()
    dictionaryTurkey[province]=point
    
dictionaryTurkey = OrderedDict(sorted(dictionaryTurkey.items(), key=itemgetter(1),reverse=True))


plt.figure(figsize=(15,7))
sns.barplot([k for k,v in dictionaryTurkey.items()],[v for k,v in dictionaryTurkey.items()])
plt.title("The Mean Scores of Cities in Turkey")
plt.xlabel("Cities")
plt.ylabel("Points")
plt.show()

plt.figure(figsize=(25,7))
sns.swarmplot(y = dataTurkey.province, x = dataTurkey.price,hue = dataTurkey.variety)
plt.title("Wine Prices in Turkey's Cities")
plt.xlabel("Price")
plt.ylabel("Cities")
plt.grid()
dictionary = defaultdict(list)

for i in data.taster_name.unique()[18:19]:
    filt = data.taster_name == i
    wineryName = data[filt].winery
    wineryPoints = data.iloc[wineryName.index].points
    for k,v in zip(list(wineryName),list(wineryPoints)):
        dictionary[k].append(v)

        
plt.figure(figsize=(20,7))
sns.barplot([k for k,v in dictionary.items()],[sum(v)/len(v) for k,v in dictionary.items()])
plt.title(data.taster_name.unique()[18:19])
plt.xlabel("Winery")
plt.ylabel("Points")
plt.show()