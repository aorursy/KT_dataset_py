import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
data_pokemon = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
data_pokemon.head()
data_pokemon.columns
data_pokemon = data_pokemon.drop("#", axis=1)
data_pokemon.head()
data_pokemon.corr()
#Corelation Map
f,ax= plt.subplots(figsize=(10,10))
sns.heatmap(data_pokemon.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
plt.show()
data_pokemon.Speed.plot(kind='line',color='g',label='Speed',linewidth=1,alpha=0.5,grid=True,linestyle=':')
data_pokemon.Defense.plot(kind='line',color='r',label='Defense',linewidth=1,alpha=0.5,grid=True,linestyle='-')
plt.show()
#SCATTER PLOT
data_pokemon.plot(kind='scatter',x='Attack',y='Defense',alpha=0.5,color='b')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.title('Attack-Defense Scatter Plot')
plt.show()
#HISTOGRAM
data_pokemon.Speed.plot(kind='hist',bins=50,figsize=(10,10),grid=True)
plt.show()
#plt.clf()  clear plt
#Dictionary Yapısı

dictionary={'spain':'madrid','usa':'vegas'}
print(dictionary.keys())
print(dictionary.values())
dictionary['spain']= "barcelona"
dictionary['france']="paris" #add new entry
#del dictionary['france']
#dictionary.clear()
#del dictionary 
for key,value in dictionary.items():
    print(key," : ", value)
#Seriler ve DataFrame Kavramı
seri=data_pokemon['Defense']
seri.head()
dataframe=data_pokemon[['Defense']]
dataframe.head()
#type(dataframe)
x= data_pokemon.Defense>200
x
data_pokemon[np.logical_and(data_pokemon.Defense>200,data_pokemon.Attack>100)]
data_pokemon[(data_pokemon.Defense>200)&(data_pokemon.Attack>100)]
data_pokemon[x]