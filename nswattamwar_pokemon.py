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
df=pd.read_csv('../input/Pokemon.csv')

df=df.drop('#',axis=1)

df = df.set_index('Name')

df.head()
df.columns = [c.replace(' ', '_') for c in df.columns]

po_type=set(df.Type_1)

po_type
summary=df.describe()

summary
po_ttotal=[]

for i in po_type:

   

    y=df.Total[df.Type_1==i]

    add=sum(y)

    po_ttotal.append(add)

    

print(po_ttotal)
keys=list(po_type)

values=po_ttotal

#new_dict = {k: v for k, v in zip(keys, values)} # list comprehension

dictionary = dict(zip(keys, values))

print(dictionary)

import matplotlib.pyplot as plt

x=plt.bar(dictionary.keys(), dictionary.values(),width = 0.5, color='g',)



x = plt.gca().xaxis

for item in x.get_ticklabels():

    item.set_rotation(90)

x = plt.gca()

x.set_xlabel('Type of pokemon')

x.set_ylabel('Total power')

x.set_title('Total strength')
#df.head()

y=df[df['Legendary']==True]

z=df[df['Legendary']==False]

print('The legendary pokemons are {}'.format(len(y)))

print('The non legendary pokemons are {}'.format(len(z)))
a=df[df['Type_1']=='Grass']

b=df[df['Type_1']=='Bug']



s1=(sum(a['Total']))

s2=(sum(b['Total']))

s1=plt.scatter(x='Attack',y='Defense', c='blue',data=a)

s2=plt.scatter(x='Attack',y='Defense', c='red',data=b)

x = plt.gca()

x.set_xlabel('Attack')

x.set_ylabel('Defense')

x.set_title('Comparision')

x.legend((s1,s2),('Grass','Bug'))
print('select type to plot 0.Normal, 1.Dark, 2.Ground, 3.Dragon, 4.Electric, 5.Psychic, 6.Fighting, 7.Flying, 8.Bug, 9.Fire, 10.Fairy, 11.Steel, 12.Poison, 13.Grass, 14.Ice, 15.Rock, 16.Ghost, 17.Water')

d=int(input(' '))



e=int(input(' '))



keys=list(po_type)

t1=keys[d]

t2=keys[e]





a=df[df['Type_1']==t1]

b=df[df['Type_1']==t2]



s1=(sum(a['Total']))

s2=(sum(b['Total']))

s1=plt.scatter(x='Attack',y='Defense', c='blue',data=a)

s2=plt.scatter(x='Attack',y='Defense', c='red',data=b)

x = plt.gca()

x.set_xlabel('Attack')

x.set_ylabel('Defense')

x.set_title('Comparision of different types')

x.legend((s1,s2),(t1,t2))






