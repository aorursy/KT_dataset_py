import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
data = pd.read_csv("../input/listings.csv")
data.head(15) #first 15 datas
f,ax = plt.subplots(figsize=(9,9))

sns.heatmap(data.corr(),annot=True,linewidth=.5,fmt ='.2f',ax = ax)
room_types = data.groupby(['room_type'])

for name, rdata in room_types:

    print(name)
room_price = room_types['price'].agg(np.mean)
print(room_price)
room_types['price'].agg(np.mean).plot(kind='bar')

plt.show()
neighbourhood = data.groupby(['neighbourhood'])

#for i,y in neighbourhood : 

#    print(i)
data.groupby(['neighbourhood'])['price'].agg(['mean','count'])
data.groupby(['neighbourhood','room_type'])['price'].agg(['mean'])

data.groupby(['neighbourhood','room_type'])['price'].agg(['mean', 'count'])
plt.rcParams["figure.figsize"] = [20, 20]

data.groupby(['neighbourhood','room_type'])['price'].agg(['mean', 'count']).plot.bar(stacked=True)