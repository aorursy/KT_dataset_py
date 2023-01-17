# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/fifa-20-complete-player-dataset/players_20.csv')
data.columns
data.describe()
data.info()

#checking data
# in order to avoid insufficient data,I will delete nations that have less than 11 players,so I have to find nations

#that have less than 11 players

weak_data=data.nationality.value_counts()

weak_data.tail(72)

# as you can see the first 71 countries seems insufficient to use for our data
weak_data=weak_data.tail(71)

weak_data=weak_data.index # taking only index, not values
nation_list=list(data.nationality.unique())

nation_list = [x for x in nation_list if x not in weak_data] #creating a clean nation list
avg_weight=[] # creating a list to gather average weight
for i in nation_list:

    x=data[data["nationality"]==i]

    rate=sum(x.weight_kg)/len(x)

    avg_weight.append(rate) # creating my list
#sorting the data

new_data=pd.DataFrame({"nation_list":nation_list,"average_weight":avg_weight})

new_index=(new_data["average_weight"].sort_values(ascending=False)).index.values

sorted_data=new_data.reindex(new_index)

last_data=sorted_data.head(20)  
#ploting the graph

plt.figure(figsize=(35,15))

sns.barplot(x=last_data["nation_list"],y=last_data["average_weight"],palette=sns.cubehelix_palette(20, start=4, rot=-.5))

plt.xticks(rotation=50,fontsize=20)

plt.yticks(fontsize=20)

plt.xlabel("Nations",fontsize=25)

plt.ylabel("Weight in kg",fontsize=25,rotation=90)

plt.title("Countries that have most heavy players",fontsize=45)

plt.legend(["average weight"],fontsize=20)

plt.grid(True)