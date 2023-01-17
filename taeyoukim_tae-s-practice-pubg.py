import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv("../input/train_V2.csv")
test = pd.read_csv("../input/test_V2.csv")
train.head()
test.head()
train.info()
data = train.copy()

data.loc[data['kills'] > data['kills'].quantile(0.99)] = '8+'

plt.figure(figsize=(15,10))

sns.countplot(data['kills'].astype('str').sort_values())

plt.title("Kill Count",fontsize=15)

plt.show()
kills = train.copy()



kills['killsCategories'] = pd.cut(kills['kills'], [-1, 0, 2, 5, 10, 60], labels=['0_kills','1-2_kills', '3-5_kills', '6-10_kills', '10+_kills'])



plt.figure(figsize=(15,8))

sns.boxplot(x="killsCategories", y="winPlacePerc", data=kills)

plt.show()
data = train.copy()

data = data[data['heals'] < data['heals'].quantile(0.99)]

data = data[data['boosts'] < data['boosts'].quantile(0.99)]



f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='heals',y='winPlacePerc',data=data,color='lime',alpha=0.8)

sns.pointplot(x='boosts',y='winPlacePerc',data=data,color='blue',alpha=0.8)

plt.text(4,0.6,'Heals',color='lime',fontsize = 17,style = 'italic')

plt.text(4,0.55,'Boosts',color='blue',fontsize = 17,style = 'italic')

plt.xlabel('Number of heal/boost items',fontsize = 15,color='blue')

plt.ylabel('Win Percentage',fontsize = 15,color='blue')

plt.title('Heals vs Boosts',fontsize = 20,color='blue')

plt.grid()

plt.show()