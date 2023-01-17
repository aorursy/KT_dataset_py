import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/house-price/House_Price.csv', header=0, engine= 'python')
data.head(5)
data.isnull().sum()
data['n_hos_beds'] = data['n_hos_beds'].fillna(data['n_hos_beds'].mean())
data.describe()
data_n = data.drop(['price'], axis=1).copy()
num = [f for f in data_n if data_n[f].dtypes !='O']

num
cat = [f for f in data_n if data_n[f].dtypes =='O']

cat
for f in cat:

    dataC = data_n.copy()

    sns.countplot(data[f])

    plt.show()
del data['bus_ter']
for f in num:

    dataC = data_n.copy()

    sns.jointplot(dataC[f], y=data['price'])

    plt.show()
room_99 = np.percentile(data['n_hot_rooms'],99)

room_99
data[data['n_hot_rooms']>room_99]
data.n_hot_rooms[data['n_hot_rooms']>room_99*2 ]= room_99*2
data[data['n_hot_rooms']>room_99]
rainfall = np.percentile(data['rainfall'], 1)

rainfall
data[data['rainfall']<rainfall]
data.rainfall[data['rainfall']<rainfall*0.3] = rainfall*0.3
data[data['rainfall']<rainfall]
for f in num:

    dataC = data_n.copy()

    data[f].hist()

    plt.xlabel(f)

    plt.show()
data['crime_rate'] = np.log(data['crime_rate']+1)
data['age'] = np.log(data['age']+1)
sns.jointplot(data['crime_rate'], data['price'])
data['dist_avg'] = (data['dist1'] + data['dist2'] + data['dist3'] + data['dist4'])/4
data = data.drop(['dist1', 'dist2', 'dist3', 'dist4'], axis=1)
sns.jointplot(data['n_hot_rooms'], data['price'])
data.corr()
corr = data.corr()
plt.figure(figsize=(14, 10))

sns.set(style='white')



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(12, 10))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})