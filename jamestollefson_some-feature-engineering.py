# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

from  geopy.distance import great_circle

from shapely.geometry import MultiPoint

from sklearn.preprocessing import LabelEncoder

from mpl_toolkits.mplot3d import Axes3D



train = pd.read_json('../input/train.json')

print(train.info())
coords = train.as_matrix(columns=['longitude', 'latitude'])

epsilon = .0001

db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine', n_jobs=-1).fit(np.radians(coords))

cluster_labels = db.labels_

num_clusters = len(set(cluster_labels))

clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])

print('Number of clusters: {}'.format(num_clusters))   
neighbors = pd.Series(cluster_labels, index=train.index)

train['clusters'] = neighbors

print(train[train['clusters'] > 0])
train['display_address'] = train['display_address'].str.lower()

train['display_address'] = train['display_address'].str.replace('east ', 'e ')

train['display_address'] = train['display_address'].str.replace('west ', 'w ')

train['display_address'] = train['display_address'].str.replace('north ', 'n ')

train['display_address'] = train['display_address'].str.replace('south ', 's ')

train['display_address'] = train['display_address'].str.replace(' street', ' st')

train['display_address'] = train['display_address'].str.replace(' st.', ' st')

train['display_address'] = train['display_address'].str.replace(' avenue', ' ave')

train['display_address'] = train['display_address'].str.replace(' ave.', ' ave')

train['display_address'] = train['display_address'].str.replace(' boulevard', ' blvd')

train['display_address'] = train['display_address'].str.replace(' blvd.', ' blvd')

train['display_address'] = train['display_address'].str.replace(' pl.', ' pl')

train['display_address'] = train['display_address'].str.replace(' place', ' pl')



number = LabelEncoder()

train['display_address'] = number.fit_transform(train['display_address'].astype('str'))
train['logprice'] = np.log(train['price'])



_ = sns.distplot(train.logprice)

plt.show()

_ = sns.violinplot(x='interest_level', y='logprice', data=train)

plt.show()
highint = train[train['interest_level'] == 'high']

medianhigh = highint.price.median()

medint = train[train['interest_level'] == 'medium']

medianmed = medint.price.median()

lowint = train[train['interest_level'] == 'low']

medianlow = lowint.price.median()

print(medianlow, medianmed, medianhigh)



stdlow = lowint.price.std()

stdmed = medint.price.std()

stdhigh = highint.price.std()

print(stdlow, stdmed, stdhigh)



highint = train[train['interest_level'] == 'high']

logmedianhigh = highint.logprice.median()

medint = train[train['interest_level'] == 'medium']

logmedianmed = medint.logprice.median()

lowint = train[train['interest_level'] == 'low']

logmedianlow = lowint.logprice.median()

print(logmedianlow, logmedianmed, logmedianhigh)



stdloglow = lowint.logprice.std()

stdlogmed = medint.logprice.std()

stdloghigh = highint.logprice.std()

print(stdloglow, stdlogmed, stdloghigh)
train['rooms'] = train['bathrooms'] + train['bedrooms'] + 1

print(train.info())
x = train.logprice

y = train.rooms

hue = train.interest_level

_ = plt.scatter(x=x, y=y, alpha = 0.25, cmap=hue, c=['blue', 'red', 'green'])

_ = plt.xlabel('Log Price')

_ = plt.ylabel('Number of Rooms')

_ = plt.title('Rooms vs. Price')

plt.show()
pd.options.mode.chained_assignment = None



train['room_cost'] = train['rooms'] / train['logprice']

train['interest'] = 1

train['interest'][train['interest_level'] == 'medium'] = 2

train['interest'][train['interest_level'] == 'high'] = 3

_ = sns.violinplot(x='interest', y='room_cost', data=train)

plt.show()

print(train.room_cost)

print(train.rooms.unique())



m1 = train[train['rooms'] == 1].room_cost.median()

m1half = train[train['rooms'] == 1.5].room_cost.median()

m2 = train[train['rooms'] == 2].room_cost.median()

m2half = train[train['rooms'] == 2].room_cost.median()

m3 = train[train['rooms'] == 3].room_cost.median()

m3half = train[train['rooms'] == 3.5].room_cost.median()

m4 = train[train['rooms'] == 4].room_cost.median()

m4half = train[train['rooms'] == 4.5].room_cost.median()

m5 = train[train['rooms'] == 5].room_cost.median()

m5half = train[train['rooms'] == 5.5].room_cost.median()

m6 = train[train['rooms'] == 6].room_cost.median()

m6half = train[train['rooms'] == 6.5].room_cost.median()

m7 = train[train['rooms'] == 7].room_cost.median()

m7half = train[train['rooms'] == 7.5].room_cost.median()

m8 = train[train['rooms'] == 8].room_cost.median()

m8half = train[train['rooms'] == 8.5].room_cost.median()

m9 = train[train['rooms'] == 9].room_cost.median()

m9half = train[train['rooms'] == 9.5].room_cost.median()

m10 = train[train['rooms'] == 10].room_cost.median()

m10half = train[train['rooms'] == 10.5].room_cost.median()

m11 = train[train['rooms'] == 11].room_cost.median()

m11half = train[train['rooms'] == 11.5].room_cost.median()

m12 = train[train['rooms'] == 12].room_cost.median()

m12half = train[train['rooms'] == 12.5].room_cost.median()

m13 = train[train['rooms'] == 13].room_cost.median()

m13half = train[train['rooms'] == 13.5].room_cost.median()

m14 = train[train['rooms'] == 14].room_cost.median()

m14half = train[train['rooms'] == 14.5].room_cost.median()

m15 = train[train['rooms'] == 15].room_cost.median()

m15half = train[train['rooms'] == 15.5].room_cost.median()



train['underpriced'] = 0



train['underpriced'][train['rooms'] == 1.0][train['room_cost'] < m1] = 1

train['underpriced'][train['rooms'] == 1.5][train['room_cost'] < m1half] = 1

train['underpriced'][train['rooms'] == 2.0][train['room_cost'] < m2] = 1

train['underpriced'][train['rooms'] == 2.5][train['room_cost'] < m2half] = 1

print(train['underpriced'])

print(train['rooms'])
