import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv', index_col='row_id')
plt.rcParams['figure.figsize'] = (12,10)
train.sample(n=1000000).plot(x='x', y='y', kind='hexbin', gridsize=100)
plt.title('Density of checkins')
plt.xlabel('x')
train.x.hist(bins=1000)
g = train.groupby('place_id')
place = g.mean()

place['counts'] = g.x.count()

std = g.std()
mean = g.mean()
place['rg'] = np.sqrt(std.x**2 + std.y**2)
place['area'] = std.x * std.y
plt.rcParams['figure.figsize'] = (16,4)
place.rg.hist(bins=200, log=True);
plt.xlabel('Radius of Gyration')
place.area.hist(bins=200, log=True);
plt.xlabel('Area')
min_count = 1500
print(place.rg[place.counts[place.counts > min_count].index].max())
print(place.rg[place.counts[place.counts > min_count].index].min())
largest_place = place.rg[place.counts[place.counts > min_count].index].argmax()
smallest_place = place.rg[place.counts[place.counts > min_count].index].argmin()

plt.rcParams['figure.figsize'] = (4,4)

train.ix[(train.place_id == largest_place),:].plot(x='x', y='y', kind='scatter')
plt.title('largest rg')
plt.xlim(0,10)
plt.ylim(0,10);
train.ix[(train.place_id == smallest_place),: ].plot(x='x', y='y', kind='scatter')
plt.title('smallest rg')
plt.xlim(0,10)
plt.ylim(0,10);
min_count = 1000
print(place.area[place.counts[place.counts > min_count].index].max())
print(place.area[place.counts[place.counts > min_count].index].min())
largest_area = place.area[place.counts[place.counts > min_count].index].argmax()
smallest_area = place.area[place.counts[place.counts > min_count].index].argmin()
plt.rcParams['figure.figsize'] = (4,4)
train[train.place_id == largest_area].plot(x='x', y='y', kind='scatter')
plt.title('largest area')

plt.xlim(0,10)
plt.ylim(0,10);
plt.rcParams['figure.figsize'] = (4,4)
train[train.place_id == smallest_area].plot(x='x', y='y', kind='scatter')
plt.title('smallest area')

plt.xlim(0,10)
plt.ylim(0,10);
std.x.mean()/std.y.mean()