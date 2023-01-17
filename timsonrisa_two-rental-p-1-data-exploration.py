import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
%matplotlib inline
train = pd.read_json('../input/train.json')
test  = pd.read_json('../input/test.json')
print(f'Training set is of size:{train.shape}')
print(f'Test set is of size:{test.shape}')
train.head(1)
def PlotNormHist(data, axes, binaryFeat):
    param = [False, True]
    for i,cur_ax in enumerate(axes):
        cur_data = data[data[binaryFeat]==param[i]]
        int_level = cur_data['interest_level'].value_counts()
        int_level = int_level/sum(int_level)
        sns.barplot(int_level.index, int_level.values, alpha=0.8,
                    order=['low','medium','high'], ax=cur_ax)
        cur_ax.set_xlabel(param[i], fontsize=15)
        cur_ax.set_ylim(bottom=0, top=1)
        cur_ax.grid()
train['nPhotos'] = train['photos'].apply(lambda x: min(10, len(x)))
plt.figure(figsize=(10,5))
sns.violinplot(x='interest_level', y='nPhotos', data=train, order=['low','medium','high'])
plt.xlabel('# Interest Level', fontsize=12)
plt.ylabel('# of Photos', fontsize=12)
plt.grid()
plt.show()
train['hasDesc'] = train['description'].apply(lambda x: len(x.strip())!=0)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
PlotNormHist(train, axes, 'hasDesc')
long_llim = np.percentile(train.longitude.values, 1)
long_ulimit = np.percentile(train.longitude.values, 99)
lat_llim = np.percentile(train.latitude.values, 1)
lat_ulimit = np.percentile(train.latitude.values, 99)
train = train[(train['longitude']>long_llim) & (train['longitude']<long_ulimit) & 
              (train['latitude']>lat_llim) & (train['latitude']<lat_ulimit)]
lats = list(train['latitude'])
lons = list(train['longitude'])
fig = plt.figure(figsize=(15, 15))
m = Basemap(projection='merc',llcrnrlat=min(lats),urcrnrlat=max(lats),\
            llcrnrlon=min(lons),urcrnrlon=max(lons), resolution='h')
x, y = m(lons,lats)
sns.scatterplot(x, y, hue=train['interest_level'], style=train['interest_level'])
ulimit = np.percentile(train.price.values, 99)
train['price'][train['price']>ulimit] = ulimit

plt.figure(figsize=(8,6))
sns.distplot(train.price.values, bins=50, kde=True)
plt.xlabel('price', fontsize=12)
plt.grid()
plt.show()
plt.figure(figsize=(10,5))
sns.violinplot(x='interest_level', y='price', data=train, order=['low','medium','high'])
plt.xlabel('# Interest Level', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.grid()
plt.show()
feat_dict = {}
for ind, row in train.iterrows():
    for f in row['features']:
        f = f.lower().replace('-', '')
        if f in feat_dict:
            feat_dict[f] += 1
        else:
            feat_dict[f] = 1 
new_feat_dict = {}
for k,v in feat_dict.items():
    if v>50: new_feat_dict[k] = v  
new_feat_dict.keys()
def CreateCategFeat(data, features_list):
    f_dict = {'hasParking':['parking', 'garage'], 'hasGym':['gym', 'fitness', 'health club'],
              'hasPool':['swimming pool', 'pool'], 'noFee':['no fee', "no broker's fees"],
              'hasElevator':['elevator'], 'hasGarden':['garden', 'patio', 'outdoor space'],
              'isFurnished': ['furnished', 'fully  equipped'], 
              'reducedFee':['reduced fee', 'low fee'],
              'hasAC':['air conditioning', 'central a/c', 'a/c', 'central air', 'central ac'],
              'hasRoof':['roof', 'sundeck', 'private deck', 'deck'],
              'petFriendly':['pets allowed', 'pet friendly', 'dogs allowed', 'cats allowed'],
              'shareable':['shares ok'], 'freeMonth':['month free'],
              'utilIncluded':['utilities included']}
    for feature in features_list:
        data[feature] = False
        for ind, row in data.iterrows():
            for f in row['features']:
                f = f.lower().replace('-', '')
                if any(e in f for e in f_dict[feature]):
                    data.at[ind, feature]= True     
cat_features = ['hasParking', 'hasGym', 'hasPool', 'noFee', 'hasElevator',
                'hasGarden', 'isFurnished', 'reducedFee', 'hasAC', 'hasRoof',
                'petFriendly', 'shareable', 'freeMonth', 'utilIncluded']
CreateCategFeat(train, cat_features)
for cur_feature in cat_features:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,3))
    PlotNormHist(train, axes, cur_feature)
    fig.suptitle(cur_feature, fontsize=16)
import datetime
train['created'] = pd.to_datetime(train['created'])
train['month']   = train['created'].dt.month
plt.figure(figsize=(8,6))
sns.countplot(x='month', hue='interest_level', data=train, hue_order=['low','medium','high'])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('month', fontsize=12)
plt.grid()
train['weekday'] = train['created'].apply(lambda x: x.weekday())
plt.figure(figsize=(8,6))
sns.countplot(x='weekday', hue='interest_level', data=train, hue_order=['low','medium','high'])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Weekday', fontsize=12)
plt.grid()