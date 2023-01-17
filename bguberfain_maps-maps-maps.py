import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from matplotlib import cm



%matplotlib inline
train_df = pd.read_json("../input/train.json")



west, south, east, north = -74.02, 40.64, -73.85, 40.86



interest_map = {'low': 0, 'medium': 1, 'high': 2}



train_df['interest_score'] = train_df['interest_level'].map(interest_map)



train_df['num_phots'] = train_df['photos'].map(len)



train_df['feat_elevator'] = train_df['features'].map(lambda x: 'Elevator' in x)

train_df['feat_animals_allowed'] = train_df['features'].map(lambda x: ('Cats Allowed' in x) or ('Dogs Allowed' in x))

train_df['feat_hardwood_floor'] = train_df['features'].map(lambda x: ('Hardwood Floors' in x) or ('HARDWOOD' in x))

train_df['feat_doorman'] = train_df['features'].map(lambda x: 'Doorman' in x)

train_df['feat_dishwasher'] = train_df['features'].map(lambda x: 'Dishwasher' in x)

train_df['feat_no_fee'] = train_df['features'].map(lambda x: 'No Fee' in x)

train_df['feat_laundry'] = train_df['features'].map(lambda x: ('Laundry in Building' in x) or ('Laundry in Unit' in x))

train_df['feat_fit_center'] = train_df['features'].map(lambda x: 'Fitness Center' in x)

train_df['feat_pre_war'] = train_df['features'].map(lambda x: ('Pre-War' in x) or ('prewar' in x))

train_df['feat_roof_deck'] = train_df['features'].map(lambda x: 'Roof Deck' in x)

train_df['feat_outdoor_space'] = train_df['features'].map(lambda x: ('Outdoor Space' in x) or ('Common Outdoor Space' in x))

train_df['feat_pool'] = train_df['features'].map(lambda x: 'Swimming Pool' in x)

train_df['feat_new_construction'] = train_df['features'].map(lambda x: 'New Construction' in x)

train_df['feat_terrace'] = train_df['features'].map(lambda x: 'Terrace' in x)

train_df['feat_loft'] = train_df['features'].map(lambda x: 'Loft' in x)



train_df = train_df[(train_df['longitude'] > west) & (train_df['longitude'] < east) & (train_df['latitude'] > south) & (train_df['latitude'] < north)]

def plot_column(fig, ax, column, gridsize=10, scale=None):

    m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north, ax=ax,

            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')

    x, y = m(train_df['longitude'].values, train_df['latitude'].values)

    

    if column is None:

        ax.set_title("count")

        hb = m.hexbin(x, y, gridsize=gridsize, bins=scale, cmap=cm.hot)

    else:

        ax.set_title(column)

        hb = m.hexbin(x, y, C=train_df[column].values, gridsize=gridsize, bins=scale, cmap=cm.hot)

    cb = fig.colorbar(hb, ax=ax)
fig, ax = plt.subplots(2, 3, figsize=(14,10))

ax = ax.ravel()



plot_column(fig, ax[0], None, 50, 'log')

plot_column(fig, ax[1], 'bathrooms')

plot_column(fig, ax[2], 'bedrooms')

plot_column(fig, ax[3], 'price')

plot_column(fig, ax[4], 'num_phots')

plot_column(fig, ax[5], 'interest_score')

features = ['feat_elevator', 'feat_animals_allowed', 'feat_hardwood_floor', 'feat_doorman',

            'feat_dishwasher', 'feat_no_fee', 'feat_laundry', 'feat_fit_center', 'feat_pre_war',

            'feat_roof_deck', 'feat_outdoor_space', 'feat_pool', 'feat_new_construction',

            'feat_terrace', 'feat_loft']



fig, ax = plt.subplots(5, 3, figsize=(14, 20))

ax = ax.ravel()



for axis, feat in zip(ax, features):

    plot_column(fig, axis, feat)