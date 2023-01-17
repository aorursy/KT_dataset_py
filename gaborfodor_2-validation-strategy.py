%matplotlib inline

import os

import pandas as pd

import datetime as dt

import numpy as np

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import datetime as dt

from tqdm import tqdm

import matplotlib.pyplot as plt

import seaborn as sns
plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 18

pd.set_option('display.max_columns', 99)

start = dt.datetime.now()



validation_splits = pd.DataFrame([

    ['Atlanta', 33.791, 33.835],

    ['Boston', 42.361, 42.383],

    ['Chicago', 41.921, 41.974],

    ['Philadelphia', 39.999, 40.046],

], columns=['City', 'l1', 'l2'])



train = pd.read_csv(

    '../input/bigquery-geotab-intersection-congestion/train.csv')

test = pd.read_csv('../input/bigquery-geotab-intersection-congestion/test.csv')



train['IsTrain'] = 1

test['IsTrain'] = 0



full = pd.concat([train, test], sort=True)



full = full.merge(validation_splits, on='City')

full['ValidationGroup'] = 1

full.loc[full.Latitude <= full.l1 , 'ValidationGroup'] = 0

full.loc[full.Latitude > full.l2 , 'ValidationGroup'] = 2

full.drop(['l1', 'l2'], axis=1, inplace=True)



m = full.groupby(['City', 'IntersectionId'])[[

    'IsTrain', 'Latitude', 'Longitude']].mean()

c = full.groupby(['City', 'IntersectionId'])[['RowId']].count()

df = pd.merge(m, c, left_index=True, right_index=True)

df = df.rename(columns={'RowId': 'Cnt'})

df = df.reset_index()
for i, city in enumerate(df.City.unique()):

    coords = df[df.City == city]

    fig, ax = plt.subplots()

    sc = ax.scatter(

        y=coords.Latitude,

        x=coords.Longitude,

        c=coords.IsTrain,

        s = 3 * np.sqrt(coords.Cnt),

        alpha=0.7,

        linewidths=0.,

        cmap=plt.cm.RdYlGn

    )

    l1 = validation_splits[validation_splits.City == city].l1.values[0]

    l2 = validation_splits[validation_splits.City == city].l2.values[0]

    plt.plot([coords.Longitude.min(), coords.Longitude.max()],

             [l1, l1], color='orange', lw = 3)

    plt.plot([coords.Longitude.min(), coords.Longitude.max()],

             [l2, l2], color='green', lw = 3)

    plt.grid()

    cbar = plt.colorbar(sc, orientation='horizontal')

    cbar.set_label('Train observations%')

    plt.tight_layout()

    plt.title(f'{city} - train/test split')

    plt.ylabel('Latitude')

    plt.xlabel('Longitude')

    

plt.show();
full.groupby(['IsTrain', 'ValidationGroup'])[['RowId']].count()