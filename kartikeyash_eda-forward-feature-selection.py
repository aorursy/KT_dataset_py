# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import warnings

warnings.filterwarnings('ignore')

import json

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 

import pandas_profiling as pp

import plotly.express as px

from collections import defaultdict

import lightgbm as lgb

from sklearn.neighbors import NearestNeighbors

from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv')

df.head()
df_test = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv')

df_test.head()
def draw_train_test_intersections(city):

    df_draw = df[df['City'] == city]

    df_draw = df_draw[['Latitude','Longitude']]

    df_draw.drop_duplicates(inplace=True)

    df_draw1 = df_test[df_test['City'] == city]

    df_draw1 = df_draw1[['Latitude', 'Longitude']]



    df_draw1.drop_duplicates(inplace=True)

    df_common = pd.merge(df_draw, df_draw1, how='inner')

    df_common.drop_duplicates(inplace=True)

    df_draw['Intersection'] = 'Train'

    df_draw1['Intersection'] = 'Test'

    df_common['Intersection'] = 'Common'

    df_draw = pd.concat([df_draw, df_draw1, df_common])

    fig = px.scatter_mapbox(df_draw, lat="Latitude", lon="Longitude", color="Intersection", zoom=10, opacity=0.7)

    fig.update_layout(

        mapbox_style="stamen-terrain",

        margin={"r":0,"t":0,"l":0,"b":0},

    )

    return fig
for city in df.City.unique():

    print(city)

    fig = draw_train_test_intersections(city)

    fig.show()
c_names = ['pink', 'red', 'green', 'blue', 'black', 'white', 'brown', 'aqua', 'yellow', 'purple']

for city in df['City'].unique():

    r = df[df['City'] == city][['Latitude', 'Longitude']].drop_duplicates().sample(10, random_state=2)

    r_la, r_lo = r['Latitude'].to_list(), r['Longitude'].to_list()

    df_temp = df[df['Latitude'].isin(r_la) & df['Longitude'].isin(r_lo)]

    temp = df_temp.groupby(['Latitude', 'Longitude', 'Hour'])['TotalTimeStopped_p80'].transform(lambda x: x.mean())

    df_temp['agg_tts'] = temp

    fig, axs = plt.subplots(figsize=(16,8))

    intersection_names = []

    for i in range(10):

        t = df_temp[(df_temp['Latitude'] == r_la[i]) & (df_temp['Longitude'] == r_lo[i])][['Hour', 'agg_tts', 'EntryStreetName', 'ExitStreetName']].sort_values('Hour')

        x, y = t['Hour'].to_list(), t['agg_tts'].to_list()

        en_sn, ex_sn = t['EntryStreetName'].to_list()[0], t['ExitStreetName'].to_list()[0]

        intersection_names.append('{0} -> {1}'.format(en_sn, ex_sn))

        axs = sns.lineplot(x=x, y=y, ax=axs, palette=c_names)

    axs.set_xlabel('Hour')

    axs.set_ylabel('Total time stopped p_80')

    axs.set_title(city, fontsize=24)

    axs.legend(intersection_names)
road_encoding = {

    "Street":"Street",

    "St":"Street",

    "Avenue":"Avenue",

    "Ave":"Avenue",

    "Boulevard":"Boulevard",

    "Road":"Road",

    "Drive":"Drive",

    "Lane":"Lane",

    "Tunnel":"Tunnel",

    "Highway":"Highway",

    "Way":"Way",

    "Parkway":"Parkway",

    "Parking":"Parking",

    "Oval":"Oval",

    "Square":"Square",

    "Place":"Place",

    "Bridge":"Bridge",

}



def encode(x):

    if pd.isna(x):

        return "Street"

    for road in road_encoding.keys():

        if road in x:

            return road_encoding[road]

    return "Street"



for city in df['City'].unique():

    df_temp = df[df['City'] == city]

    df_temp['StreetType'] = df_temp['EntryStreetName'].apply(encode)

    temp = df_temp.sort_values('StreetType').groupby(by=['Hour', 'StreetType'])['TotalTimeStopped_p80'].transform(lambda x: x.mean())

    df_temp['avg_time'] = temp

    fig, axs = plt.subplots(figsize=(16,8))

    axs = sns.barplot(x='StreetType', y='avg_time', data=df_temp, ax=axs, ci=None)

    axs.set_xlabel('StreetType')

    axs.set_ylabel('Total time stopped p_80')

    axs.set_title(city, fontsize=24)
# lon_lat_train = df[['Longitude', 'Latitude']].drop_duplicates()

# lon_lat_test = df_test[['Longitude', 'Latitude']].drop_duplicates()

# lon_lat = pd.concat([lon_lat_train, lon_lat_test])

# lon_lat.drop_duplicates(inplace=True)



lon_lat = df[['Longitude', 'Latitude']].drop_duplicates()



neigh = NearestNeighbors(5) # 5 because first one will be the same number itself



lon_lat_array = lon_lat.to_numpy()



neigh.fit(lon_lat_array)



lon_lat_train = df[['Longitude', 'Latitude']].drop_duplicates()

lon_lat_test = df_test[['Longitude', 'Latitude']].drop_duplicates()

lon_lat = pd.concat([lon_lat_train, lon_lat_test])

lon_lat.drop_duplicates(inplace=True)

lon_lat_array = lon_lat.to_numpy()



nearest_points = defaultdict(list)

for lo, la in lon_lat_array:

    nearest_points['{}-{}'.format(lo, la)] = list(neigh.kneighbors([[lo, la]], 5, return_distance=False)[0][1:])



nearest_points_val = defaultdict(list)

for k, pt in nearest_points.items():

    nearest_points_val[k] = [(lon_lat_array[i][0],lon_lat_array[i][1]) for i in pt]



nearest_points_df = defaultdict(list)

for lo, la in zip(df['Longitude'].to_list(), df['Latitude'].to_list()):

    for i in range(4):

        nearest_points_df["nplo{}".format(i)].append(nearest_points_val["{}-{}".format(lo,la)][i][0])

        nearest_points_df["npla{}".format(i)].append(nearest_points_val["{}-{}".format(lo,la)][i][1])



nearest_points_df = pd.DataFrame(nearest_points_df)

df = pd.concat([df, nearest_points_df], axis=1)



df.head()
nearest_points_df = defaultdict(list)

for lo, la in zip(df_test['Longitude'].to_list(), df_test['Latitude'].to_list()):

    for i in range(4):

        nearest_points_df["nplo{}".format(i)].append(nearest_points_val["{}-{}".format(lo,la)][i][0])

        nearest_points_df["npla{}".format(i)].append(nearest_points_val["{}-{}".format(lo,la)][i][1])



nearest_points_df = pd.DataFrame(nearest_points_df)

df_test = pd.concat([df_test, nearest_points_df], axis=1)

df_test.head()
df = df[['Latitude', 'Longitude', 'EntryHeading', 'ExitHeading', 'Hour', 'Weekend',

       'Month', 'TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80',

       'TimeFromFirstStop_p20', 'TimeFromFirstStop_p40',

       'TimeFromFirstStop_p50', 'TimeFromFirstStop_p60',

       'TimeFromFirstStop_p80', 'DistanceToFirstStop_p20',

       'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80', 'City', 'nplo0',

       'npla0', 'nplo1', 'npla1', 'nplo2', 'npla2', 'nplo3', 'npla3']]



df_test = df_test[['Latitude', 'Longitude', 'EntryHeading', 'ExitHeading', 'Hour', 'Weekend',

       'Month', 'City', 'nplo0',

       'npla0', 'nplo1', 'npla1', 'nplo2', 'npla2', 'nplo3', 'npla3']]
directions = {

    'N': 0,

    'NE': 1/4,

    'E': 1/2,

    'SE': 3/4,

    'S': 1,

    'SW': 5/4,

    'W': 3/2,

    'NW': 7/4

}



df['EntryHeading'] = df['EntryHeading'].map(directions)

df['ExitHeading'] = df['ExitHeading'].map(directions)



df_test['EntryHeading'] = df_test['EntryHeading'].map(directions)

df_test['ExitHeading'] = df_test['ExitHeading'].map(directions)
metric_map = json.load(open('/kaggle/input/bigquery-geotab-intersection-congestion/submission_metric_map.json'))

print(metric_map)
for column in df_test.columns:

    if column == 'City':

        continue

    temp = pd.concat([df[column], df_test[column]])

    mx = float(max(temp))

    mn = float(min(temp))

    df[column] = df[column].apply(lambda x: (x-mn)/(mx-mn))

    df_test[column] = df_test[column].apply(lambda x: (x-mn)/(mx-mn))
df.head()
df_test.head()
cols = ['Latitude', 'Longitude', 'EntryHeading', 'ExitHeading', 'Hour', 'Weekend', 'Month', 'nplo0',

       'npla0', 'nplo1', 'npla1', 'nplo2', 'npla2', 'nplo3', 'npla3']



# Thanks to https://www.kaggle.com/ragnar123/feature-engineering-and-forward-feature-selection (Source 2)

param = {

    'application': 'regression', 

    'learning_rate': 0.05, 

    'metric': 'rmse', 

    'seed': 42, 

    'bagging_fraction': 0.7, 

    'feature_fraction': 0.9, 

    'lambda_l1': 0.0, 

    'lambda_l2': 5.0, 

    'max_depth': 30, 

    'min_child_weight': 50.0, 

    'min_split_gain': 0.1, 

    'num_leaves': 230

}



nfold = 5

all_preds = {0 : {}, 1 : {}, 2 : {}, 3 : {}, 4 : {}, 5 : {}}



for metric_id, metric in metric_map.items():

    for city in df.City.unique():

        train = df[df['City'] == city]

        labels = train[metric]

        mx = float(max(labels))

        mn = float(min(labels))

        labels = labels.apply(lambda x: (x-mn)/(mx-mn))



        train = train[cols]



        test = df_test[df_test['City'] == city]

        test = test[cols]

        test_idx = test.index



        kf = KFold(n_splits=nfold, random_state=1111, shuffle=True)

        print('Training and predicting for target {}, {}, {}'.format(city, metric_id, metric))



        oof = np.zeros(len(train))

        preds = np.zeros(len(test))



        for train_index, valid_index in kf.split(train):

            xg_train = lgb.Dataset(train.iloc[train_index],

                                   label=labels.iloc[train_index]

                                   )

            xg_valid = lgb.Dataset(train.iloc[valid_index],

                                   label=labels.iloc[valid_index]

                                   )



            clf = lgb.train(param, xg_train, 100000, valid_sets=[xg_train, xg_valid], 

                            verbose_eval=500, early_stopping_rounds=100)

            oof[valid_index] = clf.predict(train.iloc[valid_index], num_iteration=clf.best_iteration) 



            preds += clf.predict(test, num_iteration=clf.best_iteration) / nfold

            break

        preds = [((mx-mn) * p) + mn for p in preds]

        all_preds[int(metric_id)].update({idx:val for idx, val in zip(test_idx, preds)})

        print("\n\nCV RMSE: {:<0.4f}".format(np.sqrt(mean_squared_error(labels, oof))))
final_preds = {0 : [], 1 : [], 2 : [], 3 : [], 4 : [], 5 : []}

for metric in final_preds.keys():

    final_preds[metric] = [all_preds[metric][i] for i in sorted(all_preds[metric].keys())]

print(df_test.shape, len(final_preds[0]))
submission = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/sample_submission.csv')

data2 = pd.DataFrame(final_preds).stack()

data2 = pd.DataFrame(data2)

submission['Target'] = data2[0].values

submission.to_csv('lgbm.csv', index=False)