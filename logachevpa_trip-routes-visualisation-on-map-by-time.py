from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/CmynXBLlCaY" frameborder="0" allowfullscreen></iframe>')
%matplotlib inline

import pandas as pd

import numpy as np

import requests

from tqdm import tqdm

import matplotlib.pyplot as plt

import matplotlib

import json

import matplotlib as mpl

plt.rcParams.update(plt.rcParamsDefault)

params = {

    'axes.labelsize': 12,  # fontsize for x and y labels (was 10)

    'axes.titlesize': 8,

    'font.size': 12,

    'legend.fontsize': 8, 

    'xtick.labelsize': 10,

    'ytick.labelsize': 10,

    'figure.figsize': [30, 10],

    'font.family': 'serif'

}

matplotlib.rcParams.update(params)

from pymongo import MongoClient

client = MongoClient()
test = pd.read_csv('data/test.csv')

train = pd.read_csv('data/train.csv')
def extract_features(A):

    A['pickup_datetime'] = pd.to_datetime(A['pickup_datetime'])

    return A
test = extract_features(test)

train = extract_features(train)
test.head()
import grequests



def make_url(r):

    url = 'http://0.0.0.0:5000/route/v1/driving/{},{};{},{}?annotations=true&alternatives=false&steps=true&overview=full&geometries=geojson'

    return url.format(r['pickup_longitude'], r['pickup_latitude'], 

                      r['dropoff_longitude'], r['dropoff_latitude'])



def response_processing(r, response):

    try:

        data = response.json()

        out = {}

        coords = data['routes'][0]['geometry']['coordinates']

        r['segments'] = []

        for idx, coord in enumerate(coords[:-1]):

            r['segments'].append([coord, coords[idx+1]])



        r['time'] = data['routes'][0]['legs'][0]['annotation']['duration']

    #     out['time'] = [0.001+t*sum_real_time/sum_time for t in time]

        r['sum_time'] = data['routes'][0]['duration']+0.0001



        r['distance'] = data['routes'][0]['legs'][0]['annotation']['distance']

        

    except:

        print('error processing response')

    return r



mg = client.pymongo_test.routes

#mg.delete_many({})

mg.count()

chunk_size = 10**2

for chunk in tqdm(range(int(len(train)/chunk_size))):

    df_chunk = train.iloc[chunk_size*chunk:chunk_size*(chunk+1)].reset_index().to_dict(orient='records')

    requests = [grequests.get(make_url(r)) for r in df_chunk]

    out = [response_processing(df_chunk[idx], response) 

           for idx, response in enumerate(grequests.map(requests))]

    mg.insert_many(out)

mg.create_index([('id', 1)])

mg.count()
lon_min, lon_max, lat_min, lat_max = (-74.017236999999994,

 -73.87130999999998,

 40.699087999999999,

 40.784724999999999)
def merge_segments(df):

    adf = []

    for i in df.to_dict(orient='records'):

        try:

            for idx, segment in enumerate(i['segments']):

                s = {'id': i['id'], 'distance': i['distance'][idx], 'segment': segment, 

                     'velocity': sum(i['distance']) / i['trip_duration'],

                    'velocity_osrm': i['distance'][idx] / (0.001 + i['time'][idx] * i['trip_duration'] / i['sum_time'])}

                adf.append(s)

        except:

            print('error')

    adf = pd.DataFrame(adf)

    adf['lon0'] = adf.segment.apply(lambda x: x[0][0])

    adf['lon1'] = adf.segment.apply(lambda x: x[1][0])

    adf = adf[adf.velocity_osrm < 50]



    adf_gr = adf.groupby(['lon0', 'lon1'], as_index=False)['velocity_osrm'].agg(['mean', 'count']).add_suffix('_velocity').reset_index()

    adf_gr = adf_gr[adf_gr.count_velocity>1].merge(adf.drop_duplicates(['lon0', 'lon1']), on=['lon0', 'lon1'], how='left')

    return adf_gr
cmap = mpl.cm.RdYlGn

norm = mpl.colors.Normalize(vmin=0, vmax=20)

def plot(adfm, lon_min, lon_max, lat_min, lat_max, output_path, text_on_image):

    adfm_out = adfm.copy()

    plt.gcf().clear()

    fig, ax = pl.subplots()

    lc = mc.LineCollection(adfm_out['segment'].values, colors=cmap(norm((adfm_out['mean_velocity']))), 

                           alpha=0.9, linewidths=adfm_out['count_velocity']/65)

    ax.add_collection(lc)

    ax.text(0.95, 0.01, text_on_image,

        verticalalignment='bottom', horizontalalignment='right',

        transform=ax.transAxes,

        color='white', fontsize=20)

    plt.xlim(lon_min, lon_max)

    plt.ylim(lat_min, lat_max)

    plt.imshow(image, zorder=0, extent=[lon_min, lon_max, lat_min, lat_max])

    plt.axis('off') 

    plt.savefig(output_path, dpi=400,bbox_inches='tight', pad_inches=0)
file_paths = 'output/{}.png'

for hour in range(24):

    for minute in range(6):

        filtered_df = train_df[(train_df['pickup_datetime'].dt.minute > 10*minute) & 

           (train_df['pickup_datetime'].dt.minute < 10*(minute+1)) & 

           (train_df['pickup_datetime'].dt.hour == hour)]

        if len(filtered_df) > 0:

            print(len(filtered_df))

            print(hour, minute)

            df = pd.DataFrame(list(mg.find({'id': {'$in': filtered_df['id'].values.tolist()}})))

            plot(merge_segments(df), lon_min, lon_max, lat_min, lat_max, 

                 file_paths.format(5*hour+minute), '{}:{}'.format(hour, minute*10))