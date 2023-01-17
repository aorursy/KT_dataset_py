!cat /kaggle/input/odfinvasivespecies1/invasive-species-workshop-1/requirements.txt | xargs -n 1 pip install
!curl -sL https://deb.nodesource.com/setup_13.x | bash -

!apt-get install -y nodejs
# Set up the Kepler GL widget to display maps inline in the notebooks

!jupyter nbextension install --py --sys-prefix keplergl

!jupyter nbextension enable --py --sys-prefix keplergl
# Render test for Kepler - if you don't see a map, please refresh this page. 

from keplergl import KeplerGl

map_1 = KeplerGl()

map_1
from pathlib import Path

kaggle_path = Path('/kaggle/input/odfinvasivespecies1/invasive-species-workshop-1')
import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")

    

from IPython.display import Image

from IPython.core.display import HTML

from IPython.display import IFrame



%matplotlib inline
# Imports

import numpy as np

import pandas as pd

import geopandas as gpd

import json

import swifter
# Load all the data we have



## Presence

pres_df = pd.read_csv(str(Path(kaggle_path, 'Presence/records.csv')),

                      delimiter='\t')[['decimalLatitude', 'decimalLongitude', 

                                       'eventDate']].dropna().set_index('eventDate')

pres_df = pres_df.sort_index().reset_index()



## Port data

baltic_ports = pd.read_csv(Path(kaggle_path,'baltic_ports.txt'), sep=' ', index_col=0)



## River data

hydro = gpd.GeoDataFrame.from_file(Path(kaggle_path,'Europe_Hydrography/Europe_Hydrography.shp')).to_json()
from keplergl import KeplerGl

map_1 = KeplerGl(width = 800, height=600)

map_1.add_data(pres_df, name='pres')

map_1
from keplergl import KeplerGl

map_1 = KeplerGl(width = 800, height=600)

map_1.add_data(pres_df, name='presence')

map_1.add_data(baltic_ports, name='absence')

map_1.add_data(hydro, name='rivers')

map_1
all_ports = [[feature['geometry']['coordinates'][1], feature['geometry']['coordinates'][0]] for \

             feature in json.load(open(Path(kaggle_path, 'ports.json')))['features']]
from sklearn.neighbors import KNeighborsClassifier
def get_nearest_station(array, w_stations):

    knn = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=1, metric='euclidean')

    X = np.array([[i[0], i[1]] for i in w_stations])

    y = np.arange(len(w_stations))

    knn.fit(X, y)

    ids = knn.predict(array)

    return ids
nsoc_data = pd.read_csv(Path(kaggle_path, 'north_sea_temp_salinity_historical.txt'), sep=' ')

nsoc_data['coordinates'] = list(zip(nsoc_data.lat, nsoc_data.lon))

nsoc_data = nsoc_data[(nsoc_data.temp > -100) & (nsoc_data.salinity > -100)]
pres_df['coordinates'] = list(zip(pres_df.decimalLatitude, pres_df.decimalLongitude))

pres_df['reference_port'] = pres_df.swifter.apply(lambda x: \

                                          all_ports[get_nearest_station(

                                          np.array(x['coordinates']).reshape(1, -1), all_ports)[0]], 1)

pres_df[['ref_port_lat', 'ref_port_lon']] = pd.DataFrame(pres_df['reference_port'].tolist(), index=pres_df.index)    



pres_df[['temp', 'salinity']] = pres_df.swifter.apply(lambda x: \

                                nsoc_data[nsoc_data.date.str[:7] == x['eventDate'][:7]].iloc[get_nearest_station(

                                np.array(

                                x['reference_port']).reshape(1, -1),

                                nsoc_data[nsoc_data.date.str[:7] == x['eventDate'][:7]]['coordinates'])[0]] \

                                [['temp', 'salinity']] 

                                if len(nsoc_data[nsoc_data.date.str[:7] == x['eventDate'][:7]]) > 0 else None, 1)
# export for fast import

#pres_df.to_csv('pres_df.csv')
pres_df = pd.read_csv(Path(kaggle_path, 'pres_df.csv'))
np_ports = pd.read_csv(Path(kaggle_path,'baltic_ports.txt'), sep=' ', index_col=0)

baltic_temp = pd.read_csv(

                Path(kaggle_path,'baltic_temperatures_latest.txt'), sep='\t', encoding='cp1252')[['Provtagningsdatum', 

                                                                                'Provets latitud (DD)',

                                                                                'Provets longitud (DD)',

                                                                                'Mätvärde']].dropna()

baltic_sal = pd.read_csv(

                Path(kaggle_path,'baltic_salinity_latest.txt'), sep='\t', encoding='cp1252')[['Provtagningsdatum',

                                                                            'Provets latitud (DD)',

                                                                            'Provets longitud (DD)',

                                                                            'Mätvärde']].dropna()
baltic_temp = baltic_temp[baltic_temp.Provtagningsdatum.str[5:7].isin(['12', '01', '02'])]

baltic_sal = baltic_sal[baltic_sal.Provtagningsdatum.str[5:7].isin(['12', '01', '02'])]
baltic_temp.columns = ['date', 'lat', 'lon', 'temperature']

baltic_sal.columns = ['date', 'lat', 'lon', 'salinity']
baltic_temp[['lat', 'lon', 'temperature']] = baltic_temp[['lat', 'lon', 'temperature']].apply(

                                             lambda x: x.str.replace(',', '.')).astype(float)

baltic_sal[['lat', 'lon', 'salinity']] = baltic_sal[['lat', 'lon', 'salinity']].apply(

                                            lambda x: x.str.replace(',', '.')).astype(float)
baltic_temp = baltic_temp.groupby(by=['lat','lon']).mean().reset_index()

baltic_sal = baltic_sal.groupby(by=['lat','lon']).mean().reset_index()
np_ports['temp'] = [baltic_temp.iloc[i]['temperature'] for 

                    i in get_nearest_station([i[0] for i in zip(

                    np_ports[['lat', 'lon']].values)], 

                    [i[0] for i in zip(baltic_temp[['lat', 'lon']].values)])]

np_ports['salinity'] = [baltic_sal.iloc[i]['salinity'] for i in get_nearest_station([i[0] for i in zip(

                    np_ports[['lat', 'lon']].values)], 

                    [i[0] for i in zip(baltic_sal[['lat', 'lon']].values)])]
p_plot = pres_df[['eventDate', 'ref_port_lat', 'ref_port_lon', 'temp', 'salinity']].iloc[1:]

p_plot.columns = ['date', 'lat', 'lon', 'temp', 'salinity']

p_plot = p_plot[p_plot.date.str[5:7].isin(['10', '11', '12', '01', '02'])]

p_plot = p_plot.groupby(by=['lat','lon']).mean().reset_index()

p_plot['obs'] = 1
np_ports['obs'] = 0
plot_df = pd.concat([p_plot[['lat', 'lon', 'temp', 'salinity', 'obs']], 

                     np_ports[['lat', 'lon', 'temp', 'salinity', 'obs']]])
plot_df = plot_df.dropna()
map_2 = KeplerGl(width = 800, height=600)

map_2.add_data(plot_df, name='o')

map_2
from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
X = plot_df[['lat', 'lon', 'temp', 'salinity']]

y = plot_df['obs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777, shuffle=False)
clf = tree.DecisionTreeClassifier(max_depth=2)
clf = clf.fit(X_train[['temp', 'salinity']], y_train)
y.value_counts()
accuracy_score(y_test, clf.predict(X_test[['temp', 'salinity']]))
plt.figure(figsize=(15,10))

tree.plot_tree(clf.fit(X_train[['temp', 'salinity']], y_train)) 
IFrame(src='https://en.wikipedia.org/wiki/Dikerogammarus_villosus', width=1000, height=400)
box_df = plot_df.iloc[:,2:]

box_df['obs'] = box_df['obs'].astype(str)

box_df['temp'] = box_df['temp'].astype(float)

box_df['salinity'] = box_df['salinity'].astype(float)

box_df.boxplot(column=['temp', 'salinity'], by='obs', figsize=(15,5))
from pyimpute import load_training_vector, load_targets, impute, evaluate_clf

from sklearn.ensemble import RandomForestClassifier
post_pres_df = pd.read_csv(Path(kaggle_path, 'pres_df.csv'))
post_pres_df = post_pres_df[['decimalLatitude', 'decimalLongitude']]
post_pres_df['obs'] = 1
np_ports[['decimalLatitude', 'decimalLongitude']] = np_ports[['lat', 'lon']]
np_ports['obs'] = 0
total_df = post_pres_df.append(np_ports[['decimalLatitude', 'decimalLongitude', 'obs']])
total_df.drop_duplicates(['decimalLatitude', 'decimalLongitude', 'obs'], inplace=True)
total_df.to_csv(Path(kaggle_path,'total_df.csv'))
total_df = pd.read_csv(Path(kaggle_path,'total_df.csv'))
total_df = gpd.GeoDataFrame(total_df, 

                                 geometry=gpd.points_from_xy(total_df.decimalLongitude, 

                                                                   total_df.decimalLatitude))
total_df[['obs', 'geometry']].to_file(Path(kaggle_path,"output.json"), driver="GeoJSON")
import rasterstats

import rasterio

from rasterio.plot import show, show_hist, plotting_extent
ls
src = rasterio.open(Path(kaggle_path,'BO/Present.Surface.Salinity.Mean.tif'))

plt.figure(figsize=(15,10))

plt.imshow(src.read(1, masked=True))

plt.colorbar()
src = rasterio.open(Path(kaggle_path,'BO/Present.Surface.Temperature.Mean.tif'))

plt.figure(figsize=(15,10))

plt.imshow(src.read(1, masked=True))

plt.colorbar()
from affine import Affine



xmin = -15

xmax = 50

ymin = 30

ymax = 70





def window_from_extent(xmin, xmax, ymin, ymax, aff):

    col_start, row_start = ~aff * (xmin, ymax)

    col_stop, row_stop = ~aff * (xmax, ymin)

    return ((int(row_start), int(row_stop)), (int(col_start), int(col_stop)))
src = rasterio.open(Path(kaggle_path,'BO/Present.Surface.Temperature.Mean.tif'))
europe_window = window_from_extent(xmin, xmax, ymin, ymax, src.transform)
europe_view = src.read(1, window=europe_window)
europe_transform = src.window_transform(europe_window)
with rasterio.open(Path(kaggle_path,'BO/europe_temp.tif'), #filename

                   'w', # file mode, with 'w' standing for "write"

                   driver='GTiff', # format to write the data

                   height=europe_view.shape[0], # height of the image, often the height of the array

                   width=europe_view.shape[1], # width of the image, often the width of the array

                   count=1, # the number of bands to write

                   dtype='float64',# the dtype of the data, usually `ubyte` if data is stored in integers

                   nodata=src.nodata,

                   crs=src.crs, # the coordinate reference system of the data

                   transform=europe_transform # the affine transformation for the image

                  ) as outfile:

    outfile.write(europe_view, indexes=1) # write the `austin_nightlights` as the first band

europe = rasterio.open(Path(kaggle_path,'BO/europe_temp.tif'))

plt.figure(figsize=(15,10))

plt.imshow(europe.read(1, masked=True))

plt.colorbar()
explanatory_rasters = [Path(kaggle_path,'BO/europe_salinity.tif'),

                       Path(kaggle_path,'BO/europe_temp.tif')]

response_data = Path(kaggle_path,'output.json')



train_xs, train_y = load_training_vector(response_data,

                                         explanatory_rasters,

                                         response_field="obs")
train_xs
valid = np.where(np.sum(train_xs == [None, None], axis=1) != 2)[0]
train_xs = train_xs[valid]

train_y = train_y[valid]
clf = RandomForestClassifier(n_estimators=1000, n_jobs=1, class_weight="balanced")

clf.fit(train_xs.astype('float32'), train_y.astype('float32'))
evaluate_clf(clf, train_xs, train_y)
target_xs, raster_info = load_targets(explanatory_rasters)
import matplotlib.pyplot as plt

import os

%matplotlib inline
def plotit(x, title, cmap='Blues'):

    plt.figure(figsize=(15,10))

    plt.imshow(x, cmap=cmap, interpolation='nearest')

    plt.colorbar()

    plt.title(title)

    plt.show()

impute(target_xs, clf, raster_info, outdir=Path(kaggle_path,'BO'),

        linechunk=1000, class_prob=True, certainty=True)



assert os.path.exists(Path(kaggle_path,"BO/responses.tif"))

assert os.path.exists(Path(kaggle_path,"BO/certainty.tif"))

assert os.path.exists(Path(kaggle_path,"BO/probability_0.tif"))

assert os.path.exists(Path(kaggle_path,"BO/probability_1.tif"))
res = rasterio.open(Path(kaggle_path,'BO/probability_1.tif'))
masking = europe.read(1, masked=True).mask
plotit(np.where(masking, -0.1, res.read(1, masked=True)), 'D. Villosus Climatic Suitability', cmap='GnBu')