# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# https://archive.ics.uci.edu/ml/datasets/Communities%2Band%2BCrime

# Description of this dataset

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



print('Its alive btw')

# Any results you write to the current directory are saved as output.
import chardet

cities = pd.read_json('/kaggle/input/cities.json')

with open('/kaggle/input/crimedata.csv', 'rb') as t:

    res = chardet.detect(t.read())

res
df = pd.read_csv('/kaggle/input/crimedata.csv', encoding=res['encoding'])

df.columns = ['communityname'] + list(df.columns[1:])

df = df.replace(to_replace='?', value=np.nan)
nan_columns = []

for col in df.columns:

    if df[col].isna().sum() > 0.0:

        nan_columns.append(col)

        print('dtype={1} uniques={2} nans_percent={3} {0}'.format(col, df[col].dtype, len(df[col].unique()), 100*df[col].isna().sum()/df.shape[0]))
for col in nan_columns:

    try:

        df.loc[~df[col].isna(), col] = df[~df[col].isna()][col].astype('int')

    except Exception as e:

        df.loc[~df[col].isna(), col] = df[~df[col].isna()][col].astype('float')
# Let's make some magic graphics!



import cartopy.crs as ccrs

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(16,16))

ax=plt.axes(projection=ccrs.PlateCarree())

ax.coastlines()

plt.show()
ax=plt.axes(projection=ccrs.Orthographic())

ax.stock_img()

plt.show()
# tutorial: https://data-flair.training/blogs/python-geographic-maps-graph-data/

%matplotlib inline

ax=plt.axes(projection=ccrs.AlbersEqualArea())

ax.stock_img()

ny_lon, ny_lat=-74.0059413, 40.7127837

green_lon, green_lat=-77.3664, 35.6127

#plt.figure(figsize=(26,26))

ax.scatter([ny_lon, green_lon], [ny_lat, green_lat], transform=ccrs.Geodetic())

plt.show()
import cartopy.feature as fs



ax = plt.figure(figsize=(20,20)).add_subplot(1,1,1,projection=ccrs.PlateCarree())

ax.set_extent((-140,-60,10,60))#left right bottom top

ax.add_feature(fs.BORDERS)

ax.add_feature(fs.COASTLINE)

ax.add_feature(fs.RIVERS)

ax.scatter(cities['longitude'], cities['latitude'], s=4, color='red')

ax.stock_img()

plt.show()
df.head()
print(df['ViolentCrimesPerPop'].min(),

      df['ViolentCrimesPerPop'].max(),

      df['ViolentCrimesPerPop'].mean(),

      df['ViolentCrimesPerPop'].std(),

      df['ViolentCrimesPerPop'].median())
plt.figure(figsize=(13,7))

plt.hist(df['ViolentCrimesPerPop'].fillna(-1), bins=100)

plt.show()
ax = plt.figure(figsize=(20,20)).add_subplot(1,1,1,projection=ccrs.PlateCarree())

ax.set_extent((-140,-60,10,60))#left right bottom top

ax.add_feature(fs.BORDERS)

ax.add_feature(fs.COASTLINE)

ax.add_feature(fs.RIVERS)

ax.add_feature(fs.LAKES)

ax.scatter(cities['longitude'], cities['latitude'], s=df['ViolentCrimesPerPop'].fillna(0)/30, color='red')

ax.stock_img()

plt.show()
ax = plt.figure(figsize=(20,20)).add_subplot(1,1,1,projection=ccrs.PlateCarree())

ax.set_extent((-140,-60,10,60))#left right bottom top

ax.add_feature(fs.BORDERS)

ax.add_feature(fs.COASTLINE)

ax.add_feature(fs.RIVERS)

ax.add_feature(fs.LAKES)

top10crimes = df[~df['ViolentCrimesPerPop'].isna()]['ViolentCrimesPerPop'].sort_values()[-10:]

top10cities = df.iloc[top10crimes.index, 0]

# removing last 4 letters (Citynamecity --> Cityname)

top10cities = list(map(lambda x: x[:-4], top10cities))

# collecting coordinates of each city

top10longitude = pd.Series(); top10latitude = pd.Series()

for i, city in enumerate(top10cities):

    lon = cities[cities['city'] == city]['longitude']

    lat = cities[cities['city'] == city]['latitude']

    top10longitude = top10longitude.append(lon)

    top10latitude = top10latitude.append(lat)

    ax.text(lon-1+0.2*i, lat+0.1*i, city, transform=ccrs.Geodetic())

ax.scatter(top10longitude, top10latitude, s=30, color='red')

ax.stock_img()

plt.show()
# проверить, связано ли большое количество нанов с тем, что данные собирались по штатам, а не по городам

df.head()
# checking if percentage of nans related to that data was collected by states but not by cities

too_nan = [col for col in df.columns if df[col].isna().sum()/df.shape[0] > 0.5]

for state in df['state'].unique():

    for col in too_nan:

        size = df[df['state'] == state][col].shape[0]

        n_notnan = size - df[df['state'] == state][col].isna().sum()

        uniques = df[df['state'] == state][col].unique().shape[0]

        print('Size of state: {0:3.0f}, not NaNs: {1:3.0f}, unique values in current column: {2:3.0f}'.format(size, n_notnan, uniques))



# so if this was true there should be only one not-nan
df.shape
df_dropped = df.dropna(axis=1, thresh=1400) # drop columns with more than 54% nan values

df_dropped.shape
for col in df_dropped.columns:

    try:

        filler = df_dropped[col].describe()['top']

    except:

        filler = df_dropped[col].describe()['mean']

    df_dropped.loc[:,col] = df_dropped.fillna(filler)
df_dropped.head()
target = df_dropped['ViolentCrimesPerPop']

df_dropped = df_dropped.drop(columns=['communityname', 'state', 'fold', 'ViolentCrimesPerPop', 'nonViolPerPop'])
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import RandomizedSearchCV as rscv

from sklearn.metrics import mean_squared_error as mse

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df_dropped, target, random_state=32)



model = GradientBoostingRegressor()

param_grid = {'learning_rate':[0.001, 0.01, 0.1], 'n_estimators':[50,100,200], 'max_depth':[2,4,7]}

cv = rscv(model, param_grid, n_iter=6, verbose=1).fit(X_train, y_train)



cv.score(X_train, y_train), cv.score(X_test, y_test)
df_filled = df.copy()

for col in df_filled.columns:

    try:

        filler = df_filled[col].describe()['mean']

    except:

        filler = df_filled[col].describe()['top']

    df_filled.loc[:,col] = df_filled.fillna(filler)

df_filled.head()
target = df_filled['ViolentCrimesPerPop']

df_filled = df_filled.drop(columns=['communityname', 'state', 'fold', 'ViolentCrimesPerPop', 'nonViolPerPop'])
X_train, X_test, y_train, y_test = train_test_split(df_filled, target, random_state=32)



model = GradientBoostingRegressor()

param_grid = {'learning_rate':[0.001, 0.01, 0.1], 'n_estimators':[50,100,200], 'max_depth':[2,4,7]}

cv = rscv(model, param_grid, n_iter=6, verbose=1).fit(X_train, y_train)
cv.score(X_train, y_train), cv.score(X_test, y_test)
cv.best_estimator_
model = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,

                          learning_rate=0.1, loss='ls', max_depth=2,

                          max_features=None, max_leaf_nodes=None,

                          min_impurity_decrease=0.0, min_impurity_split=None,

                          min_samples_leaf=1, min_samples_split=2,

                          min_weight_fraction_leaf=0.0, n_estimators=100,

                          n_iter_no_change=None, presort='auto',

                          random_state=None, subsample=1.0, tol=0.0001,

                          validation_fraction=0.1, verbose=0, warm_start=False)

model.fit(X_train, y_train)

print(model.score(X_train, y_train), model.score(X_test, y_test))
imps = pd.Series(model.feature_importances_)

feas = pd.Series(df.drop(columns=['communityname', 'state', 'fold', 'ViolentCrimesPerPop', 'nonViolPerPop']).columns)

FeasImps = pd.DataFrame(columns=['features', 'importances', 'importances by %'])

FeasImps['features'] = feas

FeasImps['importances'] = imps

FeasImps['importances by %'] = np.array(imps)*100
rapes_imp = FeasImps.iloc[129, 1]

kids_imp = FeasImps[FeasImps['features'] == 'PctKidsBornNeverMar']['importances'].values[0]

burgl_imp = FeasImps[FeasImps['features'] == 'burglPerPop']['importances'].values[0]

assault_imp = FeasImps[FeasImps['features'] == 'assaultPerPop']['importances'].values[0]

FeasImps.sort_values('importances', ascending=False).head(4)
corra = []

crimes = pd.Series(target)

for col in df_filled.columns:

    try:

        corra.append(crimes.corr(df_filled[col]))

    except Exception as e:

        corra.append(0)

corrs = pd.DataFrame(columns=['features', 'correlation'])

corrs['features'] = pd.Series(df_filled.columns)

corrs['correlation'] = pd.Series(corra)
crimes.corr(df_filled[df_filled.columns[5]])
kids_corr = corrs.iloc[52, 1]

burgl_corr = corrs.iloc[135, 1]

assault_corr = corrs.iloc[133, 1]

rapes_corr = corrs.iloc[129, 1]

corrs.sort_values('correlation', ascending=False).head(4)
ax = plt.figure(figsize=(20,20)).add_subplot(1,1,1,projection=ccrs.PlateCarree())

ax.set_extent((-140,-60,10,60))#left right bottom top

ax.add_feature(fs.BORDERS)

ax.add_feature(fs.COASTLINE)

ax.add_feature(fs.RIVERS)

#ax.add_feature(fs.LAKES)



def find_top_coords(inces):

    city_list = df.iloc[inces, 0]

    city_list = list(map(lambda x: x[:-4], city_list))

    longitude=pd.Series(); latitude=pd.Series()

    for city in city_list:

        lon = cities[cities['city'] == city]['longitude']

        lat = cities[cities['city'] == city]['latitude']

        longitude = longitude.append(lon)

        latitude = latitude.append(lat)

    return longitude, latitude



ssize=50



# cities

top10crimes = df[~df['ViolentCrimesPerPop'].isna()]['ViolentCrimesPerPop'].sort_values()[-10:]

top10crimes_lon, top10crimes_lat = find_top_coords(top10crimes.index)

ax.scatter(top10crimes_lon, top10crimes_lat, s=ssize, c=np.array([99,19,19])/255, label='Cities')



# kids

top10kids = df[~df['PctKidsBornNeverMar'].isna()]['PctKidsBornNeverMar'].sort_values(ascending=False)[:10]

top10kids_lon, top10kids_lat = find_top_coords(top10kids.index)

ax.scatter(top10kids_lon, top10kids_lat, s=ssize, c=np.array([171,32,32])/255,

           label='Kids {0:.2f} - correlation, {1:.2f} - importance'.format(kids_corr, kids_imp))



# rapes

top10rapes = df[~df['rapesPerPop'].isna()]['rapesPerPop'].sort_values()[-10:]

top10rapes_lon, top10rapes_lat = find_top_coords(top10rapes.index)

ax.scatter(top10rapes_lon, top10rapes_lat, s=ssize, c=np.array([201,137,40])/255,

           label='Rapes {0:.2f} - correlation, {1:.2f} - importance'.format(rapes_imp, rapes_imp))



# burglaries

top10burgl = df[~df['burglPerPop'].isna()]['burglPerPop'].sort_values()[-10:]

top10burgl_lon, top10burgl_lat = find_top_coords(top10burgl.index)

ax.scatter(top10burgl_lon, top10burgl_lat, s=ssize, c=np.array([184,201,32])/255,

           label='Burglaries {0:.2f} - correlation, {1:.2f} - importance'.format(burgl_corr, burgl_imp))



# assaults

top10assault = df[~df['assaultPerPop'].isna()]['assaultPerPop'].sort_values()[-10:]

top10assault_lon, top10assault_lat = find_top_coords(top10assault.index)

ax.scatter(top10assault_lon, top10assault_lat, s=ssize, c=np.array([77,166,255])/255,

           label='Assaults {0:.2f} - correlation, {1:.2f} - importance'.format(assault_corr, assault_imp))



ax.stock_img()

plt.legend()

plt.show()
import plotly.graph_objects as go



fig = go.Figure()



colors=['rgb(99,19,19)', 'rgb(171,32,32)', 'rgb(201,137,40)', 'rgb(184,201,32)', 'rgb(210,242,0)']



def merge_labels(name, inces):

    labels=[]

    cities = df.iloc[inces,0]

    states = df.iloc[inces,1]

    for city, state in zip(cities, states):

        labels.append(name+' in '+str(city[:-4])+', '+str(state))

    return labels



stats=((top10crimes_lon, top10crimes_lat, top10crimes, merge_labels('Violent crimes', top10crimes.index), 'Violent crimes'),

        (top10kids_lon, top10kids_lat, top10kids, merge_labels('Kids', top10kids.index), 'Kids born to never married'),

        (top10rapes_lon, top10rapes_lat, top10rapes, merge_labels('Rapes', top10rapes.index), 'Rapes'),

        (top10burgl_lon, top10burgl_lat, top10burgl, merge_labels('Burglaries', top10burgl.index), 'Burglaries'),

        (top10assault_lon, top10assault_lat, top10assault, merge_labels('Assaults', top10assault.index), 'Assaults'))



for i, stat in enumerate(stats):

    fig.add_trace(go.Scattergeo(

        lon = stat[0], lat = stat[1],

        #text = stat[2].apply(lambda x: round(x, 3)),

        text = pd.Series([str(v)+' '+str(d) for v,d in zip(stat[2], stat[3])]),

        name = stat[4],

        marker = dict(

            size = list(20*stat[2].values/stat[2].max()),

            color=colors[i],

            line_width=0

        )

    ))

    



fig.update_layout(

    title_text = 'Interactive map demonstraiting Violent Crimes and it\'s reasons (all values calculated per 100k)',

    geo=dict(

        resolution=50,

        scope='north america',

        showframe=True,

        showcoastlines=True,

        showland=True,

        landcolor='darkgray',

        countrycolor = 'black',

        coastlinecolor='black',

        #projection_type='equirectangular', #-makes a map  more compressed

        lonaxis_range=[ -120.0, -65.0],

        lataxis_range=[ 25.0, 55.0],

        domain = dict(x=[0, 1], y=[0, 1]),

        bgcolor='rgba(173,217,255, 0.7)'

    )

)    

    

fig.show()