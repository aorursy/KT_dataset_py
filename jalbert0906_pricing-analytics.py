import os

import numpy as np

import pandas as pd

from tqdm import tqdm

import itertools

import math

from math import radians

import warnings

warnings.filterwarnings('ignore')



#Visualization

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('seaborn-whitegrid')

import plotly.graph_objs as go

import plotly.offline as py

py.init_notebook_mode(connected=True)

import seaborn as sns

sns.set_style("darkgrid")

import folium

import folium.plugins

from folium.plugins import MarkerCluster

from folium.plugins import FastMarkerCluster

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



# Using datashader

#Import Libraries

from bokeh.models import BoxZoomTool

from bokeh.plotting import figure, output_notebook, show

import datashader as ds

from datashader.bokeh_ext import InteractiveImage

from functools import partial

from datashader.utils import export_image

from datashader.colors import colormap_select, Hot, inferno, Elevation

from datashader import transfer_functions as tf

output_notebook()
def readData(path, types, chunksize, chunks):



    df_list = []

    counter = 1

    

    for df_chunk in tqdm(pd.read_csv(path, usecols=list(types.keys()), dtype=types, chunksize=chunksize)):



        # The counter helps us stop whenever we want instead of reading the entire data

        if counter == chunks+1:

            break

        counter = counter+1



        # Neat trick from https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost

        # Using parse_dates would be much slower!

        df_chunk['date'] = pd.to_datetime(df_chunk['pickup_datetime'].str.slice(0,10))

        df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)

        df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'])



        # Process the datetime and get hour of day and day of week

        # After Price Reform - Before Price Reform ('newRate')

        df_chunk['hour'] = df_chunk['pickup_datetime'].apply(lambda x: x.hour)

        df_chunk['weekday'] = df_chunk['pickup_datetime'].apply(lambda x: x.weekday())

        df_chunk['newRate'] = df_chunk['pickup_datetime'].apply(lambda x: True if x > pd.Timestamp(2012, 9, 30, 10) else False)

        

        # Aappend the chunk to list

        df_list.append(df_chunk) 



    # Merge all dataframes into one dataframe

    df = pd.concat(df_list)



    # Delete the dataframe list to release memory

    del df_list

    

    return df
 # The path where the Training set is

TRAIN_PATH = '../input/train.csv'



# The datatypes we want to pass the reading function

traintypes = {'fare_amount': 'float32',

              'pickup_datetime': 'str', 

              'pickup_longitude': 'float32',

              'pickup_latitude': 'float32',

              'dropoff_longitude': 'float32',

              'dropoff_latitude': 'float32',

              'passenger_count': 'float32'}



# The size of the chunk for each iteration

chunksizeTrain = 1_000_000



# The number of chunks we want to read

chunksnumberTrain = 3



df = readData(TRAIN_PATH, traintypes, chunksizeTrain, chunksnumberTrain)
# 1) Drop NaN

df.dropna(how = 'any', axis = 'rows', inplace = True)



# 2) 3) Drop fares below 2.5 USD or above 400 USD

df = df[df['fare_amount']>=2.5]

df = df[df['fare_amount']<400]

    

# 4) Drop passenger count below 1 or above 6

df = df[(df['passenger_count']>=1) & (df['passenger_count']<=6)] 

    

# 5) Drop rides outside NYC

minLon = -74.3

maxLon = -73.7

minLat = 40.5

maxLat = 41



df = df[df['pickup_latitude'] < maxLat]

df = df[df['pickup_latitude'] > minLat]

df = df[df['pickup_longitude'] < maxLon]

df = df[df['pickup_longitude'] > minLon]



df = df[df['dropoff_latitude'] < maxLat]

df = df[df['dropoff_latitude'] > minLat]

df = df[df['dropoff_longitude'] < maxLon]

df = df[df['dropoff_longitude'] > minLon]



# Reset Index

df.reset_index(inplace=True, drop=True)



# Convert datatype to categorical

df['hour'] = pd.Categorical(df['hour'])

df['weekday'] = pd.Categorical(df['weekday'])

df['passenger_count'] = pd.Categorical(df['passenger_count'])
trace = go.Pie(values = [df.shape[0],chunksizeTrain*chunksnumberTrain - df.shape[0]],

               labels = ["Useful data" , "Data loss due to missing values or other reasons"],

               marker = dict(colors = ['skyblue' ,'yellow'], line = dict(color = "black", width =  1.5)),

               rotation  = 60,

               hoverinfo = 'label+percent',

              )



layout = go.Layout(dict(title = 'Data Cleaning (percentage of data loss)',

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        showlegend=False

                       )

                  )



fig = go.Figure(data=[trace],layout=layout)

py.iplot(fig)

fig = go.Figure(data=[trace],layout=layout)
# Define plotting function using Datashader

def plot_data_points(longitude,latitude,data_frame) :

    export  = partial(export_image, export_path="export", background="black")

    fig = figure(background_fill_color = "black")    

    cvs = ds.Canvas(plot_width=800, 

                    plot_height=600,

                    x_range=(-74.15,-73.75), 

                    y_range=(40.6,40.9))

    agg = cvs.points(data_frame,longitude,latitude)

    #img = tf.shade(agg, cmap=Hot, how='eq_hist')

    img = tf.shade(agg)   

    image_xpt = tf.dynspread(img, threshold=0.5, max_px=4)

    return export(image_xpt,'map')



# Call function and plot

plot_data_points('pickup_longitude', 'pickup_latitude', df)
# Let's look at some clusters with Folium (20000 points)

samples = df.sample(n=min(20000,df.shape[0]))

m = folium.Map(location=[np.mean(samples['pickup_latitude']), np.mean(samples['pickup_longitude'])], zoom_start=11)

FastMarkerCluster(data=list(zip(samples['pickup_latitude'], samples['pickup_longitude']))).add_to(m)

folium.LayerControl().add_to(m)

m
# One end has to be JFK

jfk_lat_min = 40.626777

jfk_lat_max = 40.665599

jfk_lon_min = -73.823964

jfk_lon_max = -73.743085



# Filter trips originating on JFK

df_fromJFK = df[(df['pickup_latitude']<jfk_lat_max)&

            (df['pickup_latitude']>jfk_lat_min)&

            (df['pickup_longitude']<jfk_lon_max)&

            (df['pickup_longitude']>jfk_lon_min)]



# Filter trips ending on JFK

df_toJFK = df[(df['dropoff_latitude']<jfk_lat_max)&

           (df['dropoff_latitude']>jfk_lat_min)&

           (df['dropoff_longitude']<jfk_lon_max)&

           (df['dropoff_longitude']>jfk_lon_min)]



m1 = folium.Map(location=[40.645580, -73.785115], zoom_start=16)

samples = df_fromJFK.sample(n=min(500,df_fromJFK.shape[0]))

for lt, ln in zip(samples['pickup_latitude'], samples['pickup_longitude']):

            folium.Circle(location = [lt,ln] ,radius = 2, color = 'blue').add_to(m1)

            

samples = df_toJFK.sample(n=min(500,df_toJFK.shape[0]))

for lt, ln in zip(samples['dropoff_latitude'], samples['dropoff_longitude']):

            folium.Circle(location = [lt,ln] ,radius = 2, color = 'red').add_to(m1)

        

m1
from shapely.geometry import Point

from shapely.geometry.polygon import Polygon



# Define polygon using coordinates (Just took them from Google Maps by clicking on the map)

lats_vect = [40.851638, 40.763022, 40.691262, 40.713380, 40.743944, 40.794344, 40.846332]

lons_vect = [-73.952423, -74.010418, -74.026685, -73.972200, -73.962051, -73.924073, -73.926454]

lons_lats_vect = np.column_stack((lons_vect, lats_vect))

polygon = Polygon(lons_lats_vect)



# Plot the polygon using Folium

man_map = folium.Map(location=[40.7631, -73.9712], zoom_start=12)

for i in range(0,6):

    folium.PolyLine(locations=[[lats_vect[i],lons_vect[i]], [lats_vect[i+1],lons_vect[i+1]]], color='blue').add_to(man_map)

folium.PolyLine(locations=[[lats_vect[6],lons_vect[6]], [lats_vect[0],lons_vect[0]]], color='blue').add_to(man_map)

man_map
# Check for every point on df_train if it belongs to polygon or not

manhattanRides = df[df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']]

                          .apply(lambda row: ((polygon.contains(Point(row['pickup_longitude'],row['pickup_latitude']))) &

                                              (polygon.contains(Point(row['dropoff_longitude'],row['dropoff_latitude'])))), axis=1)]



# Plot the remaining dataset 'manhattanRides'

plot_data_points('pickup_longitude', 'pickup_latitude', manhattanRides)
print('Percentage of trips that happen inside Manhattan: ' + str(np.around(100*(manhattanRides.shape[0])/df.shape[0],2)))
# Simple Euclidean Distance calculator 

def quickDist(lat1, lng1, lat2, lng2):

    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])

    R = 6371

    x = (lng2 - lng1) * np.cos(0.5*(lat2+lat1))

    y = lat2 - lat1

    d = R * np.sqrt(x*x + y*y)

    return d



# Longitude distance (use same Euclidean distance function with fixed latitude)

def latDist(lat1, lng1, lat2, lng2):

    uno = quickDist((lat1+lat2)/2, lng1, (lat1+lat2)/2, lng2)

    return uno



# Calculate real distance (Manhattan distance with 29 degrees to north)

def realManDist(lat1, lng1, lat2, lng2):

    flightDist = quickDist(lat1, lng1, lat2, lng2)

    latDistance = latDist(lat1, lng1, lat2, lng2)

    if flightDist == 0:

        ret = np.nan

    else:

        th = np.arccos(latDistance/flightDist)

        ata = flightDist*np.cos(th-0.506) + flightDist*np.sin(th-0.506)

        bta = flightDist*np.cos(th+0.506) + flightDist*np.sin(th+0.506)

        ret = max(ata,bta)

    return ret
# Calculate distance for every ride on manhattanRides

manhattanRides['distance'] = manhattanRides.apply(lambda row: realManDist(row['pickup_latitude'], 

                                                                          row['pickup_longitude'], 

                                                                          row['dropoff_latitude'], 

                                                                          row['dropoff_longitude']), axis=1)
# Drop rows with distance = NaN

manhattanRides.dropna(inplace=True)



# Drop rows with distance below 0.3 (300 meters)

manhattanRides = manhattanRides[manhattanRides['distance']>=0.3]
# Download manhattanRides dataset as csv

manhattanRides.to_csv('manhattanRides.csv',index=False)
# Define how we want to split the data into sections based on 'fare_amount'

a = 40

b = 70

c = 300



# Plot normalized histogram for each section

plt.figure(figsize = (25,7))

plt.subplot(1,3,1)

plt.title('Below ' + str(a) + ' USD',color = "b")

plt.ylabel('Normalized Density')

sns.distplot(manhattanRides[manhattanRides['fare_amount']<=a]['fare_amount'], norm_hist=True, bins=np.arange(0,a))

plt.subplot(1,3,2)

plt.title('From ' + str(a) + ' USD to ' + str(b) + ' USD',color = "b")

plt.ylabel('Normalized Density')

sns.distplot(manhattanRides[(manhattanRides['fare_amount']>a)&(df['fare_amount']<=b)]['fare_amount'], norm_hist=True, bins=np.arange(a,b))

plt.subplot(1,3,3)

plt.title('From ' + str(b) + ' USD to ' + str(c) + ' USD',color = "b")

plt.ylabel('Normalized Density')

sns.distplot(manhattanRides[(manhattanRides['fare_amount']>b)&(df['fare_amount']<=c)]['fare_amount'], norm_hist=True, bins=np.arange(b,c));
# Split df_train into a dataset of the rides before the fare rules change and after the fare rules change

df_before = manhattanRides[manhattanRides['newRate']==False]

df_after = manhattanRides[manhattanRides['newRate']==True]

print('Mean fare BEFORE rate change: ' + str(np.around(df_before['fare_amount'].mean(),2)))

print('Mean fare AFTER rate change: ' + str(np.around(df_after['fare_amount'].mean(),2)))

print('Median fare BEFORE rate change: ' + str(np.around(df_before['fare_amount'].median(),2)))

print('Median fare AFTER rate change: ' + str(np.around(df_after['fare_amount'].median(),2)))
plt.figure(figsize = (30,10))

plt.subplot(1,3,1)

plt.title('Mean fare_amount by hour of the day',color = "b")

ax = sns.barplot(x='hour',y='fare_amount', data = manhattanRides, edgecolor=".1", errcolor = 'red')

plt.subplot(1,3,2)

plt.title('Mean fare_amount by day of the week',color = "b")

ax = sns.barplot(x='weekday',y='fare_amount', data = manhattanRides, edgecolor=".1", errcolor = 'red')

plt.subplot(1,3,3)

plt.title('Mean fare_amount by passenger_count',color = "b")

ax = sns.barplot(x='passenger_count',y='fare_amount', data = manhattanRides, edgecolor=".1", errcolor = 'red')
df_sorted = manhattanRides.sort_values('pickup_datetime')

df_sorted.reset_index(inplace=True, drop=True)

mvavg = df_sorted.rolling(window=2000, on='pickup_datetime')['fare_amount'].mean()

mvavg_pd = pd.DataFrame(columns=['avg', 'pickup_datetime'])

mvavg_pd['avg'] = mvavg

mvavg_pd['pickup_datetime'] = df_sorted['pickup_datetime']



df_sorted.plot('pickup_datetime','fare_amount', figsize=(30,10), title='fare_amount time series')

mvavg_pd.plot('pickup_datetime','avg', figsize=(30,10), title='fare_amount moving average');
plt.figure(figsize = (20,7))

plt.subplot(1,2,1)

plt.title('Old Rate, cents value',color = "b")

sns.distplot(np.mod(df_before['fare_amount'],1)) 

plt.subplot(1,2,2)

plt.title('New Rate, cents value',color = "g")

sns.distplot(np.mod(df_after['fare_amount'],1), color='green');
# Plot the 'fare_amount' against the distance of the trip

plt.figure(figsize = (20,15))

plt.title('Manhattan Rides', color = "b")

plt.ylabel('Fare in USD')

plt.xlabel('Distance in Km')

plt.scatter(manhattanRides['distance'], manhattanRides['fare_amount'], alpha=0.5);
# Split train/test

from sklearn.model_selection import train_test_split

manhattanRides_train, manhattanRides_test = train_test_split(manhattanRides, test_size=0.2, random_state=42)



# Split before and after

manhattanRides_train_before = manhattanRides_train[manhattanRides_train['newRate']==0]

manhattanRides_train_after = manhattanRides_train[manhattanRides_train['newRate']==1]

manhattanRides_test_before = manhattanRides_test[manhattanRides_test['newRate']==0]

manhattanRides_test_after = manhattanRides_test[manhattanRides_test['newRate']==1]
# Fitting a linear model and measuring the MSE

import statsmodels.api as sm

# Define function that makes regression and returns params

def measureMSE(df_train, df_test):

    regression = sm.OLS(df_train['fare_amount'], sm.add_constant(df_train['distance'])).fit()

    farepred = regression.predict(sm.add_constant(df_test['distance'])) 

    mse = np.around(np.sqrt((((df_test['fare_amount']-farepred)**2).sum())/(df_test.shape[0])),4)

    return [regression.params[1], regression.params[0], mse]
# Apply function on manhattanRides_train

reg_before = measureMSE(manhattanRides_train_before, manhattanRides_test_before)

reg_after = measureMSE(manhattanRides_train_after, manhattanRides_test_after)



print('Before fare rule change:')

print ('Slope: ' + str(np.around(reg_before[0],2)))

print ('Intercept: ' + str(np.around(reg_before[1],2)))

print ('RMSE: ' + str(np.around(reg_before[2],4)))



print(' ')



print('After fare rule change:')

print ('Slope: ' + str(np.around(reg_after[0],2)))

print ('Intercept: ' + str(np.around(reg_after[1],2)))

print ('RMSE: ' + str(np.around(reg_after[2],4)))
import h2o

h2o.init();
h20_train_before = h2o.H2OFrame(manhattanRides_train_before)

h20_test_before = h2o.H2OFrame(manhattanRides_test_before)

h20_train_after = h2o.H2OFrame(manhattanRides_train_after)

h20_test_after = h2o.H2OFrame(manhattanRides_test_after)



# Define feature space and label space

myCat = ['hour', 'weekday']

myNum = ['distance', 'passenger_count']

myResponse = 'fare_amount'



for i in myCat:

    h20_train_before[i] = h20_train_before[i].asfactor()

    h20_test_before[i] = h20_test_before[i].asfactor()

    h20_train_after[i] = h20_train_after[i].asfactor()

    h20_test_after[i] = h20_test_after[i].asfactor()
from h2o.estimators.gbm import H2OGradientBoostingEstimator

model_before = H2OGradientBoostingEstimator(nfolds=5, seed=42, stopping_metric = "MSE")

model_after = H2OGradientBoostingEstimator(nfolds=5, seed=42, stopping_metric = "MSE")
model_before.train(x=myNum+myCat,y=myResponse, training_frame=h20_train_before, validation_frame=h20_test_before)
model_after.train(x=myNum+myCat,y=myResponse, training_frame=h20_train_after, validation_frame=h20_test_after)
print('Before fare rule change:')

print ('RMSE: ' + model_before.cross_validation_metrics_summary().cell_values[5][1])



print(' ')



print('After fare rule change:')

print ('RMSE: ' + model_after.cross_validation_metrics_summary().cell_values[5][1])
model_before.varimp_plot()
model_after.varimp_plot()
df_demand = manhattanRides[['fare_amount', 'date', 'hour', 'distance']]

df_demand['fare/distance'] = df_demand['fare_amount']/df_demand['distance']

df_demand = df_demand[['date', 'distance', 'fare/distance']]

df_demand = df_demand.groupby('date').agg({'fare/distance':'mean', 'distance':'sum'})

df_demand.reset_index(inplace=True)  

df_demand.columns = ['date', 'mean', 'sum']

df_demand.plot('date', 'mean', figsize=(30,10), alpha=0.5, title='Average price per kilometer per day');

df_demand.plot('date', 'sum', figsize=(30,10), alpha=0.5, title='Amount of kilometers sold per day')

df_demand.plot.scatter('mean', 'sum', figsize=(30,10), alpha=0.5, title='Amount of kilometers sold VS price per kilometer');

sns.regplot(x="mean", y="sum", data=df_demand)
# The regression coefficients are:

print('Slope: ' + str(np.around(sm.OLS(df_demand['sum'], sm.add_constant(df_demand['mean'])).fit().params[1], 4)))

print('Intersect: ' + str(np.around(sm.OLS(df_demand['sum'], sm.add_constant(df_demand['mean'])).fit().params[0], 4)))
import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.linear_model import LinearRegression



# Initial Parameters

alpha = sm.OLS(df_demand['sum'], sm.add_constant(df_demand['mean'])).fit().params[0]

beta = -sm.OLS(df_demand['sum'], sm.add_constant(df_demand['mean'])).fit().params[1]

initial_p = 2

sigma = 1



# Model

def model(p):

    val = alpha - beta*p + np.random.normal(0,1)

    return val



# Regression function

def regression(p_list, D_list):

    reg = LinearRegression(fit_intercept=False).fit(p_list, D_list)

    return reg.coef_



# Performance accummulator

def performance(perf_last,p_val):

    return perf_last - (beta-p_val*(alpha-beta*p_val))



final_p = []

final_alpha = []

final_beta = []

perf_record = []

num_iterations = 1500

num_expriments = 100



for j in range(num_expriments):



    # Initialize

    p_list = [[1,initial_p], [1,2*initial_p]]

    D_list = [model(initial_p), model(2*initial_p)]

    coef = regression(p_list, D_list)

    alpha_list = [coef[0]]

    beta_list = [-coef[1]]

    perf_metric = [0]

  

    # Run

    for i in range(num_iterations):

        p_list.append([1,alpha_list[-1]/(2*beta_list[-1])])

        D_list.append(model(alpha_list[-1]/(2*beta_list[-1])))

        coef = regression(p_list, D_list)

        alpha_list.append(coef[0])

        beta_list.append(-coef[1])

        if i>=2:

            perf_metric.append(performance(perf_metric[-1],p_list[-1][1])) 

 

    # Save final price, alpha and beta of iteration

    final_p.append(p_list[-1][1])

    final_alpha.append(alpha_list[-1])

    final_beta.append(beta_list[-1])
# Price path

data=pd.DataFrame([x[1] for x in p_list[3:1000]])

data.plot(figsize=(10,5), alpha=0.5, title='Price path');
# Alpha path

data=pd.DataFrame([x for x in alpha_list[3:1000]])

data.plot(figsize=(10,5), alpha=0.5, title='Alpha path');
# Beta path

data=pd.DataFrame([x for x in beta_list[3:1000]])

data.plot(figsize=(10,5), alpha=0.5, title='Beta path');
# Price histogram

ax = sns.distplot(final_p)

ax.set(xlabel='price', ylabel='count', title='Final Price histogram');
# Alpha histogram

ax = sns.distplot(final_alpha)

ax.set(xlabel='alpha', ylabel='count', title='Final Alpha histogram');
# Beta histogram

ax = sns.distplot(final_beta)

ax.set(xlabel='beta', ylabel='count', title='Final Beta histogram');
# Performance metric

data=pd.DataFrame(perf_metric)

data.plot(figsize=(10,5), alpha=0.5, title='Performance Metric - commulative error');
import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.linear_model import LinearRegression

import random



# Initial Parameters

alpha = sm.OLS(df_demand['sum'], sm.add_constant(df_demand['mean'])).fit().params[0]

beta = -sm.OLS(df_demand['sum'], sm.add_constant(df_demand['mean'])).fit().params[1]

initial_p = 2



df_demand_sorted = df_demand.sort_values(by='mean', ascending=True)

minPrice = df_demand_sorted['mean'].min()

maxPrice = df_demand_sorted['mean'].max()



# Function that returns the closes price to the queryed one and the demand for that query 

def model(price):

    if ((price < maxPrice) & (price > minPrice)):

        idx = df_demand_sorted['mean'].sub(price).abs().idxmin()

        return [df_demand_sorted.loc[idx]['mean'], df_demand_sorted.loc[idx]['sum']]

    else:

        rand_price = random.uniform(minPrice, maxPrice)

        return model(rand_price)



# Regression function

def regression(p_list, D_list):

    reg = LinearRegression(fit_intercept=False).fit(p_list, D_list)

    return reg.coef_



# Performance accummulator

def performance(perf_last,p_val):

    return perf_last - (beta-p_val*(alpha-beta*p_val))



final_p = []

final_alpha = []

final_beta = []

perf_record = []



num_iterations = 1500

num_expriments = 100



for j in range(num_expriments):

  

    # Initialize

    p_list = [[1,initial_p], [1,2*initial_p]]

    D_list = [model(initial_p)[1], model(2*initial_p)[1]]

    coef = regression(p_list, D_list)

    alpha_list = [coef[0]]

    beta_list = [-coef[1]]

    perf_metric = [0]



    # Run

    for i in range(num_iterations):

        query = alpha_list[-1]/(2*beta_list[-1])

        vals = model(query)

        p_list.append([1,vals[0]])

        D_list.append(vals[1])

        coef = regression(p_list, D_list)

        alpha_list.append(coef[0])

        beta_list.append(-coef[1])

        if i>=2:

            perf_metric.append(performance(perf_metric[-1],p_list[-1][1]))

      

    # Save final price, alpha and beta of iteration

    final_p.append(p_list[-1][1])

    final_alpha.append(alpha_list[-1])

    final_beta.append(beta_list[-1])
# Price path

data=pd.DataFrame([x[1] for x in p_list[3:1000]])

data.plot(figsize=(10,5), alpha=0.5, title='Price path');
# Alpha path

data=pd.DataFrame([x for x in alpha_list[3:1000]])

data.plot(figsize=(10,5), alpha=0.5, title='Alpha path');
# Beta path

data=pd.DataFrame([x for x in beta_list[3:1000]])

data.plot(figsize=(10,5), alpha=0.5, title='Beta path');
# Price histogram

ax = sns.distplot(final_p)

ax.set(xlabel='price', ylabel='count', title='Final Price histogram');
# Alpha histogram

ax = sns.distplot(final_alpha)

ax.set(xlabel='alpha', ylabel='count', title='Final Alpha histogram');
# Beta histogram

ax = sns.distplot(final_beta)

ax.set(xlabel='beta', ylabel='count', title='Final Beta histogram');
# Performance metric

data=pd.DataFrame(perf_metric)

data.plot(figsize=(10,5), alpha=0.5, title='Performance Metric - commulative error');