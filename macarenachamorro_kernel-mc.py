from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 

import plotly as py

import plotly.graph_objs as go

import plotly.figure_factory as ff

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats

#from plotly.plotly import iplot

import seaborn as sns

import gc

from datetime import datetime

import statistics





#Imported rows limited to nrows number

hst_df = pd.read_csv('../input/spanish-high-speed-rail-system-ticket-pricing/high_speed_spanish_trains.csv', nrows=100000) 

#use this one for full data

#hst_df = pd.read_csv('../input/spanish-high-speed-rail-system-ticket-pricing/high_speed_spanish_trains.csv', low_memory = False) 



## Uncomment this for structure 

print('data_head',hst_df.head())

#print('data_tail',hst_df.tail())

#print(hst_df.shape())

#print(type(hst_df))

#hst_df.info()



#Filtering by Promo fare 

fare = hst_df['fare']=='Promo'

promoFare = hst_df[fare]

print("Promo Fare Info: ",promoFare.info())

print("Promo Fare Columns:",promoFare.columns)

#print(promoFare.shape())



#Delete first import. It's too big!

del hst_df

gc.collect()



#Fix date type

for i in ['insert_date','start_date','end_date']:

      promoFare[i] = pd.to_datetime(promoFare[i])

print("Promo Fare Info: ",promoFare.info())



#check for nulls

print("Check for nulls: ", promoFare.isnull().mean()*100)



#fix nulls 

#eliminar columnas with nulls innecesarias

promoFare = promoFare.drop(['price_tree'],1)

promoFare = promoFare.drop(['batch'],1)

promoFare = promoFare.drop(['id'],1)



#imputar precio 

promoFare.loc[promoFare.price.isnull(), 'price'] = promoFare.groupby('fare').price.transform('mean')

print("Check for nulls: ",promoFare.isnull().sum()/promoFare.shape[0]*100)

print("Describe", promoFare.describe)

print('Promo Fare Head', promoFare.head())

print('price Mean: ',np.mean(promoFare['price']) )

print('price Median: ', np.median(promoFare['price'])) 

print('price Mode: ', stats.mode(promoFare['price'])) 

print('Standard Deviation ', statistics.stdev(promoFare['price']))



#check min + max dates _ insert vs start

print("min start date: ", promoFare.start_date.min())

print("max start date: ", promoFare.start_date.max())

print("min insert date: ", promoFare.insert_date.min())

print("max insert date: ", promoFare.insert_date.max())

# no existe correccion necesaria para el rango de 100000 rows.



# Add Features

promoFare['day_of_week_insert'] = promoFare['insert_date'].dt.day_name()

promoFare['day_of_week_start'] = promoFare['start_date'].dt.day_name()

promoFare['day_of_week_end'] = promoFare['end_date'].dt.day_name()

promoFare['start_hour'] = promoFare['start_date'].dt.hour

promoFare['end_hour'] = promoFare['end_date'].dt.hour

promoFare['same_day_journey'] = np.where(promoFare['start_date'].dt.date==promoFare['end_date'].dt.date,'yes','no')

promoFare['travel_time_min'] = (promoFare['start_date'] - promoFare['end_date']) / np.timedelta64(1,'m')

promoFare['month_start'] = promoFare['start_date'].dt.month

promoFare['month_end'] = promoFare['end_date'].dt.month

avgtimeTravelled = promoFare['travel_time_min'].mean()



# Finding outliers // Boxplot

import plotly.express as px

fig = px.box(promoFare, x='destination', y="price",color='train_class',title='Checking Data Outliers')

fig.show()



#IQR

Q1 = promoFare['price'].quantile(0.25)

Q3 = promoFare['price'].quantile(0.75)

IQR = Q3 - Q1

print("IQR", IQR)

outliersIQR = (promoFare['price'] < (Q1 -1.5 * IQR)) | (promoFare['price'] > (Q3 + 1.5 *IQR))

print('promoFare shape', promoFare.shape)

promoPriceOut = promoFare[(promoFare['price'] < (Q1 -1.5 * IQR)) | (promoFare['price'] > (Q3 + 1.5 *IQR))]

print('promoFare out shape', promoPriceOut.shape)

print("Outliers IQR",outliersIQR) 



#Z score DType Error with date ... calculating for price only.

z = np.abs(stats.zscore(promoFare['price']))

print('Z Score', z)

threshold = 3 

outlierz = np.where(z > threshold)

print("Outliers Z Score",outlierz) 



fig = px.box(promoPriceOut, x='destination', y="price",color='train_class',title='Checking Data Outliers')

fig.show()



#Scatterplot

fig = px.scatter(promoFare, x='destination', y='price',color='train_class',render_mode='webgl')

fig.show()



fig = px.scatter(promoPriceOut, x='destination', y='price',color='train_class',render_mode='webgl')

fig.show()



#Date Analysis 

countDate = promoFare['start_date'].dt.date.value_counts()

plt.figure(figsize=(30,10))

sns.barplot(countDate.index,countDate.values, alpha = 0.8)

plt.title('Plot Mayor cantidad de viajes iniciados por fecha')

plt.xticks(rotation='vertical')

plt.ylabel('Nro de Viajes')

plt.xlabel('Fecha')

plt.show()



#countDate = countDate[1]

fig = px.scatter(promoFare, x='start_date', y='price',marginal_x='histogram', title='Histogram')

fig.show()



fig = px.scatter(promoFare, x='start_date', y='price', color='day_of_week_start',marginal_x='histogram', title='Histogram')

fig.show()



#inicio de viajes por dias 

count_start_days = promoFare['day_of_week_start'].value_counts()

trace = go.Bar(

                x= count_start_days.values,

                y= count_start_days.index , orientation = 'h',

                text = count_start_days.index)

data = [trace]

layout = go.Layout(title = 'Inicio de Viajes por Dia')

fig = go.Figure(data=data, layout= layout)

py.offline.iplot(fig)



#Analisis por Dias

count_end_days = promoFare['day_of_week_end'].value_counts()

traceEnd = go.Bar(

                x= count_end_days.values,

                y= count_end_days.index , orientation = 'h',

                text = count_end_days.index)

dataEnd = [traceEnd]

layoutEnd = go.Layout(title = 'Fin de Viajes por Dia')

figEnd = go.Figure(data=dataEnd, layout= layoutEnd)

py.offline.iplot(figEnd)



#plt.scatter(x=promoFare['price'], y= promoFare['day_of_week_start'], alpha=0.5)

#plt.xlabel('Price')

#plt.ylabel('Day of week')

#plt.title('Price and Day')

#plt.show()



fig = px.scatter(promoFare, x='price', y='day_of_week_start',color='train_class',render_mode='webgl', title='Scatter plot')

fig.show()
