# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import folium as folium #folium would be the map concentrated package to draw fancy map on the canvas and folium is a wrapper of leafjs 

import seaborn as sns # another pretty plot package that is based on matplot package in python

import missingno as msno #showing the missing value of the dataset

import matplotlib.pyplot as plt #basic plots for matplot 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from subprocess import check_output

print(check_output(["ls", "../input/nyc-taxi-trip-duration/train/"]).decode("utf8"))



#check out the data

titanic = pd.read_csv("../input/titantic/train.csv")

#train = pd.read_csv("../input/ny-taxi-trip-duration/train.zip/train.csv") 



#find the path in the file path and output the file

train=pd.read_csv("../input/nyc-taxi-trip-duration/train/train.csv")

train.head()

#there are two ways to make a basic summary in python compared to R, describe function and info function 

summary = train.describe() 

summary 
#Like SAS,you could input what you want for decile and output the summary function

perc = [0.2,0.4,0.6,0.8]

include = ['object','float','int']

desc = train.describe(percentiles = perc,include= include)

desc

#panda dataframe could check the over missing values

train.info()
#ways to visualizing the missing values,in thie particular dataset,wecould see that none of the data has missing values. 

sns.heatmap(train.isnull(), cbar=False)

#another example of showing different missing value datasets. 

sns.heatmap(titanic.isnull(),cbar=False)
#python has another package missing no speicalizing in missing value data visualization 

msno.matrix(titanic)
#Showing Distribution of the dataset 

sns.distplot(train['trip_duration'],color = 'skyblue',label = 'trip_duration')

plt.legend()

#plt.xlim(0,35000)

plt.show()





#conver all the seconds into minutes and plot a better duration 

train['trip_dur_to_m'] = round(train['trip_duration']/60,0)

train.head(10)



#generate a distribution/freq table 

train.trip_dur_to_m.value_counts(sort=True) 
#plot the duration distribution again

#initialize a figuresize plot

fig, ax = plt.subplots(figsize=(14, 4))

tripduration = train[train.trip_dur_to_m < train.trip_dur_to_m.quantile(.97)]

tripduration.groupby('trip_dur_to_m').count()['id'].plot()



#add the label to each plot 

plt.xlabel('Trip duration in minutes')

plt.ylabel('Trip count')

plt.title('Duration distribution')

plt.show()
#Showing the passagers Distribution 

sns.distplot(train['passenger_count'],color = 'red',label = 'passenger_count')

plt.legend()

plt.show()
#the improved way to plot this distribution graph 

#distribution plot 

sns.distplot(train.passenger_count,color = 'orange',kde=False, bins=train.passenger_count.max(), 

                vertical=True, axlabel="Passengers distribution");

train.passenger_count.value_counts(sort=False)
#plot by vendor id 

vendor_id_count = train['vendor_id'].value_counts() 

#the data structure here is series, so this one should be sorted by inex 

vendor_id_count.sort_index 



#barplot example for vendor id 

sns.barplot(x= vendor_id_count.values,y= vendor_id_count.index,data=vendor_id_count,palette='Set2')

plt.xlabel('vendor_id')

plt.ylabel('total rides')

plt.show() 
train.head(10)
tripduration = train[train.trip_dur_to_m < train.trip_dur_to_m.quantile(.97)]

tripduration.head()
#extract date/time from the dataset

#extract columns from two data timestamps

def extract_time_interval(df,colname,start,end):

    df_c = df.copy() 

    df_c[f'{colname}'] = (df_c[end] - df_c[start]).astype('timedelta64[m]')

    return df_c 



import datetime



#extract all the date and time 

#type(train['pickup_datetime'])

def datetime_extract(df, columns, modeling=False):

    df_ = df.copy()

    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for col in columns:

        try:

            prefix = col

            if "_" in col:

                prefix = col.split("_")[0]

            ts = f"{prefix}_ts"

            df_[ts] = pd.to_datetime(df_[col])

            df_[f"{prefix}_month"] = df_[ts].dt.month

            df_[f"{prefix}_weekday"] = df_[ts].dt.weekday

            df_[f"{prefix}_day"] = df_[ts].dt.day

            df_[f"{prefix}_hour"] = df_[ts].dt.hour

            df_[f"{prefix}_minute"] = df_[ts].dt.minute

            if not modeling: 

                df_[f"{prefix}_date"] = df_[ts].dt.date

                df_[f"{prefix}_dayname"] = df_[f"{prefix}_weekday"].apply(lambda x: day_names[x])

            else:

                df_.drop(columns=[ts, col], axis = 1)

        except:

            pass

    return df_



   
train = datetime_extract(train, ['pickup_datetime', 'dropoff_datetime'])

train.head(10)

train_time = extract_time_interval(train, 'delta_m', 'pickup_ts', 'dropoff_ts')

train.head(10)
#time series plot 

#count the passgeners during the day 

fig, ax = plt.subplots(ncols=2, figsize=(14, 5))

for i, col in enumerate(['pickup', 'dropoff']):

    ax[i].plot(train.groupby([f'{col}_date']).sum()['passenger_count'])

    ax[i].set(xlabel='Months', ylabel="Total passengers", title="Total passengers per date")
#Import Libraries

from bokeh.models import BoxZoomTool

from bokeh.plotting import figure, output_notebook, show

import datashader as ds

from datashader.bokeh_ext import InteractiveImage

from functools import partial

from datashader.utils import export_image

from datashader.colors import colormap_select, Greys9, Hot, inferno,Set1

from datashader import transfer_functions as tf

output_notebook()



#plot datapoints by location coordinates

def plot_data_points(longitude,latitude,data_frame,focus_point) :

    #plot dimensions

    x_range, y_range = ((-74.14,-73.73), (40.6,40.9))

    plot_width  = int(750)

    plot_height = int(plot_width//1.2)

    export  = partial(export_image, export_path="export", background="black")

    fig = figure(background_fill_color = "black")    

    #plot data points

    cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height,

                    x_range=x_range, y_range=y_range)

    agg = cvs.points(data_frame,longitude,latitude,

                      ds.count(focus_point))

    img = tf.shade(agg, cmap= Hot, how='eq_hist')

    image_xpt  =  tf.dynspread(img, threshold=0.5, max_px=4)

    return export(image_xpt,"NYCT_hot")



plot_data_points('pickup_longitude', 'pickup_latitude',train,"passenger_count")
plot_data_points('dropoff_longitude', 'dropoff_latitude',train,"passenger_count")