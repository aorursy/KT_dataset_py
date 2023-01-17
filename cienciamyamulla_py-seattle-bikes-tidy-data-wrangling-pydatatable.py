# Install datatable package

!pip install datatable
# Loading libraries

import datatable as dt

from datatable import *

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

sns.set_style('whitegrid')

plt.style.use('ggplot')

#plt.style.use('fivethirtyeight')
print(f'The loaded datable version is : {dt.__version__}')
# Importing data

seattle_bikes_dt = dt.fread("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-04-02/bike_traffic.csv")
# datatable syntax image

from IPython.display import Image

Image("../input/dt.jpg")
# Glance at first 5 observations

seattle_bikes_dt[0:5,:]
# Setting dt row configuration to display first 4 and last 4 observations

dt.options.display.head_nrows=4

dt.options.display.tail_nrows=4

##dt.options.display.max_nrows=8
# looking at first 4 and last 4observations

seattle_bikes_dt
# Creating a pandas df from dt

seattle_bikes_df = seattle_bikes_dt.to_pandas()
# Look at pandas seatle bike df col names and types

seattle_bikes_df.info()

# changing date column type in seatle df

seattle_bikes_df['date']= pd.to_datetime(seattle_bikes_df['date'],format="%m/%d/%Y %I:%M:%S %p")
# Glance at seatle df field types

seattle_bikes_df.info()
# A df with Additional columns created from date type  

seatle_dates_df = pd.DataFrame({

    

    'year' : seattle_bikes_df['date'].dt.year,

    'month': seattle_bikes_df['date'].dt.month,

    'day'  : seattle_bikes_df['date'].dt.day,

    'hour' : seattle_bikes_df['date'].dt.hour,

    'week_day': seattle_bikes_df['date'].dt.day_name()

    

})
# Convering date df to dt

seatle_dates_dt = dt.Frame(seatle_dates_df)
# delating a string date column from dt

del seattle_bikes_dt['date']
# Concatinating two dts to have a tidy dt

seatle_bikes_dt_tidy = dt.cbind(seatle_dates_dt,seattle_bikes_dt)
# Tidy DT first 4 and last 4 observations

seatle_bikes_dt_tidy
# Modifying observations of crossing col - set - 1

seatle_bikes_dt_tidy[f.crossing=="39th Ave NE Greenway at NE 62nd St",f.crossing]='Greenwayway-NE-62Strt'
# Modifying observations of crossing col - set - 2

seatle_bikes_dt_tidy[f.crossing=="Broadway Cycle Track North Of E Union St",f.crossing]='BroadwayCycleTrack-N'
# Modifying observations of crossing col - set - 3

seatle_bikes_dt_tidy[f.crossing=="NW 58th St Greenway at 22nd Ave",f.crossing]='Greenway-NW-58Strt'
# Checking field types of dt

for i in range(0,seatle_bikes_dt_tidy.shape[1]):

    print(f'The column - {seatle_bikes_dt_tidy.names[i]} - is a type of {seatle_bikes_dt_tidy.stypes[i]}')
# Converting pedestrian column value to a type bool

seatle_bikes_dt_tidy['ped_count']=dt.bool8
# Viewing observations whose ped_count is not NA's

seatle_bikes_dt_tidy[~dt.isna(f.ped_count),:]
# Data manipulation and Visualization

plt.figure(figsize=(14,6))

sns.barplot(y='crossing',x='count',

            data=seatle_bikes_dt_tidy[:,count(),by(f.crossing)

                                     ][:,:,sort(-f.count)

                                      ].to_pandas())

plt.title("How many of bikes/pedestrains have passed though the different crossings?")

plt.show()
# Data vis and Manipulation

plt.figure(figsize=(14,6))

sns.barplot(x='direction',y='count',

            data=seatle_bikes_dt_tidy[:,count(),by(f.direction)

                                     ][:,:,sort(-f.count)

                                      ].to_pandas())

plt.title("How many of bikes/pedestrains have gone from the direction?")

plt.show()
# Data Vis

plt.figure(figsize=(14,8))

sns.barplot(y='crossing',x='total', hue='direction',

            data=seatle_bikes_dt_tidy[:,{

                'total': count()

            },by(f.direction,f.crossing)

            ].to_pandas())

plt.title("How many of bikes/pedestrians pass though crossings in different directions?")

plt.show()
# Data Vis

plt.figure(figsize=(14,6))

sns.barplot(x='year',y='count',

            data=seatle_bikes_dt_tidy[:,count(),by(f.year)

                                     ].to_pandas())

plt.title("Seattle - Bike traffic trends over the years")

plt.show()
# Groupig based on NA's field

seatle_bikes_dt_tidy[:,{

    'total':count()

},by(dt.isna(f.ped_count),

     f.year,f.crossing)

]
plt.figure(figsize=(14,14))

# Facetgrid

vis_5 = sns.FacetGrid(data=seatle_bikes_dt_tidy[:,{'bike_count':dt.sum(f.bike_count)},by(f.crossing,f.hour)

                    ][:,{'hour':f.hour,

                         'pct_bike' : dt.math.rint((f.bike_count/dt.sum(f.bike_count))*100)

                        },by(f.crossing)

                     ].to_pandas(),

                      col='crossing',col_wrap=4,margin_titles=False)

# plotting a line plot

vis_g = vis_5.map(sns.lineplot,'hour','pct_bike',color='r')

vis_g.set(xlim=(0, 24), ylim=(0, 18),xticks=[0, 3, 6,9,12,15,18,21,24], yticks=[1, 5, 10,15,20])

[plt.setp(ax.texts, text="") for ax in vis_5.axes.flat]

vis_5.set_titles(col_template = '{col_name}')

plt.tight_layout()
# Glance at data

seatle_bikes_dt_tidy
# Taking hours col value

horas = seatle_bikes_dt_tidy[:,f.hour]
# Binning on Horas DT column

horas_bins = ["Morning" if (dia>=7 and dia <11) else

 "Mid Day" if (dia>=11 and dia <16) else

 "Evening" if (dia>=16 and dia <18) else "Night" for dia in horas['hour'].to_list()[0]]
# Framing a new DT from the above list of time bins

time_window_dt=dt.Frame(time_window=horas_bins)
# Concatinating two DT's

time_window_hour_dt = dt.cbind(horas,time_window_dt)
# Creating Unique hours

time_window_hour_dt_unique_dict = time_window_hour_dt[:,first(f[1:]),by(f[0])]
# Setting a key index on dt for joining 

time_window_hour_dt_unique_dict.key="hour"
# glance at unique dt values

time_window_hour_dt_unique_dict
# A new DT created after joining the 2 DT's using inner join

seatle_bikes_dt_tidy_v_1 = seatle_bikes_dt_tidy[:,:,join(time_window_hour_dt_unique_dict)]
# Check how many of time windows are existed in joined dt

seatle_bikes_dt_tidy_v_1[:,count(),by(f.time_window)]
# a new DT- bikes pass through crossings

seatle_bikes_per_crossing_dt = seatle_bikes_dt_tidy_v_1[:,{

    'missing_bike_counts':dt.sum(dt.isna(f.bike_count)),

    'bike_counts':dt.sum(f.bike_count)

    },by(f.crossing,f.time_window)

][:,{

    'time_window': f.time_window,

    'missing_bike_counts':f.missing_bike_counts,

    'bike_counts' : f.bike_counts,

    'bike_percent_cross' : dt.math.rint((f.bike_counts/dt.sum(f.bike_counts))*100)      

},by(f.crossing)]
# Top two observation of bike cross rates in each of crossings

seatle_bikes_per_crossing_dt[:2,:,by(f.crossing),sort(-f.bike_percent_cross)]
# Least two observation of bike cross rates in each of crossings

seatle_bikes_per_crossing_dt[:2,:,by(f.crossing),sort(f.bike_percent_cross)]
# A first observation per each crossing

seatle_bikes_per_crossing_dt[:,first(f[1:]),by(f.crossing)]
# A last observation per each crossing

seatle_bikes_per_crossing_dt[:,last(f[1:]),by(f.crossing)]
# Visualization

plt.figure(figsize=(12,6))

sns.barplot(x='time_window',y='bike_percent_cross',ci=None,data=seatle_bikes_per_crossing_dt.to_pandas())

plt.show()
plt.figure(figsize=(14,14))

# Facetgrid

vis_6 = sns.FacetGrid(data=seatle_bikes_per_crossing_dt.to_pandas(),

                      col='crossing',col_wrap=4,margin_titles=False)

# plotting a line plot

vis_g = vis_6.map(sns.barplot,'time_window','bike_percent_cross',color='g',order=['Morning','Mid Day','Evening','Night'])

[plt.setp(ax.texts, text="") for ax in vis_6.axes.flat]

vis_6.set_titles(col_template = '{col_name}')

plt.tight_layout()
# Data aggregations

seatle_bikes_per_crossing_days_dt = seatle_bikes_dt_tidy_v_1[:,{

    'missing_bike_counts':dt.sum(dt.isna(f.bike_count)),

    'bike_counts':dt.sum(f.bike_count)

    },by(f.crossing,f.week_day,f.hour)

][:,{

    'week_day': f.week_day,

    'missing_bike_counts':f.missing_bike_counts,

    'bike_counts' : f.bike_counts,

    'bike_percent_cross' : (f.bike_counts/dt.sum(f.bike_counts))*100     

},by(f.crossing)]
# glance at DT

seatle_bikes_per_crossing_days_dt
# visualization

plt.figure(figsize=(14,14))

# Facetgrid

vis_7 = sns.FacetGrid(data=seatle_bikes_per_crossing_days_dt.to_pandas(),

                      col='crossing',col_wrap=4,margin_titles=True)

# plotting a bar plot on grid

vis_g = vis_7.map(sns.barplot,'week_day','bike_percent_cross',color='g',ci=None,order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

# Adjusting lables on plot

[ax.set_xticklabels(ax.get_xticklabels(), rotation=65, horizontalalignment='right') for ax in vis_g.axes.flat ]

# Adjusting grid names

vis_7.set_titles(col_template = '{col_name}')

plt.tight_layout()
# Creating a DT

seatle_bikes_per_crossing_months_dt = seatle_bikes_dt_tidy_v_1[:,{

    'missing_bike_counts':dt.sum(dt.isna(f.bike_count)),

    'bike_counts':dt.sum(f.bike_count)

    },by(f.crossing,f.month)

][:,{

    'month': f.month,

    'missing_bike_counts':f.missing_bike_counts,

    'bike_counts' : f.bike_counts,

    'bike_percent_cross' : (f.bike_counts/dt.sum(f.bike_counts))*100     

},by(f.crossing)]
# Changing a month field type to string

seatle_bikes_per_crossing_months_dt['month'] = str32
# Visualization

plt.figure(figsize=(14,14))

# Facetgrid

vis_8 = sns.FacetGrid(data=seatle_bikes_per_crossing_months_dt.to_pandas(),

                      col='crossing',col_wrap=4,margin_titles=True)

# plotting a bar plot on grid

vis_g = vis_8.map(sns.barplot,'month','bike_percent_cross',color='g',ci=None,order=['1','2','3','4','5','6','7','8','9','10','11','12'])

# Adjusting lables on plot

[ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right') for ax in vis_g.axes.flat ]

# Adjusting grid names

vis_8.set_titles(col_template = '{col_name}')

plt.tight_layout()
# Creating a DT

seatle_bikes_per_crossing_day_hour_dt = seatle_bikes_dt_tidy_v_1[:,{

    'missing_bike_counts':dt.sum(dt.isna(f.bike_count)),

    'bike_counts':dt.sum(f.bike_count)

    },by(f.crossing,f.week_day,f.hour)

][:,{

    'week_day': f.week_day,

    'hour' : f.hour,

    'missing_bike_counts':f.missing_bike_counts,

    'bike_counts' : f.bike_counts,

    'bike_percent_cross' : (f.bike_counts/dt.sum(f.bike_counts))*100     

},by(f.crossing)]
# Glance

seatle_bikes_per_crossing_day_hour_dt
# Visualization

plt.figure(figsize=(14,14))

# Facetgrid

vis_9 = sns.FacetGrid(data=seatle_bikes_per_crossing_day_hour_dt.to_pandas(),

                      col='week_day', row='crossing',margin_titles=True)

# plotting a bar plot on grid

vis_g = vis_9.map(sns.barplot,'hour','bike_percent_cross',color='g',ci=None)

# Adjusting lables on plot

[ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='right') for ax in vis_g.axes.flat ]

# Adjusting grid names

vis_9.set_titles(col_template = '{col_name}')

plt.tight_layout()