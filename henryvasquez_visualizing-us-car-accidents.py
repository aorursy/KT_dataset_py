# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#start by loading data set and looking at first few rows

data = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_Dec19.csv')

data.head()
#See how many values are missing from each column - doesn't matter for some variables since they're not that important

for col in range(0, len(list(data))):

    print('{column} has {nas} missing values'.format(column = list(data)[col], nas = data[[list(data)[col]]].isnull().sum().sum()))
#first create list of NA percentages

na_percentages = []

lam = lambda x: np.round(100*data[[x]].isnull().sum().sum()/len(data),2)



for col in list(data):

    na_percentages.append(lam(col))



#create data frame

plotdata = pd.DataFrame({'Variable': list(data),

                        'NA Percentage': na_percentages})



#let's go ahead and factor out the variables that have 0% - sort values- reset index

plotdata = plotdata[plotdata['NA Percentage'] > 0.00].sort_values('NA Percentage', ascending = True).reset_index(drop = True)



#now plot

plt.barh(plotdata['Variable'], plotdata['NA Percentage'])

plt.title('NA Percentages by Column') #title

plt.xlabel('Percentage of NAs') #xaxis label

plt.ylabel('Variable') #yaxis label

plt.xticks(rotation = 90) #rotate ticks so they're legible

plt.show()
#get unique states-49

states = list(set(data['State']))

counts = []



#get counts for each state

for state in states:

    counts.append(len(data[data['State'] == state]))



# create df - sort by 

state_plotdata = pd.DataFrame({'State': states,

                        'Total': counts}).sort_values('Total', ascending = False).reset_index(drop = True)



#now plot

plt.bar(state_plotdata['State'], state_plotdata['Total'])

plt.title('Accidents by State') #title

plt.ylabel('State') #yaxis label

plt.xlabel('Total Accidents') #xaxis label

plt.xticks(rotation = 90)

plt.show()

#start by importing necessary packages

import datashader as ds

import datashader.transfer_functions as tf

from datashader.utils import lnglat_to_meters

from datashader.colors import colormap_select, inferno

from datashader.utils import export_image

from functools import partial
#we need to tranform lat and lng to meters from origin using lnglat_to_meters, we'll set this to x and y

data.loc[:, 'x'], data.loc[:, 'y'] = lnglat_to_meters(data.Start_Lng, data.Start_Lat)



#now set the image

background = 'black'

plot_width = int(2500)

plot_height = int(plot_width*7.00/12)

export = partial(export_image, background = background, export_path = 'export')

cm = partial(colormap_select, reverse=(background!='black'))

cvs = ds.Canvas(plot_width, plot_height)

agg = cvs.points(data, 'x', 'y')

export(tf.shade(agg, cmap = cm(inferno), how = 'log'), 'US_car_crash_by_state')
#first create a column that states color based on Severity

data['Color'] = np.where(data['Severity'] == 1, '#ffa600',

                         np.where(data['Severity'] == 2, '#ff6e54',

                                  np.where(data['Severity'] == 3, '#955196', 

                                           '#003f5c')

                                 )

                          )

#if color isn't a categorical var, you run into an issue later with tf.shade()

data['Color'] = pd.Categorical(data['Color'])



#begin plot

background = 'white'

export = partial(export_image, background = background, export_path = 'export')

cvs = ds.Canvas(plot_width*2, plot_height*2)

agg = cvs.points(data, 'x', 'y', ds.count_cat('Color'))

view = tf.shade(agg, color_key = data['Color'])

export(tf.spread(view, px = 2), 'US_car_crash_severity')

#check severity totals

data.groupby(['Severity']).size()