# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


tweets = pd.read_csv('../input/twcs/twcs.csv')

tweets.columns

tweets.tail(10)
# I need to get the timestamp of the tweet that the company is responding to

#Separating the original dataframe into inbounds and outbounds
inbounds = tweets.loc[tweets['inbound'] == True]
outbounds = tweets.loc[tweets['inbound'] == False]

#Merging/joining to be able to later find time between responses. Messy as a variable because the table looks so messy.
messy = pd.merge(outbounds, inbounds, left_on='in_response_to_tweet_id', right_on='tweet_id', how='outer')

#Changing timestamp format
messy['outbound_time'] = pd.to_datetime(messy['created_at_x'], format='%a %b %d %H:%M:%S +0000 %Y')
messy['inbound_time'] = pd.to_datetime(messy['created_at_y'], format='%a %b %d %H:%M:%S +0000 %Y')

#Calculating time between between outbound response and inbound message
messy['response_time'] = messy['outbound_time'] - messy['inbound_time']

messy.head()
#Making sure the data type is a timedelta/duration
print('from ' + str(messy['response_time'].dtype))

#Making it easier to later do averages by converting to a float datatype
messy['converted_time'] = messy['response_time'].astype('timedelta64[s]') / 60

print('to ' + str(messy['converted_time'].dtype))
# Getting the average response time per company.
messy.groupby('author_id_x')['converted_time'].mean()
#I saw it says 94 mins is the average time it takes for a response. This does not seem realistic.
#Focusing in on Uber and taking out outliers.

Uber = messy[messy['author_id_x'] == 'Uber_Support']
uber_times = Uber['converted_time']

uber_times.dropna()

def remove_outlier(uber_times):
    q1 = uber_times.quantile(0.25)
    q3 = uber_times.quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = uber_times.loc[(uber_times > fence_low) & (uber_times < fence_high)]
    return df_out

no_outliers = remove_outlier(uber_times)

import matplotlib.pyplot as plt
hist_plot = no_outliers.plot.hist(bins=50)
hist_plot.set_title('Uber Support Response Time')
hist_plot.set_xlabel('Mins to Response')
hist_plot.set_ylabel('Frequency')
plt.show()

print('Uber\'s average response time is ' + str(round(no_outliers.mean(),2)) + ' minutes.' )
#AskLyft

lyft = messy[messy['author_id_x'] == 'AskLyft']
lyft_times = lyft['converted_time']
lyft_times.dropna()

def remove_outlier(lyft_times):
    q1 = lyft_times.quantile(0.25)
    q3 = lyft_times.quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = lyft.loc[(lyft_times > fence_low) & (lyft_times < fence_high)]
    return df_out


lyft_no_outliers = remove_outlier(lyft_times)

import matplotlib.pyplot as plt
hist_plot = lyft_no_outliers['converted_time'].plot.hist(bins=30)
hist_plot.set_title('Lyft Support Response Time')
hist_plot.set_xlabel('Response time (min)')
hist_plot.set_ylabel('Frequency')
plt.show()

print('Lyft\'s average response time is ' + str(round(lyft_no_outliers['converted_time'].mean(),2)) + ' minutes.' )