#importing the necessary modules into the notebook



import pandas

import time

import numpy as np

from collections import Counter

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#Reading in the csv and converting to dataframe. 



csv_file = r'../input/911.csv'

data = pandas.read_csv(csv_file)

data.head()
#Let's first see the most frequently occuring time for cop calls 



timesp = data['timeStamp']

time_2 = [str(i) for i in timesp]

time_only = [int(i[11:13]) for i in time_2]
#between 4 and 5 p.m. is the time most calls to 911 are placed. (2015-2017)



time_count = Counter(time_only).most_common()
#plot of frequency of occurence of calls by time

#as you can see there aren't many calls from midnight to around 6 am after which it spikes till 5 pm.



plt.hist(time_only, bins=100)

plt.xlabel('Time of day')

plt.ylabel('Amount of calls recieved')

plt.show()
#we shall now find the most frequently occuring kind of emergency call



call_type = [i for i in data['title']]

tp_count = Counter(call_type).most_common()



#vehicle accident is the most frequent kind of emergency
#bar chart showing frequency of call types

#as you can see it's imbalanced with the vehicle accidents dominating the chart. 



labels, values = zip(*Counter(call_type).items())



indexes = np.arange(len(labels))

width = 1



plt.bar(indexes, values, width)

plt.xlabel('TYPE OF ACCIDENT')

plt.ylabel('FREQUENCY')

#plt.xticks(indexes + width * 0.5, labels)

plt.show()
township = [i for i in data['twp']]



labels, values = zip(*Counter(township).items())



indexes = np.arange(len(labels))

width = 1



plt.bar(indexes, values, width)

plt.xlabel('Township')

plt.ylabel('Frequency of all calls')

#plt.xticks(indexes + width * 0.5, labels)

plt.show()
merion_df = data[data['twp']=='LOWER MERION']



merion_call_count = Counter(merion_df['title']).most_common()



merion_count_df = pandas.DataFrame()

type_lst = [i[0] for i in merion_call_count]

count_lst = [i[1] for i in merion_call_count]

merion_count_df['call type'] = type_lst

merion_count_df['frequency'] = count_lst



merion_count_df
time_slice = [i[11:13] for i in data['timeStamp']]

data_time = data.copy()

data_time['time'] = time_slice
#building average number of accidents in category per day. 



date = [i[:10] for i in data['timeStamp']]

data_time['date'] = date

date_set = set()

for i in data_time['date']:

    date_set.add(i)

    

day_by_day = []

for i in date_set:

    day_by_day.append(Counter(data_time['title'][data_time['date'] == i]))



title_set = set()

for i in data_time['title']:

    title_set.add(i)



title_set_lst = [i for i in title_set]
#function to extract daily average given input emergency type (title)





def get_title_mean(x):

    title_eg = []

    for i in day_by_day:

        title_eg.append(i[x])

    

    title_np = np.array(title_eg)

    return title_np.mean()
#Dataframe of daily average of each type of emergency. 



titles = []

averages = []

for i in title_set_lst:

    titles.append(i)

    averages.append(get_title_mean(i))



avgs_df = pandas.DataFrame()

avgs_df['Emergency Type'] = titles

avgs_df['Daily Average'] = averages



avgs_df
#function to give you the total frequency of call type based on the range of time of day by the hour. 

#for example, a total of 7006 vehicle accidents have occured between 12pm and 3pm from Dec 2015 to Feb 2017. 



def get_freq_by_time(start_t, end_t):

    emr = []

    emr_df = data_time['title'].as_matrix()

    time_df = data_time['time'].as_matrix()

    for i, b in zip(emr_df, time_df):

        if int(b) in range(start_t, end_t):

            emr.append(i)

    freq = Counter(emr).most_common()

    freq_titles = [i[0] for i in freq]

    freq_freq = [i[1] for i in freq]

    freq_df = pandas.DataFrame()

    freq_df['Type of Emergency'] = freq_titles

    freq_df['Frequency'] = freq_freq

    

    plt.hist(freq_freq, bins=110)

    return freq_df

    plt.show()

    
#example of above function in action 



get_freq_by_time(12, 16)