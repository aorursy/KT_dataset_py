# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex2 import *

print("Setup Complete")
import pandas as pd



# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



# Fill in the line below to read the file into a variable home_data

home_data = pd.read_csv(iowa_file_path)



# Call line below with no argument to check that you've loaded the data correctly

step_1.check()



# Set column names

col_names = ["timestamp", "elb", "client:port", "backend:port", "request_processing_time", "backend_processing_time",

                "response_processing_time", "elb_status_code", "backend_status_code", "received_bytes", "sent_bytes", 

                "request", "user_agent", "ssl_cipher", "ssl_protocol"]



# Import data

# Error loading data without column names, 23 records need to be skipped and get their warnings as output (error_bad_lines=False) 

df = pd.read_csv('../input/weblogs/2015_07_22_mktplace_shop_web_log_sample.log', delim_whitespace=True, names=col_names)

#df = pd.read_csv('../input/testdata2/test_data_2.txt', delim_whitespace=True, names=col_names)
print(df.count());
df.head()
df['client'] = df['client:port'].str.split(':').str[0]
df['backend'] = df['backend:port'].str.split(':').str[0]
df.sort_values(by = ['client','timestamp'], inplace=True)
from datetime import datetime

prevs_client = 0

prev_timestamp = 0

newsession = []

session_dur = []

format = "%Y-%m-%dT%H:%M:%S.%fZ"

#df['newsession'] = [ 0 for i in df['client:port']]

for i, j in zip(df['client'],df['timestamp']):

    if i == prevs_client:

        d1 = datetime.strptime(j, format)

        d2 = datetime.strptime(prev_timestamp, format)

        if ((d1-d2).seconds) > (15 * 60):

            #print(i,prevs_client,d1,d2,(d1-d2).seconds)

            newsession.append(1)

            session_dur.append((d1-d2).seconds)

        else:

            newsession.append(0)

            session_dur.append((d1-d2).seconds)

    else:

        newsession.append(0)

        session_dur.append(0)



    prevs_client = i

    prev_timestamp = j
df1 = df.merge(pd.DataFrame(newsession).rename(columns={0:'newsession'}), left_index=True, right_index=True )
df1.head()
#Sessionize the web log by IP. Sessionize = aggregrate all page hits by visitor/IP during a session.

df1.groupby(['client']).newsession.sum().sort_index(ascending=True).nlargest()
#Find the most engaged users, ie the IPs with the longest session times in seconds

df2 = df1.merge(pd.DataFrame(session_dur).rename(columns={0:'session_dur'}), left_index=True, right_index=True )

df2.groupby(['client']).session_dur.max().sort_index(ascending=True).nlargest()
# Complete session duration from the logs

df2.session_dur.sum()
# Total number of sessions from the logs

df2.newsession.sum()
# Determine the average session time

df2.session_dur.mean() / 60
df2['URL'] = df2['request'].str.split(' ').str[1]
df2.drop(columns = ['elb','request','user_agent','ssl_cipher','client:port','backend:port'])
# Top no of URL hits from IP

df2.groupby(['client']).URL.count().sort_index(ascending=True).nlargest()
df2['timehhmmss'] = pd.to_datetime(df2['timestamp']).dt.strftime('%H:%M:%S')
df2.head()
# No of URL hits for each hour

df2.groupby(pd.to_datetime(df2['timehhmmss']).dt.strftime('%H')).URL.count().sort_index(ascending=True)
# Determine unique URL visits per session. To clarify, count a hit to a unique URL only once per session.

df2.groupby(['client','newsession','URL']).URL.count().loc[lambda x: x == 1].count() / df2.newsession.sum()
expectedload = df2.groupby(pd.to_datetime(df2['timehhmmss']).dt.strftime('%H:%M')).URL.count().reset_index(name = 'clicks')
expectedload.head()
import matplotlib.pyplot as plt

#Visualizing the Train results

plt.scatter(expectedload['timehhmmss'], expectedload['clicks'], color = 'red')

plt.plot(expectedload['timehhmmss'], expectedload['clicks'], color = 'blue')

plt.title('Salary vs Experience (Training Set Results)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
