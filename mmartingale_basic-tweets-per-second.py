import pandas as pd
import sklearn as sk
import matplotlib as mp
import scipy as sp
import numpy as np
%matplotlib inline
import arrow
import sqlite3
connection = sqlite3.connect('../input/database.sqlite')
list_of_tables = connection.execute("SELECT * FROM sqlite_master where type='table'")
print(list_of_tables.fetchall())


data = pd.read_sql('select * from Sentiment', connection)
pd.read_sql('select * from Sentiment', connection)

query =   """SELECT candidate, SUM(sentiment_confidence) / COUNT(sentiment_confidence) as AvgSentimentConfidence
             FROM Sentiment
             GROUP BY candidate
             ORDER BY AvgSentimentConfidence DESC"""
pd.read_sql(query, connection)

# Tweets per second graph
data_arr = np.array(data)
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
e =  dict( ((y,x) for (x,y) in enumerate(np.unique(data_arr[:,1]))))
candidates = [str(e[z]) for z in data_arr[:,1]]
dates = [arrow.get(x, 'YYYY-MM-DD HH:mm:ss Z') for x in  data_arr[:,-4]]


fmt = "%Y-%d-%m %H:%M:%S"
 
# Does not work on python 3 for some reason, i'll fix it later

#plt.rcParams["figure.figsize"] = (16,8)
#fig, ax = plt.subplots()
#plt.yticks(e.values(), e.keys())
#hours = mdates.HourLocator()
#fmt = mdates.DateFormatter('%H:%M')
#ax.xaxis.set_major_locator(hours)
#ax.xaxis.set_major_formatter(fmt)
#plt.title("Twitter mentions X Time (Hours)")
#plt.margins(0.01, 0.05)
#plt.plot_date(mp.dates.date2num(dates), candidates)



# That was kind of lame, i want a line graph...
query =   """SELECT candidate, sentiment, tweet_created 
             FROM Sentiment
            ORDER BY tweet_created ASC"""
sentiments = np.array(pd.read_sql(query, connection))


senti2 = np.array([(x,y, arrow.get(z, 'YYYY-MM-DD HH:mm:ss Z')) for (x,y,z) in  sentiments])

candidate_t_counters = dict()

for candidate in np.unique(senti2[:,0]):
    candidate_t_counters[candidate] = dict()
    
for tweet in senti2:
    time_d = candidate_t_counters[tweet[0]]
    time_fixed = (tweet[2].timestamp / 60) * 60
    
    if time_fixed in time_d:
        time_d[time_fixed] += 1
    else:
        time_d[time_fixed] = 1


        
for candidate in candidate_t_counters.keys():
    tot_votes = 0
    # we access by assuming time asc, make sure sorted
    k_time_dicts = sorted(candidate_t_counters[candidate].keys())
    for k_time in k_time_dicts:
        tot_votes += candidate_t_counters[candidate][k_time]
        candidate_t_counters[candidate][k_time] = tot_votes
# Clear not mentioned plot
candidate_t_counters = {k: v for (k,v) in candidate_t_counters.items() if "mentioned" not in k}
import collections
plt.rcParams["figure.figsize"] = (16,8)
plt.margins(0, 0)
for candidate in list(candidate_t_counters.keys()):
    s_p = collections.OrderedDict(candidate_t_counters[candidate])
    plt.plot(sorted(s_p.keys()), sorted(s_p.values()), label=candidate)
plt.legend(loc=2)
start_debate = arrow.get("2015-08-06 16:00-07:00")
time_adjusted = dict()
for (k,v) in candidate_t_counters.items():
        time_adjusted[k] = dict((k1, v1) for (k1, v1) in v.items() if k1 > start_debate.timestamp)
for candidate in time_adjusted.keys():
    s_p = collections.OrderedDict(time_adjusted[candidate])
    plt.plot(sorted(s_p.keys()), sorted(s_p.values()), label=candidate)
a, b = plt.xticks(list(time_adjusted['Donald Trump'].keys())[::75], map(lambda x: x.strftime("%H:%M"),map(arrow.get,list(time_adjusted[candidate].keys())[::75])))
plt.title("Tweets from 2015-08-06 16:00-07:00")
plt.legend(loc=2)