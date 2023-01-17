import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import datetime

from collections import Counter

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
i = pd.read_csv('../input/911.csv')
i.info()
i.head()
i['isFirstHalfOfTheDay'] = i.timeStamp.apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").time() <= datetime.time(12,0))

i['isNight'] = i.timeStamp.apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").time() >= datetime.time(19,0))
# Lets plot the counts for the two halves of the day

sns.countplot(i.isFirstHalfOfTheDay)
# Lets extract the type and cause of the emergency

i['Type'] = i.title.apply(lambda x : x.split(':')[0].strip())

i['Cause'] = i.title.apply(lambda x : x.split(':')[1].strip('- '))
i.Type.unique()
i.Cause.unique()
## Now lets see the count plots based on type of the emergency

sns.countplot(i.Type)
## Now lets see the counts of emergencies by splitting the time into different time



def get_timeslice(timestamp):

    if timestamp.time() < datetime.time(6,0):

        return 0

    elif timestamp.time() >= datetime.time(6,0) and timestamp.time() < datetime.time(12,0):

        return 1

    elif timestamp.time() >= datetime.time(12,0) and timestamp.time() < datetime.time(18,0):

        return 2

    else:

        return 3
## Now lets split the 24hrs time into 4 zones and run a count plot on it.

i['timeSlice'] = i.timeStamp.apply(lambda x : get_timeslice(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))) 
sns.countplot(i.timeSlice)
a = i.groupby(['timeSlice', 'Type']).e.sum()
a.unstack().plot(kind='bar', stacked=True)
i[i.timeSlice == 2].Cause.value_counts()[:10]
# Here's another cool way to see the distribution of emergencies wrt Type and timeSlice

sns.heatmap(a.unstack())
encoder = preprocessing.OneHotEncoder()

b = encoder.fit_transform(i.timeSlice.reshape(-1,1))
r = LogisticRegression(multi_class='multinomial', solver='lbfgs')

r.fit(b, i.Type)
sum(r.predict(b) == i.Type)/len(i.Type)