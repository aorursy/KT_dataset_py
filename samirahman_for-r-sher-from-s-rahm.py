#Dataframe Packages
import numpy as np 
import pandas as pd 
import random as rnd


import os
print(os.listdir("../input"))
# visualisation and graphs
import seaborn as sns
sns.set(color_codes=True)

import matplotlib.pyplot as plt
%matplotlib inline

from scipy import stats, integrate
np.random.seed(sum(map(ord, "distributions")))

#Date and Time packages
import datetime as dt
import time
from datetime import datetime
from dateutil.parser import parse

main_file_path = '../input/Essence.csv'
data = pd.read_csv(main_file_path)
print('Sami')
data.info()
eventgroup = data.groupby(by='event')
eventgroup.nunique()
#Unique Cookies
10 + 111233 + 26122 + 12894
eventgroup.count()
#No. of Cookies
data.dtypes
data['time'] = pd.to_datetime(data.time)
data.dtypes
#See if the time variable has been converted to a datetime object
data['hour'] = data.time.dt.hour
data[data.event== 'CONVERSION'].hour.value_counts().sort_index()
#Create hour variable and see an index of counts per hour based on the conversion subvariable 
data[data.event== 'CONVERSION'].hour.value_counts().sort_index().plot()
data['weekday'] = data.time.dt.weekday_name
data['weekday'] = pd.Categorical(data['weekday'], categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], ordered=True)
#Create weekday variable and use a categorical function to order them correctly rather than alphabetically 
data[data.event== 'CONVERSION'].weekday.value_counts().sort_index()
data[data.event== 'CONVERSION'].weekday.value_counts().sort_index().plot(kind= 'bar')
data[data.event== 'CONVERSION'].weekday.value_counts().sort_index().plot()
#slap a bar graph and line plot in there
m = data.event.str.contains('CONVERSION').groupby(data['user-id']).transform('any')
exposure = data[m]
#Filter data if it contains conversions dropping groups without a sale
first = exposure.groupby(exposure['user-id'])
second = first.hour.max() - first.hour.min()
second.mean()
x = second
sns.distplot(x);
data = pd.read_csv(main_file_path)

event_count  = data['user-id'].value_counts()
event_count = event_count[:4,]
plt.figure(figsize=(10,5))
sns.barplot(event_count.index, event_count.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('user-id', fontsize=0.1)

plt.show()
import pandas as pd
data["event"] = data["event"].astype('category')
one_hot = pd.get_dummies(data['event'])

one_hot.corr()
