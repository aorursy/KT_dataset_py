# most of code is copied from https://www.kaggle.com/andersonk/facebook-v-predicting-check-ins/timestamps-structure


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.signal

%matplotlib inline
plt.rcParams['figure.figsize'] = (14.0, 100.0)

def calculate_time(minutes):
    """
    0 for morning
    1 for lunch time
    2 for afternoon
    3 for dinner time
    4 for late night
    """
    time_of_day = [4,4,4,4,4,4,0,0,0,0,0,1,1,1,1,2,2,2,3,3,3,3,4,4]
    return ((minutes%(24*60*7))//(24*60))*5+time_of_day[(minutes%(24*60))//60]

train = pd.read_csv('../input/train.csv')

n,m = np.shape(train)
time = train['time']
train['TODOW'] = time.apply(calculate_time)

places_by_frequency = train.groupby('place_id')['place_id'].agg('count').sort_values(ascending=False).index.tolist()
nplaces = 20
places_by_frequency = places_by_frequency[:nplaces]

place_n = 0
time_axis = np.array([i for i in range(35)])
time_axis = np.reshape(time_axis,(35,1))
fig, axs = plt.subplots(nplaces,1)

for place_id in places_by_frequency:
  times = np.squeeze(train[train['place_id']==place_id].as_matrix(columns=['TODOW']))
  times_by_frequency = np.bincount(times)/len(times)
  times_by_frequency = np.append(times_by_frequency,[0.0]*(35-len(times_by_frequency)))
  times_by_frequency = np.reshape(times_by_frequency,(35,1))
  min_prob, max_prob = np.min(times_by_frequency),np.max(times_by_frequency)
  axs[place_n].plot(time_axis, times_by_frequency)
  axs[place_n].set_xlim([0, 34])
  axs[place_n].set_ylim([0.0, max_prob+0.01])
  axs[place_n].set_xticks(np.arange(0, 34, 1))
  axs[place_n].grid(True)
  title = 'distribution of time for place: '+str(place_id)
  axs[place_n].set_title(title)
  axs[place_n].set_xlabel('time of day, day of week')
  place_n += 1

##plt.setp([a.get_yticklabels() for a in axarr[:, 0]], visible=False)
fig.tight_layout()
plt.show()