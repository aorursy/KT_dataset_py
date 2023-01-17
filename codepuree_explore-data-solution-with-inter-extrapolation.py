# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

df.head()
df.info()
df.describe()
plt.title('Plotting pickup & dropoff')

plt.scatter(df['pickup_longitude'], df['pickup_latitude'], label='pickup')

plt.scatter(df['dropoff_longitude'], df['dropoff_latitude'], label='dropoff')

plt.legend()
def distance_euclidean(lat1, lon1, lat2, lon2):

    return np.sqrt(np.power(lat1 - lat2, 2) + np.power(lon1 -lon2, 2))
df['distance_euclidean'] = df.apply(lambda x: distance_euclidean(x['pickup_latitude'], x['pickup_longitude'], x['dropoff_latitude'], x['dropoff_longitude']), axis=1)
def distance_manhattan(lat1, lon1, lat2, lon2):

    return np.abs(lat2 - lat1) + np.abs(lon2 - lon1)
df['distance_manhattan'] = df.apply(lambda x: distance_manhattan(x['pickup_latitude'], x['pickup_longitude'], x['dropoff_latitude'], x['dropoff_longitude']), axis=1)
df.head()
plt.title('Distance & Time')

plt.scatter(df['distance_manhattan'], df['trip_duration'], label='Manhattan distance')

plt.scatter(df['distance_euclidean'], df['trip_duration'], label='Euclidean distance')

plt.xlabel('Distance')

plt.ylabel('Duration')

plt.legend()
df_zero_less = df[df['distance_manhattan'] > 0.0]



print('Removed {} ({:.2f}%) trips, where the distance traveled were zero.'.format(len(df) - len(df_zero_less), (len(df) - len(df_zero_less)) / len(df) * 100))
for i in  range(1, 24):

    # Trips longer than 12 hours

    hours_max = i * 60 * 60 # in seconds

    trips_longer = df[df['trip_duration'] > hours_max]

    print('There are {} ({:.2f}%) trips, which are longer than {} hours.'.format(len(trips_longer), len(trips_longer) / len(df) * 100, i))
hours_max = 4 * 60 * 60 # in seconds

df_shorter_fourh = df_zero_less[df_zero_less['trip_duration'] < hours_max]

print('Removed {} ({:.2f}%) trips from dataframe.'.format(len(df_zero_less) - len(df_shorter_fourh), (len(df_zero_less) - len(df_shorter_fourh)) / len(df_zero_less) * 100))
df_cleaned = df_shorter_fourh

df_cleaned.describe()
plt.title('Distance & Time')

plt.scatter(df_cleaned['distance_manhattan'], df_cleaned['trip_duration'], label='Manhattan distance')

plt.scatter(df_cleaned['distance_euclidean'], df_cleaned['trip_duration'], label='Euclidean distance')

plt.xlabel('Distance')

plt.ylabel('Duration')

plt.legend()
fig = plt.figure()

ax = Axes3D(fig)

df_time = df_cleaned['pickup_datetime'].apply(lambda x: int(x[11:13]))

ax.scatter(df_cleaned['trip_duration'], df_time, df_cleaned['distance_manhattan'], label='Manhattan distance')

# ax.scatter(df_cleaned['distance_euclidean'], df_cleaned['trip_duration'], df_time, label='Euclidean distance')

ax.set_title('Distance, Duration & pickup_datetime')

ax.set_xlabel('Duration')

ax.set_ylabel('pickup_datetime')

ax.set_zlabel('Distance')

ax.legend()