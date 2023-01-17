# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib 

from matplotlib import pyplot as plt

from math import sin, cos, sqrt, atan2, radians
df = pd.read_csv('../input/train.csv')

df.head()
def distance_bw_pickup_drop(latitude_1, longitude_1, latitude_2, longitude_2):

    R = 6373.0

    dlon = longitude_2 - longitude_1

    dlat = latitude_2 - latitude_1

    a = sin(dlat / 2)**2 + cos(latitude_1) * cos(latitude_2) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance
df['distance'] = df.apply(lambda row: distance_bw_pickup_drop(radians(row.pickup_latitude),radians(row.pickup_longitude), radians(row.dropoff_latitude), radians(row.dropoff_longitude) ), axis=1)
df.head()
plt.scatter(df['distance'], df['trip_duration'])

plt.show()