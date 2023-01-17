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
import googlemaps

import pandas as pd

test=pd.read_csv('../input/test.csv')

print(test.info())

oris = test.pickup_latitude.astype(str) + ',' + test.pickup_longitude.astype(str) 

dest = test.dropoff_latitude.astype(str) + ',' + test.dropoff_longitude.astype(str)

dur = []

gmaps = googlemaps.Client(key='Your_Apikey')
for i in range(625134):

    distance = gmaps.distance_matrix(origins = oris[i],destinations = dest[i])

    durations = distance['rows'][0]

    durations = durations['elements'][0]

    durations = durations['duration']['value']

    dur.append(durations)