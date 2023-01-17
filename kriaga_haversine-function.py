# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def haversine_np(lon1, lat1, lon2, lat2):

    """

    Calculate the great circle distance between two points

    on the earth (specified in decimal degrees)



    All args must be of equal length.    



    """

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2



    c = 2 * np.arcsin(np.sqrt(a))

    km = 6367 * c

    return km

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

kr = pd.read_csv("../input/train.csv")

kr.describe()
lon1 = kr['pickup_longitude']

lat1 = kr['pickup_latitude']

lon2 = kr['dropoff_longitude']

lat2 = kr['dropoff_latitude']

lon1
#lon1, lon2, lat1, lat2 = -73.98812866,-73.99017334,40.73202896,40.75667953;

km=[]

for i in range(0,100):

    df = pd.DataFrame(data={'lon1':[lon1[i]],'lat1':[lat1[i]],'lon2':[lon2[i]],'lat2':[lat2[i]]})

    km.append(haversine_np(df['lon1'],df['lat1'],df['lon2'],df['lat2']))

km