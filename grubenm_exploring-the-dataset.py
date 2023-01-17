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
df_station = pd.read_csv("../input/austin_bikeshare_stations.csv")

df_bike = pd.read_csv("../input/austin_bikeshare_trips.csv")
df_station.head()
df_bike.head()
import matplotlib.pyplot as plt
df_bike["duration_minutes"].mean()
df_bike["duration_minutes"].median()
df_bike["duration_minutes"].max()
len(df_bike[df_bike["duration_minutes"] > 500]) / len(df_bike)
plt.hist(df_bike["duration_minutes"])
plt.hist(df_bike["duration_minutes"], range=(0,500))
plt.hist(df_bike["duration_minutes"], range=(0,50))
df_bike["checkout_time"].head()
type(df_bike["checkout_time"][0])
df_bike["checkout_time"][1]
import datetime
datetime.datetime.strptime("2:06:04", "%H:%M:%S")
def hmsToDatetime(x):

    return datetime.datetime.strptime(x, "%H:%M:%S").time()
hmsToDatetime("2:06:04")
checkout = df_bike["checkout_time"].apply(hmsToDatetime)
checkout.head()
plt.hist(checkout.values)
df_bike.subscriber_type.value_counts().plot(kind='bar', figsize=(10,10))