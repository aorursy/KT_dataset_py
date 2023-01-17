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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df=pd.read_csv("../input/uber-raw-data-apr14.csv")
print(df)
datetimedf=df.loc[:,'Date/Time']
print(datetimedf)

t=pd.to_datetime(datetimedf, format="%m/%d/%Y %H:%M:%S")
df['day']=t.dt.day
print(df)
df1=df.iloc[:,4]
df1.plot.hist(alpha=0.75, bins=30)
plt.show()
df['hour']=t.dt.hour
print(df)
df2=df.ix[:,5]
df2.plot.hist(alpha=0.75, bins=23)
plt.show()

df3=pd.read_csv('../input/Uber-Jan-Feb-FOIL.csv')
print(df3)
df3.plot.scatter(x='active_vehicles', y='trips')
plt.show()
high_trips=(df3[df3['active_vehicles']>=1600])
print(high_trips)
high_trips.describe()