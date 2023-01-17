import matplotlib.pyplot as plt

import plotly.plotly as py

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline

plt.rcParams['figure.figsize']=(12,5)

df=pd.read_csv("../input/seattleWeather_1948-2017.csv")



# Any results you write to the current directory are saved as output.
df.head()
#Add the temperature in celsius too 

df["TMAX_CELSIUS"]=df["TMAX"].apply(lambda x: (x-32)*(5.0/9.0))

df["TMIN_CELSIUS"]=df["TMIN"].apply(lambda x: (x-32)*(5.0/9.0))

df.head()
df.RAIN.groupby(df.RAIN).count().plot(kind="bar")
df['date'] = pd.to_datetime(df['DATE'])

ax=df.plot(x='date', y=["TMAX","TMIN"])

df['date'] = pd.to_datetime(df['DATE'])

ax=df.plot(x='date', y=["TMAX_CELSIUS","TMIN_CELSIUS"])