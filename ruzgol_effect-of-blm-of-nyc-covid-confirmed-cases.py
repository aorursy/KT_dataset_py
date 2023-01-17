# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import io

import requests

import datetime as dt

import matplotlib.pyplot as plt



import datetime

import matplotlib.dates as mdates



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read the meta data from NY Times Github page

url="https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"

s=requests.get(url).content

df_Meta=pd.read_csv(io.StringIO(s.decode('utf-8')))
window1=7

lag_window1=3

window2=14

lag_window2=7

df_MP=df_Meta.loc[(df_Meta["county"]=="New York City")]

df_MP=df_Meta.loc[(df_Meta["county"]=="Multnomah")]



df_MP["cases_MA"+'_{}'.format(window1)]=df_MP["cases"].rolling(window1).mean()

df_MP["cases_MA"+'_{}'.format(window1)+"_Diff"]=df_MP["cases"].diff().rolling(window1).mean()

df_MP["cases_MA"+'_{}'.format(window1)+"_Diff"]=df_MP["cases_MA"+'_{}'.format(window1)+"_Diff"].shift(-lag_window1)





df_MP["cases_MA"+'_{}'.format(window2)]=df_MP["cases"].rolling(window2).mean()

df_MP["cases_MA"+'_{}'.format(window2)+"_Diff"]=df_MP["cases"].diff().rolling(window2).mean()

df_MP["cases_MA"+'_{}'.format(window2)+"_Diff"]=df_MP["cases_MA"+'_{}'.format(window2)+"_Diff"].shift(-lag_window2)



df_MP['date'] = pd.to_datetime(df_MP['date'], errors='coerce')

df_MP['time'] = df_MP['date'].dt.strftime('%m/%d')

df_MP.head(5)
plt.figure(figsize=(20,10)) 

plt.xticks(np.arange(0, df_MP.date.shape[0]+1, 7.0))

plt.plot(df_MP.time.iloc[50:], df_MP["cases_MA"+'_{}'.format(window1)+"_Diff"].iloc[50:],label='Minus '+'{}'.format(lag_window1)+'-day lagged time derivative of '+'{}'.format(window1)+' day moving average of Confirmed cases')

plt.plot(df_MP.time.iloc[50:], df_MP["cases_MA"+'_{}'.format(window2)+"_Diff"].iloc[50:],label='Minus '+'{}'.format(lag_window2)+'-day lagged time derivative of '+'{}'.format(window2)+' day moving average of Confirmed cases')

#plt.axvline(x="03/26",color="b",linewidth=0.5,linestyle="-.",label='03/26 Minnesota Stay at home order announced')

plt.axvline(x="06/15",color="k",linewidth=0.5,linestyle="-.",label='06/15 Phase 1 reopening')

plt.axvline(x="05/25",color="r",linewidth=0.5,linestyle="-.",label='05/25 George Floyd died')

plt.axvline(x="06/22",color="k",linewidth=0.75,linestyle="--",label='06/22 Phase 2 reopening')

plt.ylabel('Time derivative of confirmed cases',fontsize=16)

plt.title('Lagged moving averaged of New York City time derivative confirmed cases',fontsize=16)

plt.legend(loc=2, prop={'size': 16})
df_Meta.loc[(df_Meta["county"]=="Multnomah")]