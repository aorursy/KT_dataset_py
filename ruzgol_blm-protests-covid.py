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
url="https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"

s=requests.get(url).content

df_Meta=pd.read_csv(io.StringIO(s.decode('utf-8')))
window1=7

window2=14

df_LA=df_Meta.loc[df_Meta["county"]=="Los Angeles"]



df_LA["cases_MA"+'_{}'.format(window1)]=df_LA["cases"].rolling(window1).mean()

df_LA["cases_MA"+'_{}'.format(window1)+"_Diff"]=df_LA["cases"].diff().rolling(window1).mean()

df_LA["cases_MA"+'_{}'.format(window1)+"_Diff"]=df_LA["cases_MA"+'_{}'.format(window1)+"_Diff"].shift(-window1)





df_LA["cases_MA"+'_{}'.format(window2)]=df_LA["cases"].rolling(window2).mean()

df_LA["cases_MA"+'_{}'.format(window2)+"_Diff"]=df_LA["cases"].diff().rolling(window2).mean()

df_LA["cases_MA"+'_{}'.format(window2)+"_Diff"]=df_LA["cases_MA"+'_{}'.format(window2)+"_Diff"].shift(-window2)



df_LA['date'] = pd.to_datetime(df_LA['date'], errors='coerce')

df_LA['time'] = df_LA['date'].dt.strftime('%m/%d')

df_LA.head(5)
plt.figure(figsize=(20,10)) 

plt.xticks(np.arange(0, df_LA.date.shape[0]+1, 7.0))

plt.plot(df_LA.time.iloc[40:], df_LA["cases_MA"+'_{}'.format(window1)+"_Diff"].iloc[40:],label='Time derivative of '+'{}'.format(window1)+' day moving average of Confirmed cases')

plt.plot(df_LA.time.iloc[40:], df_LA["cases_MA"+'_{}'.format(window2)+"_Diff"].iloc[40:],label='Time derivative of  '+'{}'.format(window2)+' day moving average of Confirmed cases')

plt.axvline(x="03/16",color="b",linewidth=0.5,linestyle="-.",label='03/16 California Stay at home order announced')

plt.axvline(x="05/08",color="k",linewidth=0.5,linestyle="-.",label='05/08 Phase 1 reopening')

plt.axvline(x="05/30",color="r",linewidth=0.5,linestyle="-.",label='05/30 BLM protest start')

plt.axvline(x="06/06",color="r",linewidth=0.5,linestyle="-.",label='06/06 BLM protest end')

plt.axvline(x="06/12",color="k",linewidth=0.5,linestyle="--",label='06/12 Phase 2 reopening')

plt.ylabel('Time derivative of confirmed cases')

plt.title('Lagged moving averaged of Los Angeles county confirmed cases')

plt.legend()