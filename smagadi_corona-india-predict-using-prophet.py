# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fbprophet import Prophet

	

# Default value of display.max_rows is 10 i.e. at max 10 rows will be printed.

# Set it None to display all rows in the dataframe

pd.set_option('display.max_rows', None)
df = pd.read_csv('../input/covid19-corona-virus-india-dataset/complete.csv')





df.head()
df.rename(columns={ 'Country/Region':'Country','Total Confirmed cases (Indian National)':'Confirmed_Indians','Total Confirmed cases ( Foreign National )':'Confirmed_Foriegners'}, inplace=True)

df.rename(columns={'Name of State / UT':'State','Cured/Discharged':'Cured'}, inplace=True)

sum_column = df["Confirmed_Indians"] + df["Confirmed_Foriegners"]

df["Total_confirmed_cases"] = sum_column

df.head()
df['Date']=pd.to_datetime(df.Date)

df_dropped = df.drop(columns=['State', 'Confirmed_Indians','Cured','Latitude','Longitude','Death','Confirmed_Foriegners'])

gr_df = df_dropped.groupby(['Date'])['Total_confirmed_cases'].sum().reset_index() 





gr_df = gr_df.astype({"Total_confirmed_cases": float})

gr_df.rename(columns={'Total_confirmed_cases':'y','Date':'ds'}, inplace=True)

gr_df.count






m=Prophet()

m.fit(gr_df)

future = m.make_future_dataframe(periods=60)

future.tail()



forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)



# xaxis is the time

# y axis gives the confirmed cases 
