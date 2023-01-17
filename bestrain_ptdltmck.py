# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn')

death_df=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

conf_df=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

recov_df=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

df=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df.head()
conf_df.head(10)
recov_df.head(10)
conf_df.drop(columns=['Country/Region','Province/State','Lat','Long']).sum().plot(figsize=(20,10),title='Worldwide confirmed Cases Over Time')
conf_df.groupby(['Country/Region']).sum().drop(columns=['Lat','Long']).sort_values(by=conf_df.columns[-1],ascending=False).transpose().iloc[:,:10].plot(figsize=(20,10),title='Countries with Confirmed Cases Over Time')
death_df.groupby(['Country/Region']).sum().drop(columns=['Lat','Long']).sort_values(by=death_df.columns[-1],ascending=False).transpose().iloc[:,:10].plot(figsize=(20,10))
conf_df.groupby(['Country/Region']).sum().drop(columns=['Lat','Long']).sort_values(by=conf_df.columns[-1],ascending=False).iloc[:10,:].plot.barh(y=conf_df.columns[-1],figsize=(20,10))
death_df.groupby(['Country/Region']).sum().drop(columns=['Lat','Long']).sort_values(by=['8/10/20'],ascending=False).transpose().iloc[:,:10].plot(figsize=(15,10))
temp_df=pd.DataFrame(death_df.groupby(['Country/Region']).sum()[[death_df.columns[-1]]]/conf_df.groupby(['Country/Region']).sum()[[conf_df.columns[-1]]]).rename(columns={"8/12/20": "death rate"})

temp_df['recovered rate']=pd.DataFrame(recov_df.groupby(['Country/Region']).sum()[[recov_df.columns[-1]]]/conf_df.groupby(['Country/Region']).sum()[[conf_df.columns[-1]]])
temp_df.sort_values(by=['death rate'],ascending=False)[:10].plot.barh(y=['death rate'],title='Death rate ( death/confirmed )')
temp_df.sort_values(by=['recovered rate'],ascending=False)[:10].plot.barh(y=['recovered rate'],title='Recovered rate ( recovered/confirmed )')
geo_df=df.groupby(['ObservationDate', 'Country/Region'])[['Confirmed', 'Deaths']].sum().reset_index()

geo_df.head()
fig = px.scatter_geo(geo_df, locations="Country/Region", 

                    locationmode='country names', color="Confirmed", 

                    hover_name="Country/Region", range_color=[0,geo_df.max()['Confirmed']], 

                     animation_frame='ObservationDate',

                    color_continuous_scale="Bluered", 

                      projection="natural earth",size='Confirmed',

                    title='Countries with Confirmed Cases')

fig.show()
fig = px.scatter_geo(geo_df, locations="Country/Region", 

                    locationmode='country names', color="Deaths", 

                    hover_name="Country/Region", range_color=[0,geo_df.max()['Deaths']], 

                     animation_frame='ObservationDate',

                    color_continuous_scale="Bluered", 

                      projection="natural earth",size='Deaths',

                    title='Countries with Death Cases')

fig.show()