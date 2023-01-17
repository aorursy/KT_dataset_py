# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import plotly modules
import plotly.graph_objs as pgh
# Input data files are available in the "../input/" directory.
#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
   # for filename in filenames:
   #     print(os.path.join(dirname, filename))

df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
dateF = df.copy()
dateF['Date_time'] = df['ObservationDate'].astype('datetime64[ns]')

df = df.rename(columns={"ObservationDate":"Date"})

#groupby the country and the date
df_countries = df.groupby(['Country/Region','Date']).sum().reset_index().sort_values('Date', ascending=False)

#drop duplicates and check there's at least one confirmed case
df_countries = df_countries.drop_duplicates(subset=['Country/Region'])
df_countries = df_countries[df_countries['Confirmed'] != 0]

#create the Choropleth Map
fig1 = pgh.Figure(data=pgh.Choropleth(locations = df_countries['Country/Region'], locationmode='country names', 
    z = df_countries['Confirmed'], colorscale='sunset',marker_line_color='black', marker_line_width=0.5))

total_cases = df_countries['Confirmed'].sum()
total_cases = int(total_cases)
total_cases =format(total_cases, "7,d")
#add the titles
fig1.update_layout(title_text=total_cases+' Confirmed Cases From Jan 22 - March 9, 2020', title_x=0.75, 
    geo=dict(showframe=False, showcoastlines=True,projection_type='equirectangular'))

dateF['Date_time'] = df['Date'].astype('datetime64[ns]')
print("First case in Dataset:")
print(dateF.Date_time.min())
print("Last case in Dataset:")
print(dateF.Date_time.max())

