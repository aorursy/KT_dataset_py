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

import plotly.graph_objects as go
ghana_health_df = pd.read_csv('/kaggle/input/health-facilities-gh/health-facilities-gh.csv')
region= ghana_health_df.groupby(['Region']).size().to_frame('region_count').reset_index()

region
fig = go.Figure(data=[go.Pie(labels=region['Region'], values=region['region_count'].values)])

fig.show()
type_df= ghana_health_df.groupby(['Type']).size().to_frame('type_count').reset_index()

type_df.sort_values(by='type_count', ascending =False)[:10]
type_df.sort_values(by='type_count', ascending =False)[:10].plot('Type','type_count',kind='bar')
owner_df= ghana_health_df.groupby(['Ownership']).size().to_frame('owner_count').reset_index()

owner_df.sort_values(by='owner_count', ascending =False)
owner_df.sort_values(by='owner_count', ascending =False).plot('Ownership','owner_count',kind='bar',color='C2')
region_owner_df= ghana_health_df.groupby(['Region','Ownership']).size().to_frame('count').reset_index()

region_owner_df
df= region_owner_df.groupby(['Region','Ownership']).sum().unstack('Ownership')

df.columns = df.columns.droplevel()

df.plot(kind='barh', stacked=True, figsize = (15,8))
region_owner_df[region_owner_df['Ownership']=='Government'].sort_values(by='count').plot('Region','count',kind='barh',color='C3')
region_owner_df[region_owner_df['Ownership']=='Private'].sort_values(by='count').plot('Region','count',kind='barh',color='C4')
region_owner_df[region_owner_df['Ownership']=='CHAG'].sort_values(by='count').plot('Region','count',kind='barh',color='C8')
region_owner_type_df= ghana_health_df.groupby(['Region','Ownership','Type']).size().to_frame('count').reset_index()

region_owner_type_df
region_district_df= ghana_health_df.groupby(['Region','District']).size().to_frame('count').reset_index()

region_district_df.sort_values(by='count', ascending=False)
ghana_health_df.plot(kind="scatter", x="Longitude", y ="Latitude", s=20, alpha= 0.4)

plt.show()
facility= ghana_health_df.groupby(['FacilityName']).size().to_frame('facility_count').reset_index()

facility.sort_values(by='facility_count', ascending =False)[:10]
town= ghana_health_df.groupby(['Town','Region']).size().to_frame('town_count').reset_index()

town.sort_values(by='town_count', ascending =False)[:10]
town.sort_values(by='town_count', ascending =False)[:10].plot('Town','town_count',kind='bar',color='C9')