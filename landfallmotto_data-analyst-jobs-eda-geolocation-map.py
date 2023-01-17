# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as ns

import plotly.express as px

import matplotlib.pyplot as plt

from geopy.geocoders import Nominatim

from tqdm.notebook import trange, tqdm
data=pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
data.head(3)
!head -1 /kaggle/input/data-analyst-jobs/DataAnalyst.csv
data.columns
data.drop(['Unnamed: 0'], axis=1, inplace=True)
data.isnull().sum()
data['Company Name']=data['Company Name'].str.replace('\\n[0-9\.]*','')

data['Company Name'].head(2)
job_counts=data['Job Title'].value_counts().rename_axis('Job Title').reset_index(name='counts')

fig = px.bar(job_counts.nlargest(10, 'counts'), x='Job Title', y='counts',

             hover_data=['counts'], color='counts',

             labels={'counts':'# of jobs'})

fig.show()
salaries=data['Salary Estimate'].str.replace('-1','0-0').str.replace(' \(Glassdoor est.\)|\$','').str.replace('K','000').str.split('-',expand=True)

salaries.columns = ['MinSalary', 'MaxSalary']



salaries.MinSalary=salaries.MinSalary.astype('int64')

salaries.MaxSalary=salaries.MaxSalary.astype('int64')



data['MinSalary']=salaries.MinSalary

data['MaxSalary']=salaries.MaxSalary 



max_df=salaries.MaxSalary.value_counts().rename_axis('MaxSalary').reset_index(name='counts')



fig = px.bar(max_df.nlargest(20, 'counts'), x='MaxSalary', y='counts',

             hover_data=['counts'], color='counts',

             labels={'counts':'# of jobs'})

fig.show()
import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Histogram(x=salaries.MinSalary,name='Min Salary'))

fig.add_trace(go.Histogram(x=salaries.MaxSalary,name='Max Salary'))





fig.update_layout(

    barmode='stack',

    title_text='Salaries Histogram',

    xaxis_title_text='Salary',

    yaxis_title_text='Count',

)

fig.show()
data['state']=data.Location.str.rsplit(',',n=1,expand=True)[1]

avg_rating_df=data.groupby('state')['Rating'].mean().reset_index(name='Average Rating')





fig = px.bar(avg_rating_df.nlargest(15, 'Average Rating'), x='state', y='Average Rating',

             hover_data=['Average Rating'], color='Average Rating',

             labels={'Average Rating':'Average Rating in State'})

fig.show()
avg_rating_df_ind=data.groupby('Industry')['Rating'].mean().reset_index(name='Average Rating')





fig = px.bar(avg_rating_df_ind.nlargest(25, 'Average Rating'), x='Industry', y='Average Rating',

             hover_data=['Average Rating'], color='Average Rating',

             labels={'Average Rating':'Average Rating in Industry'})

fig.show()
avg_rating_df_comp=data.groupby(['Company Name']).agg({'Rating':['mean'], 'MaxSalary':['mean']}).reset_index()

avg_rating_df_comp.columns = avg_rating_df_comp.columns.get_level_values(0)

top25_df=avg_rating_df_comp.sort_values(by=['MaxSalary','Rating'],ascending=False)



fig = px.bar(top25_df[top25_df.Rating>0].head(25), x='Rating', y='MaxSalary',

             hover_data=['Rating','MaxSalary','Company Name'], color='Rating',

             labels={'Company':'Company Name'})

fig.show()
def get_coordinates(loc_name):

    geolocator = Nominatim(user_agent="test user agent")

    location = geolocator.geocode(loc_name+ " US")

    return (location.latitude, location.longitude)
row_number=0

geo_locations=pd.DataFrame()





for addr in tqdm(data.Location.unique()):

    geo_locations.loc[row_number,'Location']=addr

    coord=get_coordinates(addr)

    geo_locations.loc[row_number,'Lat']=coord[0]

    geo_locations.loc[row_number,'Lot']=coord[1]

    row_number +=1
geo_locations.to_csv('Data_Analyst_Jobs_coordinates.csv', index=False)



df_count=data.Location.value_counts().rename_axis('Location').reset_index(name='counts')



job_locations=pd.merge(df_count, geo_locations, on="Location")
fig = px.scatter_mapbox(job_locations, lat="Lat", lon="Lot", size="counts",hover_name="Location",title='Interactive Job Location Density Map',

                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=3, color_discrete_sequence=["fuchsia"], mapbox_style="carto-positron")

#fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(mapbox_style="stamen-terrain") 

fig.show()
job_locations['state']=job_locations.Location.str.rsplit(',',n=1,expand=True)[1]



fig = px.treemap(job_locations, path=['state','Location'], values='counts')

fig.show()
from datetime import date



today = date.today()



Year = today.strftime("%Y")

# we treat -1 ones as new company 

data.Founded=data.Founded.apply(lambda x: int(Year) if x<0 else x)

data['CompanyAge']=int(Year)-data.Founded
fig = px.scatter_matrix(data,

    dimensions=["CompanyAge", "MaxSalary", "MinSalary","Rating"],

    color="Rating")

fig.show()
industry_df=data.groupby('Industry')['Rating'].count().reset_index(name='total_jobs')



def industry_volume(ind):

    return industry_df[industry_df.Industry==ind]['total_jobs'].to_list()[0]



data['Industry Job Size']=data.Industry.apply(lambda x: industry_volume(x) )



fig = px.scatter(data, x="Rating", y="Industry", color="Rating", size="Industry Job Size",

           hover_name="Rating", log_x=True, size_max=60)

fig.show()
fig = px.scatter_3d(data, x='Industry Job Size', y='MinSalary', z='CompanyAge',

              color='Rating')

fig.show()