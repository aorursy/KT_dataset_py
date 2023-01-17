import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline 

df = pd.read_csv('../input/restaurant-scores-lives-standard.csv')

cols = ['business_id', 'business_name', 'business_latitude', 'business_longitude', 
        'inspection_id', 'inspection_date', 'inspection_score',
        'inspection_type', 'violation_id', 'violation_description', 'risk_category']
df = df[cols]

df['inspection_date'] = pd.to_datetime(df['inspection_date'])
ax = df.groupby('inspection_date')['business_id'].nunique().plot.line(figsize = (12,6))
ax.set_title('Daily Inspections')
ax.set_xlabel("Inspection Date")
ax.set_xlabel("Number of Restaurants")
sns.despine()
# Restaurants inspected yesterday
df_yesterday = df[df.inspection_date == df.inspection_date.max()]

df_restaurants = df_yesterday.groupby(['business_id', 'business_name', 'business_latitude', 'business_longitude']).size()
df_restaurants = pd.DataFrame(df_restaurants).reset_index()

import folium

m = folium.Map(location=[df_restaurants['business_latitude'].mean(), df_restaurants['business_longitude'].mean()],
               #tiles='Stamen Toner',
               zoom_start=11)

for i in range(0, df_restaurants.shape[0]):
    folium.Marker([df_restaurants.iloc[i]['business_latitude'], 
                   df_restaurants.iloc[i]['business_longitude']], 
                  popup=df_restaurants.iloc[i]['business_name']).add_to(m)
m
