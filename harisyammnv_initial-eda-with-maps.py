import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import plotly.express as px

import matplotlib.pyplot as plt

plt.style.use('ggplot')

pd.set_option('display.max_columns', 500)
px.set_mapbox_access_token('pk.eyJ1IjoiaGFyaXN5YW0iLCJhIjoiY2poZHRqMGV4MG93MDNkcXZqcmQ3b3RzcSJ9.V-QDWKoYu_6OqATbmH9ocw')
train_df = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv')

test_df = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv')
train_df.head(10)
train_df.info()
fig, axarr = plt.subplots(1,2, figsize=(18, 8))

sns.countplot(x='City',data=train_df,ax=axarr[0]);

axarr[0].set_title('measurements per city in Train Set')

axarr[0].set_ylabel('# of Observations in Train Set');

sns.countplot(x='City',data=test_df,ax=axarr[1]);

axarr[1].set_title('measurements per city  in Test Set')

axarr[1].set_ylabel('# of Observations in Test Set');
fig, axarr = plt.subplots(1, 2, figsize=(15, 8))

train_df.groupby(['City']).IntersectionId.nunique().sort_index().plot.bar(ax=axarr[0])

axarr[0].set_title('# of Intersections per city in Train Set')

axarr[0].set_ylabel('# of Intersections');

test_df.groupby(['City']).IntersectionId.nunique().sort_index().plot.bar(ax=axarr[1])

axarr[1].set_title('# of Intersections per city in Test Set')

axarr[1].set_ylabel('# of Intersections');
print('Number of Entry Headings in Train Set: ', len(train_df.EntryHeading.unique()))

print('Number of Exit Headings in Train Set: ', len(train_df.ExitHeading.unique()))

print('Number of Entry Street Names in Train Set: ', len(train_df.EntryStreetName.unique()))

print('Number of Exit Street Names in Train Set: ', len(train_df.ExitStreetName.unique()))



print('Number of Entry Headingds in Test Set: ', len(test_df.EntryHeading.unique()))

print('Number of Exit Headings in Test Set: ', len(test_df.ExitHeading.unique()))

print('Number of Entry Street Names in Test Set: ', len(test_df.EntryStreetName.unique()))

print('Number of Exit Street Names in Test Set: ', len(test_df.ExitStreetName.unique()))
train_intersections_count=train_df.groupby(['City','Latitude','Longitude']).IntersectionId.count().reset_index()

train_intersections_count.columns=['City','Latitude','Longitude','Count_Obs']
fig = px.scatter_mapbox(train_intersections_count[train_intersections_count.City=='Atlanta'], lat="Latitude", lon="Longitude",size="Count_Obs",color="Count_Obs",  

                        color_continuous_scale=px.colors.sequential.Viridis, size_max=15, zoom=10)

fig.update_layout(mapbox_style="open-street-map")

fig.show()
fig = px.scatter_mapbox(train_intersections_count[train_intersections_count.City=='Chicago'], lat="Latitude", lon="Longitude",size="Count_Obs",color="Count_Obs",  

                        color_continuous_scale=px.colors.sequential.Viridis, size_max=15, zoom=9)

fig.update_layout(mapbox_style="open-street-map")

fig.show()
fig = px.scatter_mapbox(train_intersections_count[train_intersections_count.City=='Philadelphia'], lat="Latitude", lon="Longitude",size="Count_Obs",color="Count_Obs",  

                        color_continuous_scale=px.colors.sequential.Viridis, size_max=15, zoom=10)

fig.update_layout(mapbox_style="open-street-map")

fig.show()
fig = px.scatter_mapbox(train_intersections_count[train_intersections_count.City=='Boston'], lat="Latitude", lon="Longitude",size="Count_Obs",color="Count_Obs",  

                        color_continuous_scale=px.colors.sequential.Viridis, size_max=15, zoom=10)

fig.update_layout(mapbox_style="open-street-map")

fig.show()