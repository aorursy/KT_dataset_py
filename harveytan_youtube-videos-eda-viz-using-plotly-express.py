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
# pulling in the main datasets

data_gb = pd.read_csv('/kaggle/input/youtube-new/GBvideos.csv') # uk

data_ca = pd.read_csv('/kaggle/input/youtube-new/CAvideos.csv') # canada

data_us = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv') # us

data_in = pd.read_csv('/kaggle/input/youtube-new/INvideos.csv') # india

data_de = pd.read_csv('/kaggle/input/youtube-new/DEvideos.csv') # germany



# add new columns to identify the region

data_gb['region'] = 'UK'

data_ca['region'] = 'Canada'

data_us['region'] = 'US'

data_in['region'] = 'India'

data_de['region'] = 'Germany'



# to make sure that the structure of the datasets are the same

print(data_gb.columns == data_ca.columns)

print(data_gb.columns == data_us.columns)

print(data_gb.columns == data_in.columns)

print(data_gb.columns == data_de.columns)
data_us.info()
data_us.sample(5)
#pulling in the category datasets

data_gb_cat_json = pd.read_json('/kaggle/input/youtube-new/GB_category_id.json')

data_gb_cat = pd.json_normalize(data_gb_cat_json['items'])



data_ca_cat_json = pd.read_json('/kaggle/input/youtube-new/CA_category_id.json')

data_ca_cat = pd.json_normalize(data_ca_cat_json['items'])



data_us_cat_json = pd.read_json('/kaggle/input/youtube-new/US_category_id.json')

data_us_cat = pd.json_normalize(data_us_cat_json['items'])



data_in_cat_json = pd.read_json('/kaggle/input/youtube-new/IN_category_id.json')

data_in_cat = pd.json_normalize(data_in_cat_json['items'])



data_de_cat_json = pd.read_json('/kaggle/input/youtube-new/DE_category_id.json')

data_de_cat = pd.json_normalize(data_de_cat_json['items'])



# change the id column to 'int64' so that we can match them in the merge function below (need to have the same datatype)

data_gb_cat['id'] = data_gb_cat['id'].astype('int64')

data_ca_cat['id'] = data_ca_cat['id'].astype('int64')

data_us_cat['id'] = data_us_cat['id'].astype('int64')

data_in_cat['id'] = data_in_cat['id'].astype('int64')

data_de_cat['id'] = data_de_cat['id'].astype('int64')
# these are the columns we will keep

cols_to_keep = ['region','title', 'channel_title','snippet.title','publish_time','views','likes','dislikes','comment_count','comments_disabled','ratings_disabled']
#merging the main datasets and the category datasets

data_gb_new = data_gb.merge(data_gb_cat, left_on='category_id', right_on='id', how='left')

data_gb_new = data_gb_new[cols_to_keep]



data_ca_new = data_ca.merge(data_gb_cat, left_on='category_id', right_on='id', how='left')

data_ca_new = data_ca_new[cols_to_keep]



data_us_new = data_us.merge(data_gb_cat, left_on='category_id', right_on='id', how='left')

data_us_new = data_us_new[cols_to_keep]



data_in_new = data_in.merge(data_gb_cat, left_on='category_id', right_on='id', how='left')

data_in_new = data_in_new[cols_to_keep]



data_de_new = data_de.merge(data_gb_cat, left_on='category_id', right_on='id', how='left')

data_de_new = data_de_new[cols_to_keep]
data_final = pd.concat([data_gb_new, data_ca_new, data_us_new, data_in_new, data_de_new], axis=0)
data_final.rename(columns={'snippet.title' : 'video_category'}, inplace=True)
data_final.shape[0] - data_final.count() # nan values under 'video category' column
data_final[data_final['video_category'].isnull() == True]
# fill the nan values under 'video_category' with 'No category'

data_final['video_category'].fillna('No category', inplace=True)
data_final['upload_date'] = pd.to_datetime(data_final['publish_time'].str[:10], format='%Y-%m-%d')

data_final['upload_date_year_month'] = pd.to_datetime(data_final['publish_time'].str[:7], format='%Y-%m')



data_final['upload_year'] = data_final['upload_date'].dt.year

data_final['upload_month'] = data_final['upload_date'].dt.month



data_final.drop('publish_time', axis=1, inplace=True)
data_final['likes_%_of_view'] = (data_final['likes'] / data_final['views']) * 100 # percentage of likes as a proportion of views

data_final['dislikes_%_of_view'] = (data_final['dislikes'] / data_final['views']) * 100 # percentage of dislikes as a proportion of views

data_final['comment_%_of_view'] = (data_final['comment_count'] / data_final['views']) * 100 # percentage of comment count as a proportion of views



data_final['comments_disabled'].replace({False : 'No', True:'Yes'}, inplace=True)

data_final['ratings_disabled'].replace({False : 'No', True:'Yes'}, inplace=True)
data_final.info()
data_final.sample(5)
data_final.groupby('upload_year')[['views','likes','dislikes','comment_count']].sum().reset_index()



# observations:

#1 from 2016 to 2017, major increase in views
a = data_final.pivot_table(index='upload_year', columns='region', aggfunc='sum', values='views')

a[[i for i in a.columns if i not in ['region','upload_year']]] = a[[i for i in a.columns if i not in ['region','upload_year']]] / 1000000

a.rename(columns=lambda x : x + ' Views in M', inplace=True)



# generating the percentage change in views for each year for each region

b = a[[i for i in a.columns if i not in ['region','upload_year']]].pct_change() * 100

b.rename(columns=lambda x : x + ' % Change', inplace=True)



c = pd.concat([a,b], axis=1).sort_index(axis=1)

c.round(1)



# observations

#1 2017 and 2018 was where most regions experienced 'significant' growth in viewership of Youtube videos
data_selected_year = data_final[data_final['upload_year'].isin([2017,2018])]

data_selected_year_by_year = data_selected_year.groupby(['upload_date_year_month','region'])[['views']].sum().reset_index()





fig = px.line(data_selected_year_by_year, x='upload_date_year_month', y='views', color='region')

fig.update_layout(title_text='Views Across Regions from 2017 to 2018', title_font_size=20, legend_title_text='Region',\

                 legend=dict(

                            yanchor='top',

                            y=0.99,

                            xanchor='right',

                            x=0.999)

                 )



fig.update_xaxes(showgrid = False)

fig.update_yaxes(showgrid = False)



fig.show()
fig = px.box(data_final, x = 'comments_disabled', y='likes_%_of_view', labels={'likes_%_of_view' : 'Percentage of Likes as a Proportion of Video Views'})



fig.update_layout(title_text='Distribution of Likes % between Videos with Comments Enabled/Disabled')



fig.update_xaxes(showgrid = False)

fig.update_yaxes(showgrid = False)



fig.show()
fig = px.scatter(data_final, x='views',y = 'likes', color='region', facet_col='region', labels={'views':'Number of Views', 'likes':'Number of Likes'}, marginal_x ='histogram', height=1000, trendline='ols', trendline_color_override='black')



fig.update_layout(title_text='Visual View Between Number of Views and Likes Across Regions', title_font_size=20, showlegend=False)



fig.update_xaxes(showgrid = False)

fig.update_yaxes(showgrid = False)



fig.show()
fig = px.scatter(data_final, x='likes_%_of_view',y = 'views', color='region', facet_col='region', labels={'views':'Number of Views', 'likes_%_of_view':'% of Likes'}, marginal_x ='histogram', height=1000)



fig.update_layout(title_text='Visual View Between Number of Views and and Percentage of Likes Across Regions', title_font_size=20, showlegend=False)



fig.update_xaxes(showgrid = False)

fig.update_yaxes(showgrid = False)



fig.show()
fig = px.treemap(data_final, path=['video_category','region'], values='views', color='region')



fig.update_layout(title_text='Breakdown of Video Views by Category and Region', title_font_size=20, showlegend=False)



fig.show()
fig = px.scatter(data_final, x='views',y = 'likes', color='region', facet_col='video_category', labels={'views':'Number of Views', 'likes':'Number of Likes'},facet_col_wrap = 4, height = 1000, opacity = 0.5)

fig.update_layout(title_text='Visual View Between Number of Views and Likes Across Video Categories', title_font_size=20)



fig.update_xaxes(showgrid = False)

fig.update_yaxes(showgrid = False)



fig.show()