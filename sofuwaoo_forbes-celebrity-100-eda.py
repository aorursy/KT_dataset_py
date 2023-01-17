import numpy as np

import pandas as pd



import plotly.offline as pyo

pyo.init_notebook_mode(connected=True)

import plotly.express as px
raw_data = pd.read_csv("/kaggle/input/forbes-celebrity-100-since-2005/forbes_celebrity_100.csv")

raw_data.head()
raw_data.info()
raw_data.describe(include='all')
data = raw_data.copy()
data['Category'].unique()
data['Category'].replace(to_replace = ['Television actresses','Actresses','Television actors'], value = 'Actors',

                         inplace = True)
data['Category'].replace({'Hip-hop impresario':'Musicians'}, inplace = True)

data['Category'].unique()
data.describe(include='all')
category_nom = data[['Year','Category']].groupby(['Year','Category'])

category_nom = data.groupby(['Year','Category']).agg({'Name':['count']})

category_nom.columns = ['Total Category Nominations']

category_nom = category_nom.reset_index()

category_nom.head()
category = category_nom.groupby(['Year'])['Total Category Nominations'].transform(max) == category_nom['Total Category Nominations']

category_analysis = category_nom[category].reset_index(drop=True)

category_analysis
fig = px.bar(category_analysis, x= 'Year', y='Total Category Nominations', color='Category',

            title='Forbes Celebrity 100: Most listed Category(2005-2019)',

            hover_name='Year',

            hover_data=['Category','Total Category Nominations'],

            template='plotly_dark')



fig.show()
nomination_analysis = data.groupby(['Name','Category'],as_index=False)['Name'].size().reset_index(name='Total Forbes Listings')

nomination_analysis.sort_values(by=['Total Forbes Listings'], inplace=True, ascending=False)

nomination_analysis = nomination_analysis[nomination_analysis['Total Forbes Listings']>13]

nomination_analysis.reset_index(inplace=True, drop=True)

nomination_analysis
fig = px.bar(nomination_analysis, x= 'Total Forbes Listings', y='Name', 

             color='Category',

             orientation='h',

             title='Forbes Celebrity 100: Most listed Celebrities(2005-2019)',

             hover_name='Name',

             hover_data=['Name','Category','Total Forbes Listings'],

             template='plotly_dark').update_yaxes(categoryorder='total ascending')



fig.show()
percent_change = data.groupby(['Year'],as_index=False)['Pay (USD millions)'].sum()

percent_change['Difference in Cumulative Pay'] = percent_change['Pay (USD millions)'].diff()

percent_change['Percent Change in Pay'] = percent_change['Pay (USD millions)'].pct_change().round(decimals=2)

percent_change.rename(columns={'Pay (USD millions)': 'Cumulative Earnings(in billions)'},inplace=True)

percent_change
fig = px.line(percent_change, x= 'Year', y='Percent Change in Pay', 

             title='Cumulative Earnings: Year Over Year Percentage Change',

             hover_name='Year',

             hover_data=['Year','Cumulative Earnings(in billions)','Difference in Cumulative Pay','Percent Change in Pay'],

             template='plotly_dark')



fig.show()
top_list = data.groupby(['Year']).apply(lambda x: x.nlargest(10,'Pay (USD millions)')).reset_index(drop=True)

top_list = top_list[top_list['Year']==2019]

column_names = ['Year','Category','Name','Pay (USD millions)']

top_list = top_list.reindex(columns=column_names)

top_list.reset_index(drop=True,inplace=True)

top_list.head(10)
fig = px.bar(top_list, x= 'Pay (USD millions)', y='Name', 

             color='Category',

             title='Top 10 Celebrities Who Made It To The Top Of The Forbes List? (2019)',

             orientation='h',

             hover_name='Name',

             hover_data=['Name','Category','Pay (USD millions)'],

             template='plotly_dark').update_yaxes(categoryorder='total ascending')



fig.show()
data2 = data.groupby(['Year','Category'])['Pay (USD millions)'].transform(max) == data['Pay (USD millions)']

data_analysis = data[data2]

data_analysis = data_analysis[data_analysis['Year']==2019]

column_names = ['Year','Category','Name','Pay (USD millions)']

data_analysis = data_analysis.reindex(columns=column_names)

data_analysis.reset_index(drop=True,inplace=True)

data_analysis.head(10)
fig = px.bar(data_analysis, x= 'Pay (USD millions)', y='Name', 

             color='Category',

             title='Forbes Celebrity 100: Who Was The Highest Earning Celebrity In Each Category? (2019)',

             orientation='h',

             hover_name='Name',

             hover_data=['Name','Category','Pay (USD millions)'],

             template='plotly_dark').update_yaxes(categoryorder='total ascending')



fig.show()
category_dist = data.groupby(['Year','Category'],as_index=False).count()

category_dist = category_dist[category_dist['Year']==2019]

category_dist = category_dist.drop(['Pay (USD millions)'],axis=1)

category_dist.rename(columns={'Name':'Category Count'},inplace=True)

category_dist.reset_index(drop=True,inplace=True)

category_dist
fig = px.pie(category_dist, values='Category Count',names='Category',

            title='Percentage Distribution of Category',)



fig.update_traces(textposition='inside',textinfo='percent+label')



fig.show()