# for basic mathematics operation 
import numpy as np
import pandas as pd
from pandas import plotting
import datetime

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import datetime as dt
import missingno as msno

# for interactive visualizations
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

#Word Cloud
from PIL import Image
import requests
from io import BytesIO
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from textblob import TextBlob

#Map
import geopandas as gpd
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import tqdm
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook

# for path
import os
print(os.listdir('../input/'))
# importing the dataset
data = pd.read_csv('../input/us-police-shootings/shootings.csv')
dat = ff.create_table(data.head())
#py.iplot(dat)
dat.update_layout(autosize=False,height=200, width = 2000)
#removing name and id column
data.drop(['id', 'name'], axis = 1, inplace = True)
data['date'] = pd.to_datetime(data.date)
data['day_sent'] = data['date'].dt.strftime('%a')
data['month_sent'] = data['date'].dt.strftime('%b')
data['year_sent'] = data['date'].dt.year
data['count'] = 1

dat = ff.create_table(data.head())
dat.update_layout(autosize=False,height=200, width = 2200)
#check na's
msno.matrix(data) #no missing values in Dataframe
years = [2015,2016,2017,2018,2019,2020]
arms = data['arms_category'].unique().tolist()
grouped_by_year_and_arms = data.groupby(['year_sent',
                                        'arms_category']).sum().reset_index()[['year_sent', 'arms_category', 'count']]
fig = make_subplots(rows=3, cols=3, shared_yaxes=True, subplot_titles=("2015", "2016", "2017", "", "", "" , "2018", "2019", "2020"))

fig.add_trace(go.Bar(x = grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2015]['arms_category'].tolist(),
                     y = grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2015]['count'].tolist(), 
                     marker=dict(color=grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2015]['count'].tolist(), coloraxis="coloraxis")), 1,1)

fig.add_trace(go.Bar(x = grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2016]['arms_category'].tolist(),
                     y = grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2016]['count'].tolist(),
                     marker=dict(color=grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2016]['count'].tolist(), coloraxis="coloraxis")), 1,2)

fig.add_trace(go.Bar(x = grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2017]['arms_category'].tolist(),
                     y = grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2017]['count'].tolist(),
                     marker=dict(color=grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2017]['count'].tolist(), coloraxis="coloraxis")), 1,3)

fig.add_trace(go.Bar(x = grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2018]['arms_category'].tolist(),
                     y = grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2018]['count'].tolist(),
                     marker=dict(color=grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2018]['count'].tolist(), coloraxis="coloraxis")), 3,1)

fig.add_trace(go.Bar(x = grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2019]['arms_category'].tolist(),
                     y = grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2019]['count'].tolist(),
                     marker=dict(color=grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2019]['count'].tolist(), coloraxis="coloraxis")), 3,2)

fig.add_trace(go.Bar(x = grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2020]['arms_category'].tolist(),
                     y = grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2020]['count'].tolist(),
                     marker=dict(color=grouped_by_year_and_arms[grouped_by_year_and_arms['year_sent'] == 2020]['count'].tolist(), coloraxis="coloraxis")), 3,3)

fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False, title_text='Arms Used By Victims (2015-2020):')
fig.show()
labels = ['Male', 'Female']
size = data['gender'].value_counts()
colors = ['seagreen', 'crimson']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (6, 6)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Gender', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()
#AGE
data['age'].iplot(kind = "hist",barmode= 'overlay',
                  xTitle = "Age", title='Age Distribution', bins = 50, colors = 'crimson')
locator = Nominatim(user_agent="myGeocoder", timeout=200)
geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
temp_data = data[['city','count']]
temp_data = temp_data.groupby('city').agg({'count': 'sum'})
geocode
temp_data.reset_index(inplace = True)
tqdm.pandas()
temp_data['location'] = temp_data['city'].progress_apply(geocode)
temp_data['point'] = temp_data['location'].apply(lambda loc: tuple(loc.point) if loc else None)
temp_data[['latitude', 'longitude', 'altitude']] = pd.DataFrame(temp_data['point'].tolist(), index=temp_data.index)
#graph total
data_graph = go.Scattergeo(lon = temp_data['longitude'], lat = temp_data['latitude'],
                           text = temp_data[['city', 'count']], mode = 'markers', 
                           marker = dict(symbol = 'star',size = 5,colorscale = 'Blackbody'),
                           marker_color = temp_data['count'])

layout = dict(title = 'Plot of suffered cities (Worldwide)')

choromap = go.Figure(data = [data_graph],layout = layout)
iplot(choromap)
#Graph (USA)
temp_data['text'] = temp_data['city'] + '<br>Shootings: </br>' + (temp_data['count']).astype(str)

limits = [(0,10),(10,20),(20,40),(40,70),(70,100)]
colors = ["royalblue", "lightgrey", "orange", "seagreen", "crimson"]
cities = []
scale = 30

fig = go.Figure()

for i in range(len(limits)):
    lim = limits[i]
    df_sub = temp_data[lim[0]:lim[1]]
    fig.add_trace(go.Scattergeo(
        locationmode = 'USA-states',
        lon = df_sub['longitude'],
        lat = df_sub['latitude'],
        text = temp_data['text'],
        marker = dict(
            size = df_sub['count'] * scale,
            color = colors[i],
            line_color='rgb(40,40,40)',
            line_width=0.5,
            sizemode = 'area'
        ),
        name = '{0} - {1}'.format(lim[0],lim[1])))

fig.update_layout(
        title_text = '2015-2020 USA Shootings. <br>(Click legend to toggle traces)</br>',
        showlegend = True,
        geo = dict(
            scope = 'usa',
            landcolor = 'rgb(51, 48, 48)',
        )
    )

fig.show()
race_data=data['race'].value_counts().to_frame().reset_index().rename(columns={'index':'race','race':'count'})
fig = go.Figure(go.Funnel(y = race_data['race'].tolist(),
                          x = race_data['count'].tolist(), 
                          marker = {"color": ['deepskyblue', 'MediumPurple',
                                              'teal', 'grey', 'lightsalmon',
                                              'midnightblue'], 
                                    "line": {"color": ["wheat", "blue", "wheat", "blue", "wheat"], 
                                    "width": [0, 1, 5, 0, 4]}},
                          textfont = {"family": "Old Standard TT, serif",
                                      "size": 13, "color": "black"},
                          opacity = 0.65))
fig.update_layout(title = "All Race People Killed In Every Region", title_x = 0.5)
fig.show()
#Race distribution by Age
data.pivot(columns='race', values='age').iplot(kind='box', yTitle='age', title='Race Distribution by Age')
#manner of death by police
labels = ['Shot', 'Shot and Tasered']
size = data['manner_of_death'].value_counts()
colors = ['salmon', 'MediumPurple']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (6, 6)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Manner Of Death', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()
#total deaths
total_shoot = data[data['count']==1].shape[0]
fig = go.Figure(go.Indicator(mode = "number",
                             value = total_shoot,
                             title = {"text": "Total No. Of Shootings.",
                                      "font" : {'color': 'Black', 'size': 50, 'family': 'Raleway'}},
                             number = {'font': {'color': 'Black', 'size': 100, 'family': 'Raleway'}},
                             domain = {'x': [0,1], 'y': [0,1]}))
fig.show()
data2015 = data[data['year_sent'] == 2015].shape[0]
data2016 = data[data['year_sent'] == 2016].shape[0]
data2017 = data[data['year_sent'] == 2017].shape[0]
data2018 = data[data['year_sent'] == 2018].shape[0]
data2019 = data[data['year_sent'] == 2019].shape[0]
data2020 = data[data['year_sent'] == 2020].shape[0]

#int(data.loc[data['year_sent'] == 2015]['count'])

fig = go.Figure()
fig.add_trace(go.Indicator(mode = "number",
                             value = data2015,
                             title = {"text": "2015",
                                      "font" : {'color': 'rgb(58, 171, 163)', 'size': 25, 'family': 'Raleway'}},
                             number = {'font': {'color': 'rgb(58, 171, 163)', 'size': 25, 'family': 'Raleway'}},
                             domain = {'row': 0, 'column': 0}))

fig.add_trace(go.Indicator(mode = "number",
                             value = data2016,
                             title = {"text": "2016",
                                      "font" : {'color': 'rgb(41, 79, 150)', 'size': 33, 'family': 'Raleway'}},
                             number = {'font': {'color': 'rgb(41, 79, 150)', 'size': 33, 'family': 'Raleway'}},
                             domain = {'row': 0, 'column': 1}))

fig.add_trace(go.Indicator(mode = "number",
                             value = data2017,
                             title = {"text": "2017",
                                      "font" : {'color': 'rgb(52, 150, 41)', 'size': 35, 'family': 'Raleway'}},
                             number = {'font': {'color': 'rgb(52, 150, 41)', 'size': 35, 'family': 'Raleway'}},
                             domain = {'row': 0, 'column': 2}))

fig.add_trace(go.Indicator(mode = "number",
                             value = data2018,
                             title = {"text": "2018",
                                      "font" : {'color': 'rgb(246, 255, 8)', 'size': 47, 'family': 'Raleway'}},
                             number = {'font': {'color': 'rgb(246, 255, 8)', 'size': 50, 'family': 'Raleway'}},
                             domain = {'row': 0, 'column': 3}))

fig.add_trace(go.Indicator(mode = "number",
                             value = data2019,
                             title = {"text": "2019",
                                      "font" : {'color': 'rgb(232, 139, 0)', 'size': 63, 'family': 'Raleway'}},
                             number = {'font': {'color': 'rgb(232, 139, 0)', 'size': 63, 'family': 'Raleway'}},
                             domain = {'row': 0, 'column': 4}))

fig.add_trace(go.Indicator(mode = "number",
                             value = data2020,
                             title = {"text": "2020",
                                      "font" : {'color': 'rgb(0, 0, 0)', 'size': 66, 'family': 'Raleway'}},
                             number = {'font': {'color': 'rgb(0, 0, 0)', 'size': 70, 'family': 'Raleway'}},
                             domain = {'row': 0, 'column': 6}))

fig.update_layout(grid = {'rows': 1, 'columns': 7, 'pattern': 'independent'})
fig.show()
groupedby_time = data.groupby('date').sum().reset_index()

fig = px.line(groupedby_time, x = 'date', y = 'count', title='Timeline For The Shootings')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()
#year and months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
years = [2015, 2016, 2017, 2018, 2019, 2020]

grouped_by_year_and_day = data.groupby(['year_sent',
                                        'month_sent']).sum().reset_index()[['year_sent', 'month_sent', 'count']]

pt = grouped_by_year_and_day.pivot_table(index = 'year_sent',
                                         columns = 'month_sent',
                                         values = 'count').reindex(index = years, columns = months)

pt.iplot(kind='heatmap',colorscale="RdPu", title="Heatmap of Shootings Count As Per Month And Year")

data.rename(columns = {'year_sent': 'Year'}, inplace = True)
fig = px.parallel_categories(data,dimensions = ['gender', 'race', 'signs_of_mental_illness', 'flee'],
                             color = 'Year', color_continuous_scale=px.colors.sequential.Inferno,
                             labels = {'gender': 'Gender', 'race': 'Humankind',
                                       'signs_of_mental_illness': 'Mental Illness ?',
                                       'flee': 'Flee ?'})
fig.show()