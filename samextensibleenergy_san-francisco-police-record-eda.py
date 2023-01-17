import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime as dt
from sklearn.metrics import mean_squared_error
from math import sqrt
from mpl_toolkits.basemap import Basemap

# Import BOKEH libraries 
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
output_notebook()

# Import PLOTLY libraries 
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
calls = pd.read_csv('../input/police-department-calls-for-service.csv')
print ("Calls Dataset: Rows, Columns: ", calls.shape)

notice = pd.read_csv('../input/change-notice-police-department-incidents.csv')
print ("Notice Dataset: Rows, Columns: ", notice.shape)
calls.head()
# creating additional columns for splitting the date time

calls['call_date'] = pd.to_datetime(calls['Call Date Time'])
calls['call_year'] = calls['call_date'].dt.year
calls['call_month'] = calls['call_date'].dt.month
calls['call_monthday'] = calls['call_date'].dt.day
calls['call_weekday'] = calls['call_date'].dt.weekday

#dropping the unwanted columns 
calls = calls.drop(['Call Date', 'Call Date Time'], axis=1)
calls.head()
from wordcloud import WordCloud, STOPWORDS

# mask = np.array(Image.open('../input/word-cloud-masks/gun.png'))
txt = " ".join(calls['Original Crime Type Name'].dropna())
wc = WordCloud(max_words=1200, stopwords=STOPWORDS, colormap='copper', background_color='White').generate(txt)
plt.figure(figsize=(18,10))
plt.imshow(wc)
plt.axis('off')
plt.title('');
df1 = calls
x = df1['call_month']
hist, edges = np.histogram(x, bins=12)

p1 = figure(title="Histogram of monthly calls",tools="save",
            background_fill_color="#E8DDCB")
p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
        fill_color="#036564", line_color="#033649")
p1.legend.location = "center_right"
p1.legend.background_fill_color = "darkgrey"
p1.xaxis.axis_label = 'Months'
p1.yaxis.axis_label = 'Frequency'



x = df1['call_monthday']
hist, edges = np.histogram(x, bins=31)

p2 = figure(title="Histogram of month_Day calls",tools="save",
            background_fill_color="#E8DDCB")
p2.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
        fill_color="#036564", line_color="#033649")
p2.legend.location = "center_right"
p2.legend.background_fill_color = "darkgrey"
p2.xaxis.axis_label = 'Month_Day'
p2.yaxis.axis_label = 'Frequency'




x = df1['call_weekday']
hist, edges = np.histogram(x, bins=7)
p3 = figure(title="Histogram of Week_day calls",tools="save",
            background_fill_color="#E8DDCB")
p3.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
        fill_color="#036564", line_color="#033649")
p3.legend.location = "center_right"
p3.legend.background_fill_color = "darkgrey"
p3.xaxis.axis_label = 'Week_Day'
p3.yaxis.axis_label = 'Frequency'

output_file('histogram.html', title=" ")

show(gridplot(p1,p2,p3, ncols=3, plot_width=400, plot_height=400, toolbar_location=None))
grp_1 = pd.DataFrame(calls.groupby([ 'Call Time']).count()).reset_index() 
tempdf = grp_1
trace1 = go.Bar(
    x=tempdf['Call Time'],
    y=tempdf['Crime Id'],
    name='Location Types',
    orientation = 'v',
    marker=dict(color='red'),
    opacity=0.7
)

data = [trace1]
layout = go.Layout(
    height=400,
    margin=dict(b=150),
    barmode='group',
    legend=dict(dict(x=-.1, y=1.2)),
    title = 'Number of Phone Calls in different hours of the day',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')
notice.head()
# creating additional columns for splitting the date time

notice['Date'] = pd.to_datetime(notice['Date'])
notice['Year'] = notice['Date'].dt.year
notice['Month'] = notice['Date'].dt.month
notice['Monthday'] = notice['Date'].dt.day
notice['Day_of_Week'] = notice['Date'].dt.weekday
notice.head()
grp_0 = pd.DataFrame(notice.groupby([ 'PdDistrict']).count()).reset_index() 
grp_0.head(n=10)
df_test = notice
df = df_test.loc[df_test['PdDistrict'] == 'TENDERLOIN']
# X = df["X"].mean()
# Y = df["Y"].mean()
print ("X: ", df["X"].mean())
print ("Y: ", df["Y"].mean())
d = {'District': ['BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN',
                 'PARK', 'RICHMOND','SOUTHERN', 'TARAVAL', 'TENDERLOIN'], 
     'count': [221000, 226255, 194180, 300076, 272713,125479, 116818, 399785, 166971, 191746],
     'X': [-122.39348575465807,-122.40955408209737, -122.42880918975492,-122.41947208237181, 
           -122.42651941834849,-122.44538250133472, -122.47073603937096,-122.40514832144576,
           -122.4775493762299, -122.4121701007921
          ],
     'Y': [37.741045584280045, 37.79784019858191, 37.728485850396815, 37.76035586908415,
           37.79219117621023, 37.77198029829212, 37.7867277022189, 37.78283485237802,
           37.73971978721108, 37.7930189259414
          ],
    'pos': [(-122.39348575465807,37.741045584280045),(-122.40955408209737,37.79784019858191),
           ( -122.42880918975492,37.728485850396815),(-122.41947208237181,37.76035586908415),
           (-122.42651941834849,37.79219117621023),(-122.44538250133472,37.77198029829212),
           (-122.47073603937096,37.7867277022189),(-122.40514832144576,37.78283485237802),
           (-122.4775493762299,37.73971978721108),(-122.4121701007921,37.7930189259414)]}
area = pd.DataFrame(data=d)
area
# Basemap Documentation: https://matplotlib.org/basemap/api/basemap_api.html
    
#  Source Code: https://github.com/jamalmoir/notebook_playground/blob/master/uk_2015_houseprice.ipynb

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

fig, ax = plt.subplots(figsize=(10,20))

m = Basemap(resolution='l', # c, l, i, h, f or None
            projection='aeqd',
            lat_0=37.75, lon_0=-122.42,
            llcrnrlon=-122.50, llcrnrlat= 37.67, urcrnrlon= -122.30, urcrnrlat=37.90)  

m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.drawcoastlines()
    
def plot_area(pos):
    count = area.loc[area.pos == pos]['count']
    x, y = m(pos[0], pos[1])
    size = (count/1000000) ** 2 + 3
    m.plot(x, y, 'o', markersize=size, color='#444444', alpha=0.8)
    
area['pos'].apply(plot_area)

m
grp_2 = pd.DataFrame(notice.groupby([ 'Category']).count()).reset_index() 
tempdf = grp_2.sort_values('IncidntNum', ascending = False)
trace1 = go.Bar(
    x=tempdf['Category'],
    y=tempdf['IncidntNum'],
    name='Location Types',
    orientation = 'v',
    marker=dict(        
        color=tempdf.IncidntNum,
        colorscale = 'Jet',
        reversescale = True),
    opacity=0.7
)

data = [trace1]
layout = go.Layout(
    height=400,
    margin=dict(b=150),
    barmode='group',
    legend=dict(dict(x=-.1, y=1.2)),
    title = 'Number of Crimes in California 2002 - 2018 (till July)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')
df2 = notice
x = df2['Month']
hist, edges = np.histogram(x, bins=12)

p1 = figure(title="Histogram of MOnthly Incidents",tools="save",
            background_fill_color="#E8DDCB")
p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
        fill_color="#036564", line_color="#033649")
p1.legend.location = "center_right"
p1.legend.background_fill_color = "darkgrey"
p1.xaxis.axis_label = 'Months'
p1.yaxis.axis_label = 'Frequency'


x = df2['Monthday']
hist, edges = np.histogram(x, bins=31)

p2 = figure(title="Histogram of Month_Date Incidents",tools="save",
            background_fill_color="#E8DDCB")
p2.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
        fill_color="#036564", line_color="#033649")
p2.legend.location = "center_right"
p2.legend.background_fill_color = "darkgrey"
p2.xaxis.axis_label = 'Month_Day'
p2.yaxis.axis_label = 'Frequency'


x = df2['Day_of_Week']
hist, edges = np.histogram(x, bins=19)
p3 = figure(title="Histogram of WeekDay Incidents",tools="save",
            background_fill_color="#E8DDCB")
p3.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
        fill_color="#036564", line_color="#033649")
p3.legend.location = "center_right"
p3.legend.background_fill_color = "darkgrey"
p3.xaxis.axis_label = 'Week_Day'
p3.yaxis.axis_label = 'Frequency'


x = df2['Year']
hist, edges = np.histogram(x, bins=16)
p4 = figure(title="Histogram of Yearly Incidents",tools="save",
            background_fill_color="#E8DDCB")
p4.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
        fill_color="#036564", line_color="#033649")
            
p4.legend.location = "center_right"
p4.legend.background_fill_color = "darkgrey"
p4.xaxis.axis_label = 'Years'
p4.yaxis.axis_label = 'Frequency'

show(gridplot(p1,p2,p3,p4, ncols=2, plot_width=400, plot_height=400, toolbar_location=None))






