# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Importing basic packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt #visualize into notebook
%matplotlib inline  
pd.__version__
#importing aditional packages for visualizing
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
import time
import networkx as nx
from plotly.graph_objs import *
# Any results you write to the current directory are saved as output.
print(os.listdir("../input"))
#Load the data
COLLISIONS_CSV = '../input/nypd-motor-vehicle-collisions.csv'
DATADICTIONARY = '../input/Collision_DataDictionary.xlsx'
METADATA_JSON = '../input/socrata_metadata.json'

collisionsDF = pd.read_csv(COLLISIONS_CSV)

collisionsDF.head(2)
# Assuming that dataframe has DATE and Time columns sorted (meaning row 0 is latest date and last row represents the earliest date)
# It can be suppossed based on the tail() and head() results
oldest_date = collisionsDF.iloc[-1].DATE #tail
newest_date = collisionsDF.iloc[0].DATE #head

newest_date
collisionsDF['DATE'].dtype # IS NECESARY CONVERT DTYPE TO DATETIME
#Extract DateTime features
collisionsDF['collisions_datetime'] = pd.to_datetime(collisionsDF['DATE'], infer_datetime_format = True)   #infer_datetime_format=True) 
collisionsDF['collisions_time'] = pd.to_datetime(collisionsDF['TIME'], infer_datetime_format = True) 

#collisionsDF.loc[:, 'collisions_datetime'] = collisionsDF['collisions_datetime'].dt.date

collisionsDF.loc[:, 'collisions_weekday'] = collisionsDF['collisions_datetime'].dt.weekday
collisionsDF.loc[:, 'collisions_weekofyear'] = collisionsDF['collisions_datetime'].dt.weekofyear
collisionsDF.loc[:, 'collisions_hour'] = collisionsDF['collisions_time'].dt.hour
collisionsDF.loc[:, 'collisions_minute'] = collisionsDF['collisions_time'].dt.minute
collisionsDF.loc[:, 'collisions_dt'] = (collisionsDF['collisions_datetime'] - collisionsDF['collisions_datetime'].min()).dt.total_seconds()

#collisionsDF['collisions_time'].dtype
collisionsDF.head()

#Data Preparation for the weekly analysis
#get count of collisions every hour on every weekday.
sunday = collisionsDF[collisionsDF['collisions_weekday'] == 6]
df_sundayhourlytripcount = sunday.groupby('collisions_hour').count()
monday = collisionsDF[collisionsDF['collisions_weekday'] == 0]
df_mondayhourlytripcount = monday.groupby('collisions_hour').count()
tuesday = collisionsDF[collisionsDF['collisions_weekday'] == 1]
df_tuesdayhourlytripcount = tuesday.groupby('collisions_hour').count()
wednesday = collisionsDF[collisionsDF['collisions_weekday'] == 2]
df_wednesdayhourlytripcount = wednesday.groupby('collisions_hour').count()
thursday =  collisionsDF[ collisionsDF['collisions_weekday'] == 3]
df_thursdayhourlytripcount = thursday.groupby('collisions_hour').count()
friday = collisionsDF[collisionsDF['collisions_weekday'] == 4]
df_fridayhourlytripcount = friday.groupby('collisions_hour').count()
saturday = collisionsDF[collisionsDF['collisions_weekday'] == 5]
df_saturdayhourlytripcount = saturday.groupby('collisions_hour').count()
collisions_hr_x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
sun_crashcounty = df_sundayhourlytripcount['TIME']
mon_crashcounty = df_mondayhourlytripcount['TIME']
tues_crashcounty = df_tuesdayhourlytripcount['TIME']
wed_crashcounty = df_wednesdayhourlytripcount['TIME']
thurs_crashcounty = df_thursdayhourlytripcount['TIME']
fri_crashcounty = df_fridayhourlytripcount['TIME']
sat_crashcounty = df_saturdayhourlytripcount['TIME']

df_sundayhourlytripcount.tail(1)
sun_crashcounty.tail(1)

# Create traces
trace1 = go.Scatter(x = collisions_hr_x, y = sun_crashcounty,  mode = "lines+markers", name = 'Sunday')
trace2 = go.Scatter(x = collisions_hr_x, y = mon_crashcounty, name = 'Monday')
trace3 = go.Scatter( x = collisions_hr_x, y = tues_crashcounty, name = 'Tuesday')
trace4 = go.Scatter( x = collisions_hr_x, y = wed_crashcounty, name = 'Wednesday')
trace5 = go.Scatter( x = collisions_hr_x, y = thurs_crashcounty,  name = 'Thursday')
trace6 = go.Scatter( x = collisions_hr_x, y = fri_crashcounty,   name = 'Friday')
trace7 = go.Scatter(x =collisions_hr_x, y = sat_crashcounty,  mode = "lines+markers", name = 'Saturday')
layout = dict(title = 'Weekly Collisions by Hour', 
              xaxis= dict(title= 'Hour of the day',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Collisions quantity',ticklen= 5,zeroline= False)
             )
linedata = [trace1, trace2, trace3, trace4, trace5, trace6,trace7]
fig = dict(data=linedata, layout=layout)
py.iplot(fig, filename='timeline-lineplot')
#Preparing data for the graph
W_0 = collisionsDF[collisionsDF['collisions_weekday'] == 0].groupby('collisions_hour').count()
W_1 = collisionsDF[collisionsDF['collisions_weekday'] == 1].groupby('collisions_hour').count()
W_2 = collisionsDF[collisionsDF['collisions_weekday'] == 2].groupby('collisions_hour').count()
W_3 = collisionsDF[collisionsDF['collisions_weekday'] == 3].groupby('collisions_hour').count()
W_4 = collisionsDF[collisionsDF['collisions_weekday'] == 4].groupby('collisions_hour').count()
W_5 = collisionsDF[collisionsDF['collisions_weekday'] == 5].groupby('collisions_hour').count()
W_6 = collisionsDF[collisionsDF['collisions_weekday'] == 6].groupby('collisions_hour').count()

#W_0.head()
trace = go.Heatmap(z=[W_0.TIME,W_1.TIME,W_2.TIME,W_3.TIME,W_4.TIME,W_5.TIME,W_6.TIME],
                    y=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'],
                   x=['Midnight','1am','2am','3am','4am','5am','6am','7am','8am','9am','10am','11am','Noon','1pm','2pm',
                     '3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm'],
                  colorscale='Reds',xgap = 10,ygap = 10,)

layout = dict(title = 'Collisions per week per hour')
dataheat=[trace]
fig = dict(data = dataheat, layout=layout)
py.iplot(fig, filename='labelled-heatmap')
#Takeaway: Friday seems to have rush hours around 5pm. This may be impacting durations trip around that time. Could be interesing to know what
#is the street name with most collisions in that hour.
collisionsDF.columns
W_4['UNIQUE KEY'][W_4['UNIQUE KEY'].values==max(W_4['UNIQUE KEY'])]
collisionsDF.groupby('ON STREET NAME').count().sort_values(by='UNIQUE KEY', ascending=False).head(10)['UNIQUE KEY']
collisionsDF[(collisionsDF['collisions_weekday'] == 4) & (collisionsDF['collisions_hour'] == 17)].groupby('ON STREET NAME').count().sort_values(by='UNIQUE KEY', ascending=False).head(10)['UNIQUE KEY']
TOP5 = collisionsDF.groupby('ON STREET NAME').count().sort_values(by='UNIQUE KEY', ascending=False).head(5)['UNIQUE KEY']
TOP5.keys()
#Lets look at demand per hour.
bar1data = collisionsDF[collisionsDF['ON STREET NAME']==TOP5.keys()[0]].groupby(['collisions_hour']).count().reset_index()
bar2data = collisionsDF[collisionsDF['ON STREET NAME']==TOP5.keys()[2]].groupby(['collisions_hour']).count().reset_index()
bar3data = collisionsDF[collisionsDF['ON STREET NAME']==TOP5.keys()[3]].groupby(['collisions_hour']).count().reset_index()
bar4data = collisionsDF[collisionsDF['ON STREET NAME']==TOP5.keys()[4]].groupby(['collisions_hour']).count().reset_index()
trace0 = go.Bar(
    x=['Midnight','1am','2am','3am','4am','5am','6am','7am','8am','9am','10am','11am','Noon','1pm','2pm',
                     '3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm'],
    y= bar1data['UNIQUE KEY'],
    name='BROADWAY',
    marker=dict(color='rgb(49,130,189)' ))
trace1 = go.Bar(
    x=['Midnight','1am','2am','3am','4am','5am','6am','7am','8am','9am','10am','11am','Noon','1pm','2pm',
                     '3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm'],
    y= bar2data['UNIQUE KEY'],
    name='ATLANTIC AVENUE',
    marker=dict(color='rgb(120,70,100)',))

trace2 = go.Bar(
    x=['Midnight','1am','2am','3am','4am','5am','6am','7am','8am','9am','10am','11am','Noon','1pm','2pm',
                     '3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm'],
    y= bar3data['UNIQUE KEY'],
    name='3 AVENUE',
    marker=dict(color='rgb(130,200,110)',))
trace3 = go.Bar(
    x=['Midnight','1am','2am','3am','4am','5am','6am','7am','8am','9am','10am','11am','Noon','1pm','2pm',
                     '3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm'],
    y= bar4data['UNIQUE KEY'],
    name='NORTHERN BOULEVARD',
    marker=dict(color='rgb(102,0,0)',))
datax = [trace0, trace1, trace2, trace3]
layout = go.Layout(xaxis=dict(tickangle=-45),barmode='group')

fig = go.Figure(data=datax, layout=layout)
py.iplot(fig)