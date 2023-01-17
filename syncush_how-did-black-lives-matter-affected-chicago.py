import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.graph_objs import *
import seaborn as sns
init_notebook_mode()
print(os.listdir("../input"))
data_summary = pd.read_csv('../input/copa-cases-summary.csv')
data_officer = pd.read_csv('../input/copa-cases-by-involved-officer.csv')
data_complainant = pd.read_csv('../input/copa-cases-by-complainant-or-subject.csv')
data_summary.head(5)
data_officer.head(5)
data_complainant.head(5)
def time_of_day_to_desc(x):
    if x < 7:
        return "Early Morning"
    if x < 12:
        return "Morning"
    if x < 18:
        return "Afternoon"
    if x < 22:
        return "Evening"
    return "Night"

def weekend_or_not(x):
    # Friday , Saturday, Sunday 
    if x is 6 or x is 7 or x is 1:
        return "Weekend"
    return "Weekday"

def is_black_person_involved(x):
    if  not pd.isna(x) and 'African American / Black' in x:
        return "Yes"
    return "No"
data_summary['COMPLAINT_DATE'] = pd.to_datetime(data_summary['COMPLAINT_DATE'], infer_datetime_format=True)
data_complainant['COMPLAINT_DATE'] = pd.to_datetime(data_complainant['COMPLAINT_DATE'], infer_datetime_format=True)
data_complainant['RACE_OF_COMPLAINANT'] = data_complainant['RACE_OF_COMPLAINANT'].apply(lambda x: x if x != 'African American / Black' else 'Black')
data_summary['YEAR_OF_COMPLAINT'] = data_summary['COMPLAINT_DATE'].apply(lambda x:x.year)
data_complainant['YEAR_OF_COMPLAINT'] = data_complainant['COMPLAINT_DATE'].apply(lambda x:x.year)
data_summary['HOUR_DESC'] = data_summary['COMPLAINT_HOUR'].apply(time_of_day_to_desc)
data_complainant['HOUR_DESC'] = data_complainant['COMPLAINT_HOUR'].apply(time_of_day_to_desc)
data_complainant['IS_WEEKEND'] = data_complainant['COMPLAINT_DAY'].apply(weekend_or_not)
data_summary['IS_WEEKEND'] = data_summary['COMPLAINT_DAY'].apply(weekend_or_not)
data_summary['IS_BLACK_INV'] = data_summary['RACE_OF_COMPLAINANTS'].apply(is_black_person_involved)
data_summary.head(5)
print("Python Datetime Day: {} \nThe Number of the Day in the Data: {}".format(data_summary.iloc[0]["COMPLAINT_DATE"].strftime("%A"), data_summary.iloc[0]["COMPLAINT_DAY"]))
data_summary['YEAR_OF_COMPLAINT'].value_counts().plot('barh')
temp = data_summary['CURRENT_CATEGORY'].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "name": "Complaints Type",
      "hoverinfo":"label+percent+name",
      "hole": .1,
      "type": "pie"
    }],
  "layout": {
        "title":"Complaints Type"
    }
}
iplot(fig, filename='pie')
temp = data_summary['POLICE_SHOOTING'].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "name": "Is POLICE_SHOOTING?",
      "hoverinfo":"label+percent+name",
      "hole": .1,
      "type": "pie"
    }],
  "layout": {
        "title":"Complaints - POLICE_SHOOTING"
    }
}
iplot(fig, filename='pie')
temp = data_summary['CURRENT_STATUS'].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "name": "CURRENT_STATUS",
      "hoverinfo":"label+percent+name",
      "hole": .1,
      "type": "pie"
    }],
  "layout": {
        "title":"Complaints - CURRENT_STATUS"
    }
}
iplot(fig, filename='pie')
fig, ax = plt.subplots(nrows=4, ncols=3)
plt.subplots_adjust(left=0, right=4.2, top=5, bottom=0)
year = 2007
column = 0
for row in ax:
    for col in row:
        col.set_title(str(year))
        tempush = data_summary[data_summary['YEAR_OF_COMPLAINT'] == year]
        prop_df = (tempush['CURRENT_STATUS']
                   .value_counts(normalize=True)
                   .rename('Percentage')
                   .reset_index())
        g = sns.barplot(y="index", x="Percentage", data=prop_df, ax=col, orient='h')
        g.set_ylabel('CURRENT_STATUS')
        year += 1
fig, ax = plt.subplots(nrows=4, ncols=3)
plt.subplots_adjust(left=0, right=4.2, top=5, bottom=0)
year = 2007
column = 0
for row in ax:
    for col in row:
        col.set_title(str(year))
        tempush = data_summary[data_summary['YEAR_OF_COMPLAINT'] == year]
        prop_df = (tempush['POLICE_SHOOTING']
                   .value_counts(normalize=True)
                   .rename('Percentage')
                   .reset_index())
        g = sns.barplot(y="index", x="Percentage", data=prop_df, ax=col, orient='h')
        g.set_ylabel('POLICE_SHOOTING')
        year += 1
fig, ax = plt.subplots(nrows=4, ncols=3)
plt.subplots_adjust(left=0, right=4.2, top=5, bottom=0)
year = 2007
column = 0
for row in ax:
    for col in row:
        col.set_title(str(year))
        tempush = data_summary[data_summary['YEAR_OF_COMPLAINT'] == year]
        prop_df = (tempush['CURRENT_CATEGORY']
                   .value_counts(normalize=True)
                   .rename('Percentage')
                   .reset_index())
        g = sns.barplot(y="index", x="Percentage", data=prop_df, ax=col, orient='h')
        g.set_ylabel('CURRENT_CATEGORY')
        year += 1
fig, ax = plt.subplots(nrows=12, ncols=3)
plt.subplots_adjust(left=0, right=2.2, top=12, bottom=0)
i = 0
for row in ax:
    for col, (colum, ordere) in zip(row, [('HOUR_DESC', ["Early Morning", "Morning", "Afternoon", "Evening", "Night"]),('COMPLAINT_DAY',list(range(1, 7 + 1))), ('COMPLAINT_MONTH', list(range(1, 12 + 1)))]):
        tempush = data_summary[data_summary['YEAR_OF_COMPLAINT'] == 2007 + i]
        col.set_title(str(2007 + i))
        g = sns.countplot(ax=col, data=tempush, x=colum, palette='Set1', order=ordere)
    i += 1    
order = ['20-29', '30-39', '0-19', '40-49', '50-59', '60-69', '70+', 'Unknown']
fig, ax = plt.subplots(nrows=4, ncols=3)
plt.subplots_adjust(left=0, right=4.2, top=5, bottom=0)
year = 2007
column = 0
for row in ax:
    for col in row:
        col.set_title(str(year))
        tempush = data_complainant[data_complainant['YEAR_OF_COMPLAINT'] == year]
        prop_df = (tempush['AGE_OF_COMPLAINANT']
                   .value_counts(normalize=True)
                   .rename('Percentage')
                   .reset_index())
        g = sns.barplot(y="index", x="Percentage", data=prop_df, ax=col, orient='h', order=order)
        g.set_ylabel('AGE_OF_COMPLAINANTS')
        year += 1
sns.catplot(data=data_complainant, x='AGE_OF_COMPLAINANT', kind='count', hue='HOUR_DESC', col='IS_WEEKEND', order=order)
gender = data_complainant.groupby('LOG_NO').agg({'COMPLAINT_DATE': 'last', 'COMPLAINT_HOUR' : 'count'}).reset_index()
order = ['Male', 'Female']
fig, ax = plt.subplots(nrows=4, ncols=3)
plt.subplots_adjust(left=0, right=4.2, top=5, bottom=0)
year = 2007
column = 0
for row in ax:
    for col in row:
        col.set_title(str(year))
        tempush = data_complainant[data_complainant['YEAR_OF_COMPLAINT'] == year]
        prop_df = (tempush['SEX_OF_COMPLAINANT']
                   .value_counts(normalize=True)
                   .rename('Percentage')
                   .reset_index())
        g = sns.barplot(y="index", x="Percentage", data=prop_df, ax=col, orient='h', order=order)
        g.set_ylabel('SEX_OF_COMPLAINANT')
        year += 1
sns.catplot(data=data_complainant, x='SEX_OF_COMPLAINANT', kind='count', hue='HOUR_DESC', col='IS_WEEKEND', order=order)
sns.catplot(data=data_complainant, x='SEX_OF_COMPLAINANT', kind='count', hue='RACE_OF_COMPLAINANT', col='HOUR_DESC', order=['Male', 'Female'], col_order=["Early Morning", "Morning", "Afternoon", "Evening", "Night"], hue_order=['Black', 'Hispanic', 'White', 'Unknown', 'Asian or Pacific Islander', 'American Indian or Alaskan Native'])
temp = data_summary.set_index('COMPLAINT_DATE').groupby(pd.TimeGrouper('M')).count().dropna().reset_index()
trace_high = go.Scatter(
                x=temp['COMPLAINT_DATE'],
                y=temp['LOG_NO'],
                name = "Crimes Count",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

data = [trace_high]

layout = dict(
    title='Crimes Count Over Time',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)
fig = dict(data=data, layout=layout)
iplot(fig, filename = "Manually Set Range")
scatters = []
colors_list = ['#2da6fe', '#B4F0AE', '#c47943', '#2fa1d6', '#2fa1d6', '#5a59cc', '#901e86']
for i, race in enumerate(['Black', 'Hispanic', 'White', 'Asian or Pacific Islander', 'American Indian or Alaskan Native', 'Unknown']):
        raced_df = data_complainant[data_complainant['RACE_OF_COMPLAINANT'] == race].groupby('LOG_NO').agg({'COMPLAINT_DATE': 'last', 'COMPLAINT_HOUR' : 'count'}).reset_index()
        temp = raced_df.set_index('COMPLAINT_DATE').groupby(pd.TimeGrouper('Q')).count().dropna().reset_index()
        trace_high = go.Scatter(
                        x=temp['COMPLAINT_DATE'],
                        y=temp['COMPLAINT_HOUR'],
                        name = race,
                        line = dict(color = colors_list[i]),
                        opacity = 0.8)
        scatters.append(trace_high)

data = scatters
layout = dict(
    title='Crimes Count Over Time (Quarters)',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    ),
)
ys = [250, 225, 200, 175, 150, 125]
trace0 = go.Scatter(
    x=['2013-07-01', '2014-07-30', '2014-08-01', '2015-01-30', '2016-07-01'],
    y=ys,
    name='BLM Events',
    text=['BLM Movement Founded',
          '#icantbreathe is trending',
          'Furgeson',
          'Baltimore Riots',
          'Dallas Riots'],
    mode='text',
)
data.append(trace0)
founded = {'type':'line', 'x0':'2013-07-13', 'x1':'2013-07-13', 'y0':0, 'y1':250, 'line':dict(color='rgb(0, 0, 0)', width=1)}
icantbreath = {'type':'line', 'x0':'2014-07-17', 'x1':'2014-07-17', 'y0':0, 'y1':250, 'line':dict(color='rgb(0, 0, 0)', width=1)}
furgeson = {'type':'line', 'x0':'2014-08-09', 'x1':'2014-08-09', 'y0':0, 'y1':250, 'line':dict(color='rgb(0, 0, 0)', width=1)}
dallas = {'type':'line', 'x0':'2016-07-07', 'x1':'2016-07-07', 'y0':0, 'y1':250, 'line':dict(color='rgb(0, 0, 0)', width=1)}
baltimore = {'type':'line', 'x0':'2015-01-12', 'x1':'2015-01-12', 'y0':0, 'y1':250, 'line':dict(color='rgb(0, 0, 0)', width=1)}
layout["shapes"] = [founded, icantbreath, furgeson, baltimore, dallas]
fig = dict(data=data, layout=layout)
iplot(fig, filename = "Manually Set Range")
scatters = []
colors_list = ['#2da6fe', '#B4F0AE', '#c47943', '#2fa1d6', '#2fa1d6', '#5a59cc', '#901e86', '#4528C9', '#D0F2AE']
for i, age in enumerate(['20-29', '30-39', '0-19', '40-49', '50-59', '60-69', '70+', 'Unknown']):
        raced_df = data_complainant[data_complainant['RACE_OF_COMPLAINANT'] == 'Black']
        raced_df = raced_df[raced_df['AGE_OF_COMPLAINANT'] == age].groupby('LOG_NO').agg({'COMPLAINT_DATE': 'last', 'COMPLAINT_HOUR' : 'count'}).reset_index()
        temp = raced_df.set_index('COMPLAINT_DATE').groupby(pd.TimeGrouper('Q')).count().dropna().reset_index()
        trace_high = go.Scatter(
                        x=temp['COMPLAINT_DATE'],
                        y=temp['COMPLAINT_HOUR'],
                        name = age,
                        line = dict(color = colors_list[i]),
                        opacity = 1)
        scatters.append(trace_high)

data = scatters
layout = dict(
    title='Black Crimes Count Over Time W.R.T to Age (Quarters)',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    ),
)
ys = [250, 225, 200, 175, 150, 125]
trace0 = go.Scatter(
    x=['2013-07-01', '2014-07-30', '2014-08-01', '2015-01-30', '2016-07-01'],
    y=ys,
    name='BLM Events',
    text=['BLM Movement Founded',
          '#icantbreathe is trending',
          'Furgeson',
          'Baltimore Riots',
          'Dallas Riots'],
    mode='text',
)
data.append(trace0)
founded = {'type':'line', 'x0':'2013-07-13', 'x1':'2013-07-13', 'y0':0, 'y1':250, 'line':dict(color='rgb(0, 0, 0)', width=1)}
icantbreath = {'type':'line', 'x0':'2014-07-17', 'x1':'2014-07-17', 'y0':0, 'y1':250, 'line':dict(color='rgb(0, 0, 0)', width=1)}
furgeson = {'type':'line', 'x0':'2014-08-09', 'x1':'2014-08-09', 'y0':0, 'y1':250, 'line':dict(color='rgb(0, 0, 0)', width=1)}
dallas = {'type':'line', 'x0':'2016-07-07', 'x1':'2016-07-07', 'y0':0, 'y1':250, 'line':dict(color='rgb(0, 0, 0)', width=1)}
baltimore = {'type':'line', 'x0':'2015-01-12', 'x1':'2015-01-12', 'y0':0, 'y1':250, 'line':dict(color='rgb(0, 0, 0)', width=1)}
layout["shapes"] = [founded, icantbreath, furgeson, baltimore, dallas]
fig = dict(data=data, layout=layout)
iplot(fig, filename = "Manually Set Range")
print("Top 10 Catagories Frequency:\n-------------------------------------------------------------")
freq_cat = data_summary['CURRENT_CATEGORY'].value_counts().nlargest(10).index
print('\n'.join(freq_cat))
g = sns.countplot(data=data_summary[data_summary['CURRENT_CATEGORY'].isin(freq_cat)], x='CURRENT_CATEGORY', hue='IS_BLACK_INV', orient='h', order=freq_cat)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
scatters = []
colors_list = ['#2da6fe', '#B4F0AE', '#c47943', '#2fa1d6', '#2fa1d6', '#5a59cc', '#901e86', '#4528C9', '#D0F2AE', '#F1343A']
for i, cat in enumerate(freq_cat):
        raced_df = data_complainant[data_complainant['RACE_OF_COMPLAINANT'] == 'Black']
        raced_df = data_complainant[data_complainant['CURRENT_CATEGORY'] == cat].groupby('LOG_NO').agg({'COMPLAINT_DATE': 'last', 'COMPLAINT_HOUR' : 'count'}).reset_index()
        temp = raced_df.set_index('COMPLAINT_DATE').groupby(pd.TimeGrouper('Q')).count().dropna().reset_index()
        trace_high = go.Scatter(
                        x=temp['COMPLAINT_DATE'],
                        y=temp['COMPLAINT_HOUR'],
                        name = cat,
                        line = dict(color = colors_list[i]),
                        opacity = 1)
        scatters.append(trace_high)

data = scatters
layout = dict(
    title='Black Crimes Count by the 10 Most Frequent Categories (Quarters)',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    ),
)
ys = [250, 225, 200, 175, 150, 125]
trace0 = go.Scatter(
    x=['2013-07-01', '2014-07-30', '2014-08-01', '2015-01-30', '2016-07-01'],
    y=ys,
    name='BLM Events',
    text=['BLM Movement Founded',
          '#icantbreathe is trending',
          'Furgeson',
          'Baltimore Riots',
          'Dallas Riots'],
    mode='text',
)
data.append(trace0)
founded = {'type':'line', 'x0':'2013-07-13', 'x1':'2013-07-13', 'y0':0, 'y1':250, 'line':dict(color='rgb(0, 0, 0)', width=1)}
icantbreath = {'type':'line', 'x0':'2014-07-17', 'x1':'2014-07-17', 'y0':0, 'y1':250, 'line':dict(color='rgb(0, 0, 0)', width=1)}
furgeson = {'type':'line', 'x0':'2014-08-09', 'x1':'2014-08-09', 'y0':0, 'y1':250, 'line':dict(color='rgb(0, 0, 0)', width=1)}
dallas = {'type':'line', 'x0':'2016-07-07', 'x1':'2016-07-07', 'y0':0, 'y1':250, 'line':dict(color='rgb(0, 0, 0)', width=1)}
baltimore = {'type':'line', 'x0':'2015-01-12', 'x1':'2015-01-12', 'y0':0, 'y1':250, 'line':dict(color='rgb(0, 0, 0)', width=1)}
layout["shapes"] = [founded, icantbreath, furgeson, baltimore, dallas]
fig = dict(data=data, layout=layout)
iplot(fig, filename = "Manually Set Range")
