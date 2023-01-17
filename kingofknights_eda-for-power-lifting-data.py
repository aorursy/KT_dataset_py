import plotly
import plotly.offline as off
import plotly.graph_objs as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

off.init_notebook_mode(connected=True)
%matplotlib inline
sns.set(style="ticks", color_codes=True)

meets = pd.read_csv('../input/meets.csv')
powerlift = pd.read_csv('../input/openpowerlifting.csv')
meets.head()
country = meets.groupby('MeetCountry', as_index=False).count()
country.MeetCountry[country.MeetID < 30] = 'Others'
country = country.sort_values(by='MeetID', ascending=False)

trace = go.Pie(labels=country.MeetCountry, values=country.MeetID, hole=0.2, hoverinfo="label+percent+name+value", name='Meeting')
layout = go.Layout(title="Number of Meeting held by each country")
figure = go.Figure(data=[trace], layout=layout)
off.iplot(figure)
meets.Date = pd.to_datetime(meets.Date, format='%Y-%m-%d')
meets['Year'] = pd.DatetimeIndex(meets.Date).year

year = meets.groupby('Year', as_index=False).count()

trace = go.Bar(x=year.Year, y=year.MeetID, name='meeting')
layout = go.Layout(title='How many meeting happened in year', xaxis=dict(title='Years'), yaxis=dict(title='Count'))
figure = go.Figure(data=[trace], layout=layout)
off.iplot(figure)
fedration = meets.groupby('Federation', as_index=False).count()
fedration = fedration.sort_values(by='MeetID')

trace = go.Scatter(y=fedration.Federation, x=fedration.MeetID, mode='markers', marker=dict(size=5))
layout = go.Layout(title='Number of meeting held by Federations', xaxis=dict(title='Count'), yaxis=dict(title='Name'), width=800,
    height=1500)
figure = go.Figure(data=[trace], layout=layout)
off.iplot(figure)
powerlift.head()
categories = ['Squat4Kg', 'BestSquatKg', 'BestBenchKg', 'Deadlift4Kg', 'BestDeadliftKg', 'Bench4Kg']

powerlift[categories] = powerlift[categories].fillna(0)
powerlift['TotalKg'] = 0

for category in categories:
    powerlift['TotalKg'] = powerlift['TotalKg'] + powerlift[category]
    
x_1 = powerlift[powerlift.Sex == 'M'].BodyweightKg
x_2 = powerlift[powerlift.Sex == 'F'].BodyweightKg
x_1 = x_1.dropna()
x_2 = x_2.dropna()
x = [x_1, x_2]
label = ['Male', 'Female']

figure = ff.create_distplot(x, label, bin_size=.2)
figure['layout'].update(title='Distribution of Body weigth through all powerlifters')
off.iplot(figure)
figure = ff.create_facet_grid( powerlift, x='Equipment',y='BodyweightKg', facet_row='Sex',trace_type='box', height=1000, color_name='Equipment')
off.iplot(figure)
figure = ff.create_facet_grid( powerlift, x='Age', facet_col='Sex',trace_type='histogram',color_name='Sex')
off.iplot(figure)
figure = {
    "data": [
        {
            "type": 'violin',
            "x": powerlift['Equipment'] [ powerlift['Sex'] == 'M' ],
            "y": powerlift['Age'] [ powerlift['Sex'] == 'M' ],
            "legendgroup": 'Yes',
            "scalegroup": 'Yes',
            "name": 'Male',
            "side": 'negative',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'blue'
            }
        },
        {
            "type": 'violin',
            "x": powerlift['Equipment'] [ powerlift['Sex'] == 'F' ],
            "y": powerlift['Age'] [ powerlift['Sex'] == 'F' ],
            "legendgroup": 'No',
            "scalegroup": 'No',
            "name": 'Female',
            "side": 'positive',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'green'
            }
        }
    ],
    "layout" : {
        "title":'Distribution of Age by type of Equiment used by Powerlifters',
        "yaxis": {
            "title":'Age',
            "zeroline": False,
        },
        "xaxis":{
            "title":'Type of Equipments',
        },
        "violingap": 0,
        "violinmode": "overlay"
    }
}

off.iplot(figure, validate = False)
powerlift.describe()
def plotonType(yvalue):
    data = powerlift[powerlift[yvalue] != 0]
    figure = ff.create_facet_grid( data, x='Age', y=yvalue, facet_row='Equipment',facet_col='Sex',trace_type='scatter',width=1000)
    off.iplot(figure)
    
categories = ['BodyweightKg', 'Squat4Kg', 'BestSquatKg', 'Bench4Kg', 'Deadlift4Kg', 'TotalKg', 'Wilks']
for category in categories:
    plotonType(category)
