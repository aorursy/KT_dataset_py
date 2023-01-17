import pandas as pd

import plotly.plotly as py
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

#Read the data
df = pd.read_csv("../input/daily-inmates-in-custody.csv")

#Groups
group_gender = df.groupby("GENDER")
group_race = df.groupby("RACE")
group_custody = df.groupby("CUSTODY_LEVEL")

#Race (data1)
x = group_race.AGE.mean().index
y = round(group_race.AGE.mean(),0).tolist()

data1 = go.Bar(
            x=x,
            y=y,
            name="Race",
            text=y,
            textposition = 'auto',
            marker=dict(
                color='rgb(158,202,225)',
                
            ),
            opacity=0.6
        )

#Age (data2)
x = group_gender.AGE.mean().index
y = round(group_gender.AGE.mean(),0).tolist()

data2 = go.Bar(
            x=x,
            y=y,
            name = "Gender",
            text=y,
            textposition = 'auto',
            marker=dict(
                color='rgb(255,202,225)',
                
            ),
            opacity=0.6
        )

#Custody level (data3)
x = group_custody.AGE.mean().index
y = round(group_custody.AGE.mean(),0).tolist()

data3 = go.Bar(
            x=x,
            y=y,
            name="Custody level",
            text=y,
            textposition = 'auto',
            marker=dict(
                color='rgb(0,202,225)',
                
            ),
            opacity=0.6
        )

updatemenus = list([
    dict(active=-1,
         x=0, y=1.085,
         buttons=list([   
            dict(label = 'All',
                 method = 'update',
                 args = [{'visible': [True, True, True]},
                        {'title': 'Average Age by Race&Gender&Custody level'}]),
            dict(label = 'Race',
                 method = 'update',
                 args = [{'visible': [True, False, False]},
                        {'title': 'Average Age by Race'}]),
             dict(label = 'Gender',
                 method = 'update',
                 args = [{'visible': [False, True, False]},
                         {'title': 'Average Age by Gender'}]),
             dict(label = 'Custody level',
                 method = 'update',
                 args = [{'visible': [False, False, True]},
                         {'title': 'Average Age by Custody level'}])
        ]),
    )
])

data = [data1, data2, data3]
layout = go.Layout(
    title='Average Age by Race&Gender&Custody level',
    yaxis={'title':'Age'} ,  
    updatemenus=updatemenus
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')
race = group_race["INMATEID"].count()
race_labels = race.index.tolist()
race_values = race.tolist()

gender = group_gender["INMATEID"].count()
gender_labels = gender.index.tolist()
gender_values = gender.tolist()

custody = group_custody["INMATEID"].count()
custody_labels = custody.index.tolist()
custody_values = custody.tolist()

fig2 = {
    'data': [
        {
            'labels': race_labels,
            'values': race_values,
            'type': 'pie',
            'name': 'Race',
            'domain': {'x': [0, .33],
                       'y': [0, .99]},
            'hoverinfo':'label+percent+name',
            'textinfo':'label+value'
        },
        {
            'labels': gender_labels,
            'values': gender_values,
            'type': 'pie',
            'name': 'Gender',
            'domain': {'x': [.34, .66],
                       'y': [0, .99]},
            'hoverinfo':'label+percent+name',
            'textinfo':'label+value'
        },
        {
            'labels': custody_labels,
            'values': custody_values,
            'type': 'pie',
            'name': 'Custody',
            'domain': {'x': [.67, 1],
                       'y': [0, .99]},
            'hoverinfo':'label+percent+name',
            'textinfo':'label+value'
        }
    ],
    'layout': {'title': 'Count by Race & Gender & Custody level',
               'showlegend': False}
}

iplot(fig2, filename='pie_chart_counts')