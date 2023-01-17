import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
pd.options.mode.chained_assignment = None

from IPython.display import HTML

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.graph_objs import *
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
gender_data=pd.read_csv('../input/Reveal_EEO1_for_2016.csv')
gender_data.head()
gender_data['count'].replace(to_replace='na',value=0,inplace=True)
gender_data['count']=gender_data['count'].astype(int)
gender_data.head()
#using lambda to aggregate all of the count data from the different type of employees that work at the 15 Silicon Valley Business
#under exploration
company_count=gender_data.groupby(['company']).agg({'count': lambda x: sum((x).astype(int))})
company_count.head()

#using figure to create a large size for our viewing purposes 
plt.figure(figsize=(10,8))

#using whitegrid to identify grid lines in the bar graphs
sns.set_style('whitegrid')

#key to creating bar plot line

sns.barplot(x=company_count.index.get_values(),y=company_count['count'],palette=sns.color_palette("Paired", 10))

plt.title('Silicon Valley Companies',size=25)
plt.ylabel('Number of employees',size=14)
plt.xlabel('Companies',size=14)
plt.yticks(size=14)
plt.xticks(size=14,rotation=90)
sns.despine()
plt.show()
labels = gender_data.groupby(['gender']).agg({'count':sum}).index.get_values()
values = gender_data.groupby(['gender']).agg({'count':sum})['count'].values
colors = ['#a1d99b', '#deebf7']
trace = go.Pie(labels=labels, values=values,
               textinfo="label+percent",
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))
layout=go.Layout(title='Pie Chart of Female and Male Employee')
data=[trace]

fig = dict(data=data,layout=layout)
iplot(fig, filename='Pie Chart of Female and Male Employees')
d=gender_data.groupby(['gender','company']).agg({'count':sum}).reset_index()
trace1 = go.Bar(
    x=d[d.gender=='male']['company'],
    y=d[d.gender=='male']['count'],
    name='Males',
    marker=dict(
        color='rgb(158,202,225)'
    )
)
trace2 = go.Bar(
    x=d[d.gender=='female']['company'],
    y=d[d.gender=='female']['count'],
    name='Females',
    marker=dict(
        color='rgb(161,217,155)'
    )
)
data = [trace1, trace2]
layout = go.Layout(
    barmode='group',title='Distribution of Male and Female Employees by Company')


fig = dict(data=data, layout=layout)
iplot(fig, filename='Distribution of Male and Female Employees by Company')
d=gender_data.groupby(['company','gender']).agg({'count':sum})
d=d.unstack()
d=d['count']
d=np.round(d.iloc[:,:].apply(lambda x: (x/x.sum())*100,axis=1))
d['Ratio']=np.round(d['male']/d['female'],2)
d.sort_values(by='Ratio',inplace=True,ascending=False)
d.columns=['Female %','Male %','Ratio']
d
trace1 = go.Bar(
    y=d.index.get_values(),
    x=d['Ratio'],text=d['Ratio'],textposition='auto',
    orientation='h',
    marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),
    opacity=0.6
)

data = [trace1]
layout = go.Layout(
    barmode='group',title='Ratio of Male to Female Employees')

fig = dict(data=data, layout=layout)
iplot(fig, filename='Ratio of Male to Female Employees')
