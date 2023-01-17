# Importing libraries

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

result_df = pd.read_csv("../input/StudentsPerformance.csv")
result_df.head()
result_df.shape
result_df.info()
result_df.isnull().sum()
number = len(result_df)
print("Total Number of Students = ", number)
trace = go.Pie(labels = result_df['gender'].value_counts().index,
               values = result_df['gender'].value_counts().values,
               marker = dict(colors=['#009688','#ec5a5a'], line=dict(color='#000000', width=0.2))
              )

plotly.offline.iplot([trace], filename="piechart")
go.Figure()
result_df.head(10)
result_df['test preparation course'].unique()
result_df['race/ethnicity'].unique()

marks_df = result_df[['math score','reading score','writing score']]
marks_df.head()

trace = go.Histogram(x = marks_df['math score'],
                     name = 'Math Score',
                     marker = dict(color='#1E8449'),
                     opacity = 0.70)

trace2 = go.Histogram(x = marks_df['reading score'],
                      name = 'Reading Score',
                      marker = dict(color='#C0392B'),
                      opacity = 0.70)

trace3 = go.Histogram(x = marks_df['writing score'],
                      name = 'Writing Score',
                      marker = dict(color='#2874A6'),
                      opacity = 0.70)

data = [trace, trace2, trace3]

layout = go.Layout(barmode = 'stack', title='Score-Wise Distribution')

fig = go.Figure(data=data, layout = layout)

plotly.offline.iplot(fig)
passed_math = result_df.loc[result_df['math score'] >= 50]['math score'].count()
failed_math = result_df.loc[result_df['math score'] < 50]['math score'].count()

passed_reading = result_df.loc[result_df['reading score'] >= 50]['reading score'].count()
failed_reading = result_df.loc[result_df['reading score'] < 50]['reading score'].count()

passed_writing = result_df.loc[result_df['writing score'] >= 50]['writing score'].count()
failed_writing = result_df.loc[result_df['writing score'] < 50]['writing score'].count()

subjects = ['Math Score', 'Writing Score', 'Reading Score']

trace1 = go.Bar(y = [passed_math,passed_reading,passed_writing],
                x = subjects,
                name = 'Passed',
                marker = dict(color='#BD5039'))
trace2 = go.Bar(y = [failed_math,failed_reading,failed_writing],
                x = subjects,
                name = 'Failed',
                marker = dict(color='#358C73'))

data = [trace1,trace2]
layout = go.Layout(barmode = 'stack',
                   xaxis = dict(tickangle=-25)  )

fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig)
result_df['Total_Score'] = result_df['math score'] + result_df['reading score'] + result_df['writing score']

result_df['Percentage'] = (result_df['Total_Score']/300)*100
result_df.head()
trace = go.Pie(labels = result_df['parental level of education'].value_counts().index,
               values = result_df['parental level of education'].value_counts().values,
               marker = dict(colors=['#922B21','#CD6155','#F1948A','#F5B7B1','#5DADE2','#AED6F1'], 
               line=dict(color='#000000', width=0.2))
              )

plotly.offline.iplot([trace], filename="piechart")
go.Figure()
trace1 = go.Box(y = result_df.loc[result_df['parental level of education'] == "master's degree"]['Percentage'],
                name = "Master's Degree",
                marker = dict(color='#AF4040')
                )

trace2 = go.Box(y = result_df.loc[result_df['parental level of education'] == "bachelor's degree"]['Percentage'],
                name = "Bachelor's Degree",
                marker = dict(color='#4986B6'))

trace3 = go.Box(y = result_df.loc[result_df['parental level of education'] == "associate's degree"]['Percentage'],
                name = "Associate's Degree",
                marker = dict(color='#3C783E'))

trace4 = go.Box(y = result_df.loc[result_df['parental level of education'] == "some college"]['Percentage'],
                name = "Some College",
                marker = dict(color='#BD9428'))

trace5 = go.Box(y = result_df.loc[result_df['parental level of education'] == "high school"]['Percentage'],
                name = "High School",
                marker = dict(color='#6D3E83'))

trace6 = go.Box(y = result_df.loc[result_df['parental level of education'] == "some high school"]['Percentage'],
                name = "Some High School",
                marker = dict(color='#E5751A'))

data = [trace1,trace2,trace3,trace4,trace5,trace6]

fig = go.Figure(data=data)
plotly.offline.iplot(fig)
result_df['test preparation course'].unique()
temp_df = result_df[['lunch','Percentage']]

sns.set_style('darkgrid')
plt.figure(figsize=(12,10))

sns.swarmplot(x = temp_df['lunch'],
            y = temp_df['Percentage'],
            data = temp_df
           )
result_df['lunch'].value_counts()
temp_df = result_df[['test preparation course','Percentage']]

sns.set_style('darkgrid')
plt.figure(figsize=(12,10))

ax = sns.stripplot(x = temp_df['Percentage'],
               y = result_df['test preparation course'],
               data = temp_df,
               jitter = True,
            linewidth = .5,
            palette='YlGnBu_r')

ax.set_xlabel('Percentage',fontsize=16, fontweight='bold', color='#191970')
ax.set_ylabel('test preparation course', fontsize=16, fontweight='bold', color='#191970')
trace0 = go.Violin(y = result_df.loc[result_df['race/ethnicity'] == "group A"]['Percentage'],
                   name = "Group A",
                   box = dict(visible=True),
                   marker = dict(color='#AF4040')
                   )

trace1 = go.Violin(y = result_df.loc[result_df['race/ethnicity'] == "group B"]['Percentage'],
                   name = "Group B",
                   box = dict(visible=True),
                   marker = dict(color='#4986B6')
                   )

trace2 = go.Violin(y = result_df.loc[result_df['race/ethnicity'] == "group C"]['Percentage'],
                   name = "Group C",
                   box = dict(visible=True),
                   marker = dict(color='#3C783E')
                   )

trace3 = go.Violin(y = result_df.loc[result_df['race/ethnicity'] == "group D"]['Percentage'],
                   name = "Group D",
                   box = dict(visible=True),
                   marker = dict(color='#BD9428')
                   )

trace4 = go.Violin(y = result_df.loc[result_df['race/ethnicity'] == "group E"]['Percentage'],
                   name = "Group E",
                   box = dict(visible=True),
                   marker = dict(color='#6D3E83')
                   )


data = [trace0,trace1,trace2,trace3,trace4]

fig = go.Figure(data=data)
plotly.offline.iplot(fig)

