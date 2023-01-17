# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as py
from plotly.graph_objs import Scatter, Layout
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
#set the backgroung style sheet
sns.set_style("whitegrid")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
survey_df = pd.read_csv('../input/SurveySchema.csv')
freeFormResp_df = pd.read_csv('../input/freeFormResponses.csv')
multiChoice_df = pd.read_csv('../input/multipleChoiceResponses.csv')
survey_df.head()
ss = pd.DataFrame(survey_df.loc[1])
ss = ss.drop(['2018 Kaggle Machine Learning and Data Science Survey','Time from Start to Finish (seconds)'],axis=0)
ss[1] = pd.to_numeric(ss[1])
ss = ss.rename(columns={1:'Number of Responders'})
ss.plot(kind='bar',figsize = (15,6))
ss = ss.sort_values('Number of Responders')
print("Questions Which get Least Responce: ")
print(ss.head(3))

print("\nMost Answered Questions: ")
print(ss.tail(3))
kp = multiChoice_df['Q1'][1:].value_counts()
kp
data = [
go.Bar(
    x = list(kp.index),
    y = list(kp.values),
    marker=dict(color=['rgba(55, 128, 191, 1.0)', 'rgba(219, 64, 82, 0.7)',
               'rgba(50, 171, 96, 0.7)', 'rgb(128,0,128)'])
),]
layout= go.Layout(
    title= 'Gender Distribution',
    yaxis=dict(title='Count', ticklen=5, gridwidth=2),
    xaxis=dict(title='Gender', ticklen=5, gridwidth=2)
)
fig= go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Q1')
age_df = multiChoice_df['Q2'][1:].dropna()
order= ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-69', '70-79', '80+']
plt.figure(figsize=(12,5))
sns.countplot(age_df,order=order)
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Age Group Distribution')
plt.show()
DS_df = multiChoice_df['Q26'][1:].dropna()
plt.figure(figsize=(12,5))
sns.countplot(DS_df)
plt.xlabel('Do you consider Your a Data Scientist')
plt.ylabel('Count')
#plt.title('Age Group Distribution')
plt.show()
DP_df = multiChoice_df[['Q2','Q26']][1:].dropna()
#DP_df = DP_df.loc[DP_df['Q3'].isin(country)]
DP_df = DP_df.groupby(['Q26'])['Q2'].value_counts()
xp = ['25-29','22-24','30-34','18-21','35-39','40-44','45-49','50-54','55-59','60-69','80+','70-79']
DP_df
trace1 = go.Bar(
    x=xp,
    y=DP_df['Definitely not'],
    name='Definitely not',
    marker = dict(color="rgb(113, 50, 141)")
)
trace2 = go.Bar(
    x=xp,
    y=DP_df['Definitely yes'],
    name='Definitely yes',
    marker = dict(color="rgb(119, 74, 175)")
)

trace3 = go.Bar(
    x=xp,
    y=DP_df['Maybe'],
    name='Maybe',
    marker = dict(color="rgb(120, 100, 202)")
)

trace4 = go.Bar(
    x=xp,
    y=DP_df['Probably not'],
    name='Probably not',
    marker = dict(color="rgb(117, 127, 221)")
)

trace5 = go.Bar(
    x=xp,
    y=DP_df['Probably yes'],
    name='Probably yes',
    marker = dict(color="rgb(191, 221, 229)")
)

data = [trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')
DP_df = multiChoice_df[['Q1','Q26']][1:].dropna()
plt.figure(figsize=(16,6))
p = ['Definitely yes','Probably yes', 'Maybe','Probably not','Definitely not']
sns.countplot(data=DP_df, x='Q1', hue='Q26')
plt.title('Gender Distribution Vs Data Scientists',fontsize=15)
plt.xlabel('Gender',fontsize=14)
plt.ylabel('Count',fontsize=14)
plt.show()
country = ['United States of Americ', 'India', 'China', 'Brazil',                        
                'Canada', 'Japan', 'Russia', 'Spain', 'Australia',                     
                'France', 'Germany']  
DP_df = multiChoice_df[['Q3','Q26']][1:].dropna()
DP_df = DP_df.loc[DP_df['Q3'].isin(country)]
DP_df = DP_df.groupby(['Q26'])['Q3'].value_counts()
trace1 = go.Bar(
    x=country,
    y=DP_df['Definitely not'],
    name='Definitely not',
    marker = dict(color="rgb(113, 50, 141)")
)
trace2 = go.Bar(
    x=country,
    y=DP_df['Definitely yes'],
    name='Definitely yes',
    marker = dict(color="rgb(119, 74, 175)")
)

trace3 = go.Bar(
    x=country,
    y=DP_df['Maybe'],
    name='Maybe',
    marker = dict(color="rgb(120, 100, 202)")
)

trace4 = go.Bar(
    x=country,
    y=DP_df['Probably not'],
    name='Probably not',
    marker = dict(color="rgb(117, 127, 221)")
)

trace5 = go.Bar(
    x=country,
    y=DP_df['Probably yes'],
    name='Probably yes',
    marker = dict(color="rgb(191, 221, 229)")
)

data = [trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')
roles = ['Student','Software Engineer', 'Chief Officer', 'Consultant', 'DBA/Database Engineer', 
         'Data Analyst', 'Data Engineer', 'Data Journalist', 'Data Scientist', 
         'Developer Advocate', 'Manager', 'Marketing Analyst', 'Not employed', 
         'Other', 'Principal Investigator', 'Product/Project Manager', 
         'Research Assistant', 'Research Scientist', 'Salesperson', 'Software Engineer', 
         'Statistician', ]
DP_df = multiChoice_df[['Q6','Q26']][1:].dropna()
DP_df = DP_df.groupby(['Q26'])['Q6'].value_counts()

DP_df
trace1 = go.Bar(
    x=roles,
    y=DP_df['Definitely not'],
    name='Definitely not',
    marker = dict(color="rgb(113, 50, 141)")
)
trace2 = go.Bar(
    x=roles,
    y=DP_df['Definitely yes'],
    name='Definitely yes',
    marker = dict(color="rgb(119, 74, 175)")
)

trace3 = go.Bar(
    x=roles,
    y=DP_df['Maybe'],
    name='Maybe',
    marker = dict(color="rgb(120, 100, 202)")
)

trace4 = go.Bar(
    x=roles,
    y=DP_df['Probably not'],
    name='Probably not',
    marker = dict(color="rgb(117, 127, 221)")
)

trace5 = go.Bar(
    x=roles,
    y=DP_df['Probably yes'],
    name='Probably yes',
    marker = dict(color="rgb(191, 221, 229)")
)

data = [trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')
country_df = multiChoice_df['Q3'][1:].value_counts()
country_df.head(10)
data = [dict(
        type='choropleth',
        locations = list(country_df.index),
        locationmode='country names',
        z=(country_df.values),
        text=list(country_df.index),
        colorscale='Portland',
        reversescale=True,
)]
layout = dict(
    title = 'A Map About Population of Data Scientists in Each Country',
    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='world-map')
country = ['Australia', 'Brazil', 'Canada','China', 'France', 'Germany',
                'India','Japan', 'Russia', 'Spain', 'United States of America']
DP_df = multiChoice_df[['Q3','Q1']][1:].dropna()
DP_df = DP_df.loc[DP_df['Q3'].isin(country)]
DP_df = DP_df.groupby(['Q1','Q3'])['Q1'].count()
trace1 = go.Bar(
    x=country,
    y=DP_df['Male'],
    name='Male'
)
trace2 = go.Bar(
    x=country,
    y=DP_df['Female'],
    name='Female'
)

trace3 = go.Bar(
    x=country,
    y=DP_df['Prefer not to say'],
    name='Prefer not to say'
)

trace4 = go.Bar(
    x=country,
    y=DP_df['Prefer to self-describe'],
    name='Prefer to self-describe'
)

data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')
df = multiChoice_df[['Q1', 'Q2']][1:].dropna()
df['Q1'] = df['Q1'].replace(['Prefer not to say', 'Prefer to self-describe'], 'Others')

# Select only those indices where the top 10 countries are there
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='Q2', hue='Q1', hue_order=['Female','Male', 'Others'], order=order)
plt.xlabel('Age-group')
plt.ylabel('Count')
plt.title('Gender distribution in different age-groups')
plt.show()
