# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import seaborn as sns
# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
school_explorer = pd.read_csv('../input/2016 School Explorer.csv')
shsat_reg = pd.read_csv('../input/D5 SHSAT Registrations and Testers.csv')
shsat_reg.head()
shsat_reg.describe()
shsat_reg.info()
df2013 = shsat_reg[shsat_reg['Year of SHST'] == 2013]
df2014 = shsat_reg[shsat_reg['Year of SHST'] == 2014]
df2015 = shsat_reg[shsat_reg['Year of SHST'] == 2015]
df2016 = shsat_reg[shsat_reg['Year of SHST'] == 2016]
# Create trace1
trace1 = go.Scatter(
    x = df2013['Year of SHST'],
    y = df2013['Enrollment on 10/31'],
    mode = "markers",
    name = "2013",
    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
    text= df2013['School name']
)

# Create trace2
trace2 = go.Scatter(
    x = df2014['Year of SHST'],
    y = df2014['Enrollment on 10/31'],
    mode = "markers",
    name = "2014",
    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
    text= df2014['School name']
)

# Create trace3
trace3 = go.Scatter(
    x = df2015['Year of SHST'],
    y = df2015['Enrollment on 10/31'],
    mode = "markers",
    name = "2015",
    marker = dict(color = 'rgba(0, 55, 200, 1.8)'),
    text= df2015['School name']
)

# Create trace4
trace4 = go.Scatter(
    x = df2016['Year of SHST'],
    y = df2016['Enrollment on 10/31'],
    mode = "markers",
    name = "2016",
    marker = dict(color = 'rgba(200, 255, 0, 0.8)'),
    text= df2016['School name']
)

data = [trace1, trace2, trace3, trace4]
layout = dict(title = 'Year of SHST vs Enrollment on 10/31',
              xaxis= dict(title= 'Year of SHST',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Enrollment on 10/31',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
df2013 = shsat_reg[shsat_reg['Year of SHST'] == 2013]
df2014 = shsat_reg[shsat_reg['Year of SHST'] == 2014]
df2015 = shsat_reg[shsat_reg['Year of SHST'] == 2015]
df2016 = shsat_reg[shsat_reg['Year of SHST'] == 2016]
# Create trace1
trace1 = go.Scatter(
    x = df2013['Year of SHST'],
    y = df2013['Number of students who registered for the SHSAT'],
    mode = "markers",
    name = "2013",
    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
    text= df2013['School name']
)

# Create trace2
trace2 = go.Scatter(
    x = df2014['Year of SHST'],
    y = df2014['Number of students who registered for the SHSAT'],
    mode = "markers",
    name = "2014",
    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
    text= df2014['School name']
)

# Create trace3
trace3 = go.Scatter(
    x = df2015['Year of SHST'],
    y = df2015['Number of students who registered for the SHSAT'],
    mode = "markers",
    name = "2015",
    marker = dict(color = 'rgba(0, 55, 200, 1.8)'),
    text= df2015['School name']
)

# Create trace4
trace4 = go.Scatter(
    x = df2016['Year of SHST'],
    y = df2016['Number of students who registered for the SHSAT'],
    mode = "markers",
    name = "2016",
    marker = dict(color = 'rgba(200, 255, 0, 0.8)'),
    text= df2016['School name']
)

data = [trace1, trace2, trace3, trace4]
layout = dict(title = 'Year of SHST vs Number of students who registered for the SHSAT',
              xaxis= dict(title= 'Year of SHST',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Number of students who registered for the SHSAT',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
df2013 = shsat_reg[shsat_reg['Year of SHST'] == 2013]
df2014 = shsat_reg[shsat_reg['Year of SHST'] == 2014]
df2015 = shsat_reg[shsat_reg['Year of SHST'] == 2015]
df2016 = shsat_reg[shsat_reg['Year of SHST'] == 2016]
# Create trace1
trace1 = go.Scatter(
    x = df2013['Year of SHST'],
    y = df2013['Number of students who took the SHSAT'],
    mode = "markers",
    name = "2013",
    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
    text= df2013['School name']
)

# Create trace2
trace2 = go.Scatter(
    x = df2014['Year of SHST'],
    y = df2014['Number of students who took the SHSAT'],
    mode = "markers",
    name = "2014",
    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
    text= df2014['School name']
)

# Create trace3
trace3 = go.Scatter(
    x = df2015['Year of SHST'],
    y = df2015['Number of students who took the SHSAT'],
    mode = "markers",
    name = "2015",
    marker = dict(color = 'rgba(0, 55, 200, 1.8)'),
    text= df2015['School name']
)

# Create trace4
trace4 = go.Scatter(
    x = df2016['Year of SHST'],
    y = df2016['Number of students who took the SHSAT'],
    mode = "markers",
    name = "2016",
    marker = dict(color = 'rgba(200, 255, 0, 0.8)'),
    text= df2016['School name']
)

data = [trace1, trace2, trace3, trace4]
layout = dict(title = 'Year of SHST vs Number of students who took the SHSAT',
              xaxis= dict(title= 'Year of SHST',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Number of students who took the SHSAT',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
df2013 = shsat_reg[shsat_reg['Year of SHST'] == 2013]
df2014 = shsat_reg[shsat_reg['Year of SHST'] == 2014]
df2015 = shsat_reg[shsat_reg['Year of SHST'] == 2015]
df2016 = shsat_reg[shsat_reg['Year of SHST'] == 2016]

total_registrants = [df2013['Number of students who registered for the SHSAT'].sum(), 
                    df2014['Number of students who registered for the SHSAT'].sum(),
                    df2015['Number of students who registered for the SHSAT'].sum(),
                    df2016['Number of students who registered for the SHSAT'].sum()]

years = [2013,2014,2015,2016]

trace = go.Bar(
                x = years,
                y = total_registrants,
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                )

data = [trace]
layout = go.Layout(barmode = "group", xaxis= dict(title= 'Year',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Number of Students',ticklen= 5,zeroline= False), 
                  title='Total students who registered each year')
fig = go.Figure(data = data, layout = layout)
iplot(fig)
df2013 = shsat_reg[shsat_reg['Year of SHST'] == 2013]
df2014 = shsat_reg[shsat_reg['Year of SHST'] == 2014]
df2015 = shsat_reg[shsat_reg['Year of SHST'] == 2015]
df2016 = shsat_reg[shsat_reg['Year of SHST'] == 2016]

total_students_who_took = [df2013['Number of students who took the SHSAT'].sum(), 
                    df2014['Number of students who took the SHSAT'].sum(),
                    df2015['Number of students who took the SHSAT'].sum(),
                    df2016['Number of students who took the SHSAT'].sum()]

years = [2013,2014,2015,2016]

trace = go.Bar(
                x = years,
                y = total_students_who_took,
                marker = dict(color = 'rgba(0, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                )

data = [trace]
layout = go.Layout(barmode = "group", xaxis= dict(title= 'Year',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Number of Students',ticklen= 5,zeroline= False), 
                  title='Total students who took SHSAT for each year')
fig = go.Figure(data = data, layout = layout)
iplot(fig)
df2013 = shsat_reg[shsat_reg['Year of SHST'] == 2013]
df2014 = shsat_reg[shsat_reg['Year of SHST'] == 2014]
df2015 = shsat_reg[shsat_reg['Year of SHST'] == 2015]
df2016 = shsat_reg[shsat_reg['Year of SHST'] == 2016]

top3_school_regis_2013 = pd.DataFrame(df2013.groupby('School name')['Number of students who registered for the SHSAT'].sum().sort_values(ascending=False).reset_index())['School name'][:3]
num_of_students_2013 = pd.DataFrame(df2013.groupby('School name')['Number of students who registered for the SHSAT'].sum().sort_values(ascending=False).reset_index())['Number of students who registered for the SHSAT'][:3]

top3_school_regis_2014 = pd.DataFrame(df2014.groupby('School name')['Number of students who registered for the SHSAT'].sum().sort_values(ascending=False).reset_index())['School name'][:3]
num_of_students_2014 = pd.DataFrame(df2014.groupby('School name')['Number of students who registered for the SHSAT'].sum().sort_values(ascending=False).reset_index())['Number of students who registered for the SHSAT'][:3]

top3_school_regis_2015 = pd.DataFrame(df2015.groupby('School name')['Number of students who registered for the SHSAT'].sum().sort_values(ascending=False).reset_index())['School name'][:3]
num_of_students_2015 = pd.DataFrame(df2015.groupby('School name')['Number of students who registered for the SHSAT'].sum().sort_values(ascending=False).reset_index())['Number of students who registered for the SHSAT'][:3]

top3_school_regis_2016 = pd.DataFrame(df2016.groupby('School name')['Number of students who registered for the SHSAT'].sum().sort_values(ascending=False).reset_index())['School name'][:3]
num_of_students_2016 = pd.DataFrame(df2016.groupby('School name')['Number of students who registered for the SHSAT'].sum().sort_values(ascending=False).reset_index())['Number of students who registered for the SHSAT'][:3]

years = [2013,2014,2015,2016]
rank=['First','Second','Third']
trace1 = go.Bar(
                x = rank,
                y = num_of_students_2013,
                marker = dict(color = 'rgba(100, 50, 5, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                name = "2013",
                text = top3_school_regis_2013
                )

trace2 = go.Bar(
                x = rank,
                y = num_of_students_2014,
                marker = dict(color = 'rgba(100, 174, 55, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                name = "2014",
                text = top3_school_regis_2014
                )

trace3 = go.Bar(
                x = rank,
                y = num_of_students_2015,
                marker = dict(color = 'rgba(10, 14, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                name = "2015",
                text = top3_school_regis_2015
                )

trace4 = go.Bar(
                x = rank,
                y = num_of_students_2016,
                marker = dict(color = 'rgba(0, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                name = "2016",
                text = top3_school_regis_2016
                )

data = [trace1,trace2,trace3,trace4]
layout = go.Layout(barmode = "group", xaxis= dict(title= 'Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Number of Students',ticklen= 5,zeroline= False), 
                  title='Top 3 Schools for each year')
fig = go.Figure(data = data, layout = layout)
iplot(fig)
df2013 = shsat_reg[shsat_reg['Year of SHST'] == 2013]
df2014 = shsat_reg[shsat_reg['Year of SHST'] == 2014]
df2015 = shsat_reg[shsat_reg['Year of SHST'] == 2015]
df2016 = shsat_reg[shsat_reg['Year of SHST'] == 2016]

top3_school_took_2013 = pd.DataFrame(df2013.groupby('School name')['Number of students who took the SHSAT'].sum().sort_values(ascending=False).reset_index())['School name'][:3]
num_of_students_2013 = pd.DataFrame(df2013.groupby('School name')['Number of students who took the SHSAT'].sum().sort_values(ascending=False).reset_index())['Number of students who took the SHSAT'][:3]

top3_school_took_2014 = pd.DataFrame(df2014.groupby('School name')['Number of students who took the SHSAT'].sum().sort_values(ascending=False).reset_index())['School name'][:3]
num_of_students_2014 = pd.DataFrame(df2014.groupby('School name')['Number of students who took the SHSAT'].sum().sort_values(ascending=False).reset_index())['Number of students who took the SHSAT'][:3]

top3_school_took_2015 = pd.DataFrame(df2015.groupby('School name')['Number of students who took the SHSAT'].sum().sort_values(ascending=False).reset_index())['School name'][:3]
num_of_students_2015 = pd.DataFrame(df2015.groupby('School name')['Number of students who took the SHSAT'].sum().sort_values(ascending=False).reset_index())['Number of students who took the SHSAT'][:3]

top3_school_took_2016 = pd.DataFrame(df2016.groupby('School name')['Number of students who took the SHSAT'].sum().sort_values(ascending=False).reset_index())['School name'][:3]
num_of_students_2016 = pd.DataFrame(df2016.groupby('School name')['Number of students who took the SHSAT'].sum().sort_values(ascending=False).reset_index())['Number of students who took the SHSAT'][:3]

years = [2013,2014,2015,2016]
rank=['First','Second','Third']
trace1 = go.Bar(
                x = rank,
                y = num_of_students_2013,
                marker = dict(color = 'rgba(100, 50, 5, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                name = "2013",
                text = top3_school_took_2013
                )

trace2 = go.Bar(
                x = rank,
                y = num_of_students_2014,
                marker = dict(color = 'rgba(100, 174, 55, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                name = "2014",
                text = top3_school_took_2014
                )

trace3 = go.Bar(
                x = rank,
                y = num_of_students_2015,
                marker = dict(color = 'rgba(10, 14, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                name = "2015",
                text = top3_school_took_2015
                )

trace4 = go.Bar(
                x = rank,
                y = num_of_students_2016,
                marker = dict(color = 'rgba(0, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                name = "2016",
                text = top3_school_took_2016
                )

data = [trace1,trace2,trace3,trace4]
layout = go.Layout(barmode = "group", xaxis= dict(title= 'Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Number of Students',ticklen= 5,zeroline= False), 
                  title='Top 3 Schools for each year')
fig = go.Figure(data = data, layout = layout)
iplot(fig)
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['font.size'] = 15
plt.rcParams['figure.figsize'] = (12.0, 8.0)
sns.pointplot(y="Number of students who registered for the SHSAT", x="Year of SHST", hue="Grade level", data=shsat_reg);
sns.pointplot(y="Number of students who took the SHSAT", x="Year of SHST", hue="Grade level", data=shsat_reg);
df = school_explorer
df.head()
df.isnull().sum()
sns.countplot(df['Community School?'])
sns.countplot(df['Grade Low'])
sns.countplot(df['Grade High'])
data = df['Economic Need Index'].dropna()
sns.distplot(data)
data = df['Economic Need Index'].dropna()
trace = go.Box(
    y=data,
    name = 'Economic Need Index',
    marker = dict(
        color = 'rgb(214, 12, 140)',
    )
)
iplot([trace])
data = df['School Income Estimate'].dropna()
trace = go.Box(
    y=data,
    name = 'School Income Estimate',
    marker = dict(
        color = 'rgb(114, 12, 14)',
    )
)
iplot([trace])
data = data.dropna()
data = df['School Income Estimate'].str.replace(',','')
data = data.str.replace('$','')
data = data.astype(float)
df[data > 100000][['School Name', 'School Income Estimate']].sort_values(by='School Income Estimate',ascending=False)[:6]