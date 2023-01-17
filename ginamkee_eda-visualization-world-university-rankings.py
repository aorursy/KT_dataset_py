import numpy as np 

import pandas as pd





%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

from plotly.offline import init_notebook_mode, iplot



# for visualization

import plotly.graph_objs as go

import plotly.express as px



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
cwur = pd.read_csv('/kaggle/input/world-university-rankings/cwurData.csv')

times = pd.read_csv('/kaggle/input/world-university-rankings/timesData.csv')

attainment = pd.read_csv('/kaggle/input/world-university-rankings/educational_attainment_supplementary_data.csv')

shanghai = pd.read_csv('/kaggle/input/world-university-rankings/shanghaiData.csv')

country = pd.read_csv('/kaggle/input/world-university-rankings/school_and_country_table.csv')
fee = pd.read_csv('/kaggle/input/world-university-rankings/education_expenditure_supplementary_data.csv', engine='python')
cwur['country'].replace('USA', 'United States of America', inplace=True)
times.head()
times['num_students'] = times['num_students'].str.replace(',', '')

numstudents = times[times['num_students'].notna()]

numstudents['num_students'] = numstudents['num_students'].astype(int)
times['num_students'].fillna(23873, inplace=True)

times['num_students'] = times['num_students'].astype(int)
south_korea = times.loc[times['country'] == 'South Korea']

United_States_of_America = times.loc[times['country'] == 'United States of America']

United_Kingdom = times.loc[times['country'] == 'United Kingdom']

Germany = times.loc[times['country'] == 'Germany']

Australia = times.loc[times['country'] == 'Australia']

Canada = times.loc[times['country'] == 'Canada']

Japan = times.loc[times['country'] == 'Japan']

Italy = times.loc[times['country'] == 'Italy']

Spain = times.loc[times['country'] == 'Spain']

China = times.loc[times['country'] == 'China']

Netherlands = times.loc[times['country'] == 'Netherlands']

France = times.loc[times['country'] == 'France']

Taiwan = times.loc[times['country'] == 'Taiwan']

Sweden = times.loc[times['country'] == 'Sweden']

Switzerland = times.loc[times['country'] == 'Switzerland']
trace1 = go.Scatter(

                    x = times.world_rank,

                    y = times.research,

                    mode = "lines+markers",

                    name = "research",

                    marker = dict(color = 'rgba(16, 126, 80, 0.8)'),

                    text= times.university_name)

trace2 = go.Scatter(

                    x = times.world_rank,

                    y = times.total_score,

                    mode = "lines+markers",

                    name = "total_score",

                    marker = dict(color = 'rgba(246, 226, 80, 0.8)'),

                    text= times.university_name)



trace3 = go.Scatter(

                    x = times.world_rank,

                    y = times.teaching,

                    mode = "lines+markers",

                    name = "teaching",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= times.university_name)



data = [trace1, trace2, trace3]

layout = dict(xaxis= dict(title= 'World Rank', ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
trace1 = go.Scatter(

                    x = times.world_rank,

                    y = times.income,

                    mode = "lines+markers",

                    name = "income",

                    marker = dict(color = 'rgba(16, 126, 80, 0.8)'),

                    text= times.university_name)

trace2 = go.Scatter(

                    x = times.world_rank,

                    y = times.international,

                    mode = "lines+markers",

                    name = "international",

                    marker = dict(color = 'rgba(246, 226, 80, 0.8)'),

                    text= times.university_name)



data = [trace1, trace2]

layout = dict(xaxis= dict(title= 'World Rank', ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
X = times['country']



plt.figure(figsize = (16,10))

plt.xticks(fontsize=12, rotation=90 )

sns.countplot(x='country', data=times)
south_korea2014 = times.loc[times.year == 2014]

south_korea2014 = south_korea2014.loc[south_korea2014.country == 'South Korea']

south_korea2015 = times.loc[times.year == 2015]

south_korea2015 = south_korea2015.loc[south_korea2015.country == 'South Korea']

south_korea2016 = times.loc[times.year == 2016]

south_korea2016 = south_korea2016.loc[south_korea2016.country == 'South Korea']



trace1 = go.Scatter(x = south_korea2014.teaching,

                    y = south_korea2014.world_rank,

                    mode = 'markers',

                    name = '2014',

                    marker = dict(color = 'rgba(200, 120, 55, 98)')

)



trace2 = go.Scatter(x = south_korea2015.teaching,

                    y = south_korea2015.world_rank,

                    mode = 'markers',

                    name = '2015',

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)')

)



trace3 = go.Scatter(x = south_korea2016.teaching,

                    y = south_korea2016.world_rank,

                    mode = 'markers',

                    name = '2016',

                    marker = dict(color = 'rgba(50, 250, 55, 0.8)')

)



data = [trace1, trace2, trace3]

layout = dict(title = 'South Korea students teaching vs world rank 2014~2016',

             xaxis = dict(title = 'Teaching'),

             yaxis = dict(title = 'World rank'))

fig = go.Figure(data = data, layout = layout)

fig
south_korea = south_korea.loc[south_korea.year == 2016]

south_korea = south_korea.head(5)



trace1 = go.Bar(x = south_korea.university_name,

                y = south_korea.teaching,

                name = 'teaching',

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5))

)

trace2 = go.Bar(x = south_korea.university_name,

                y = south_korea.research,

                name = 'research',

                marker = dict(color = 'rgba(12, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5))

)

data = [trace1, trace2]

layout = go.Layout(barmode = "group", width=800, height=600)

fig = go.Figure(data = data, layout = layout)

fig
south_korea_2016 = south_korea.loc[south_korea['year'] == 2016]
south_korea_2016['num_students'].mean()
top_10_times_2011 = times.head(10)

top_10_times_2012 = times.loc[times['year'] == 2012].head(10)

top_10_times_2013 = times.loc[times['year'] == 2013].head(10)

top_10_times_2014 = times.loc[times['year'] == 2014].head(10)

top_10_times_2015 = times.loc[times['year'] == 2015].head(10)

top_10_times_2016 = times.loc[times['year'] == 2016].head(10)
bottom_10_times_2011 = times.loc[times['year'] == 2011].tail(10)

bottom_10_times_2012 = times.loc[times['year'] == 2012].tail(10)

bottom_10_times_2013 = times.loc[times['year'] == 2013].tail(10)

bottom_10_times_2014 = times.loc[times['year'] == 2014].tail(10)

bottom_10_times_2015 = times.loc[times['year'] == 2015].tail(10)

bottom_10_times_2016 = times.loc[times['year'] == 2016].tail(10)
top_10_times = top_10_times_2011.append(top_10_times_2012)

top_10_times = top_10_times.append(top_10_times_2013)

top_10_times = top_10_times.append(top_10_times_2014)

top_10_times = top_10_times.append(top_10_times_2015)

top_10_times = top_10_times.append(top_10_times_2016)
bottom_10_times = bottom_10_times_2011.append(bottom_10_times_2012)

bottom_10_times = bottom_10_times.append(bottom_10_times_2013)

bottom_10_times = bottom_10_times.append(bottom_10_times_2014)

bottom_10_times = bottom_10_times.append(bottom_10_times_2015)

bottom_10_times = bottom_10_times.append(bottom_10_times_2016)
top_10_times['international_students_num'] = top_10_times.international_students.str.extract('(\d+)')

bottom_10_times['international_students_avg'] = bottom_10_times.international_students.str.extract('(\d+)')
top_10_times.head()
top_10_times['international_students_num'] = top_10_times['international_students_num'].astype(int)
international_students_avg = bottom_10_times[bottom_10_times['international_students_avg'].notna()]
international_students_avg['international_students_avg'] = international_students_avg['international_students_avg'].astype(int)
bottom_10_times['international_students_avg'].fillna('11', inplace=True)
bottom_10_times['international_students_avg'].isnull().sum()
bottom_10_times['international_students_avg'] = bottom_10_times['international_students_avg'].astype(int)
avg = bottom_10_times[bottom_10_times['num_students'].notna()]
avg['num_students'] = avg['num_students'].astype(int)
bottom_10_times['num_students'].fillna('24271', inplace=True)
bottom_10_times['num_students'] = bottom_10_times['num_students'].astype(int)
cnt_ = bottom_10_times['country'].value_counts()

fig = { "data": [{

            "values": cnt_.values,

            "labels": cnt_.index,

            "domain": {"x": [0, .5]},

            "name": "Train types",

            "hoverinfo":"label+percent+name",

            "hole": .7,

            "type": "pie"

        }],

        "layout": {            

            "title":"Pie chart",

            "annotations": [{

            "font": { "size": 20},

        "showarrow": False,

        "text": "Pie Chart",

        "x": 0.50,

        "y": 1

        },

        ]

    }

}

iplot(fig)
USA_2014 = times.loc[times.year == 2014]

USA_2014 = USA_2014.loc[USA_2014.country == 'United States of America']

USA_2015 = times.loc[times.year == 2015]

USA_2015 = USA_2015.loc[USA_2015.country == 'United States of America']

USA_2016 = times.loc[times.year == 2016]

USA_2016 = USA_2016.loc[USA_2016.country == 'United States of America']



trace1 = go.Scatter(x = USA_2014.teaching,

                    y = USA_2014.world_rank,

                    mode = 'markers',

                    name = '2014',

                    marker = dict(color = 'rgba(200, 120, 55, 98)')

)



trace2 = go.Scatter(x = USA_2015.teaching,

                    y = USA_2015.world_rank,

                    mode = 'markers',

                    name = '2015',

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)')

)



trace3 = go.Scatter(x = USA_2016.teaching,

                    y = USA_2016.world_rank,

                    mode = 'markers',

                    name = '2016',

                    marker = dict(color = 'rgba(50, 250, 55, 0.8)')

)



data = [trace1, trace2, trace3]

layout = dict(title = 'United States of America students teaching vs world rank 2014~2016',

             xaxis = dict(title = 'Teaching'),

             yaxis = dict(title = 'World Rank'))

fig = go.Figure(data = data, layout = layout)

fig
USA_2016 = times.loc[times.year == 2016]

USA_2016 = USA_2016.loc[USA_2016.country == 'United States of America']

UK_2016 = times.loc[times.year == 2016]

UK_2016 = UK_2016.loc[UK_2016.country == 'United Kingdom']

Germany_2016 = times.loc[times.year == 2016]

Germany_2016 = Germany_2016.loc[Germany_2016.country == 'Germany']

Australia_2016 = times.loc[times.year == 2016]

Australia_2016 = Australia_2016.loc[Australia_2016.country == 'Australia']

Canada_2016 = times.loc[times.year == 2016]

Canada_2016 = Canada_2016.loc[Canada_2016.country == 'Canada']

Japan_2016 = times.loc[times.year == 2016]

Japan_2016 = Japan_2016.loc[Japan_2016.country == 'Japan']

Italy_2016 = times.loc[times.year == 2016]

Italy_2016 = Italy_2016.loc[Italy_2016.country == 'Italy']

China_2016 = times.loc[times.year == 2016]

China_2016 = China_2016.loc[China_2016.country == 'China']

Netherlands_2016 = times.loc[times.year == 2016]

Netherlands_2016 = Netherlands_2016.loc[Netherlands_2016.country == 'Netherlands']

France_2016 = times.loc[times.year == 2016]

France_2016 = France_2016.loc[France_2016.country == 'France']

Sweden_2016 = times.loc[times.year == 2016]

Sweden_2016 = Sweden_2016.loc[Sweden_2016.country == 'Sweden']

Taiwan_2016 = times.loc[times.year == 2016]

Taiwan_2016 = Taiwan_2016.loc[Taiwan_2016.country == 'Taiwan']

Spain_2016 = times.loc[times.year == 2016]

Spain_2016 = Spain_2016.loc[Spain_2016.country == 'Spain']

south_korea2016 = times.loc[times.year == 2016]

south_korea2016 = south_korea2016.loc[south_korea2016.country == 'South Korea']

Switzerland_2016 = times.loc[times.year == 2016]

Switzerland_2016 = Switzerland_2016.loc[Switzerland_2016.country == 'Switzerland']



trace1 = go.Scatter(x = USA_2016.teaching,

                    y = USA_2016.world_rank,

                    mode = 'markers',

                    marker = dict(color = 'rgba(255, 55, 255, 255)'),

                    name = 'USA'

)

trace2 = go.Scatter(x = UK_2016.teaching,

                    y = UK_2016.world_rank,

                    mode = 'markers',

                    marker = dict(color = 'rgba(155, 215, 55, 55)'),

                    name = 'UK'

)

trace3 = go.Scatter(x = Germany_2016.teaching,

                    y = Germany_2016.world_rank,

                    mode = 'markers',

                    marker = dict(color = 'rgba(135, 255, 55, 55)'),

                    name = 'Germany'

)

trace4 = go.Scatter(x = Australia_2016.teaching,

                    y = Australia_2016.world_rank,

                    mode = 'markers',

                    marker = dict(color = 'rgba(55, 55, 55, 123)'),

                    name = 'Australia'

)

trace5 = go.Scatter(x = Canada_2016.teaching,

                    y = Canada_2016.world_rank,

                    mode = 'markers',

                    marker = dict(color = 'rgba(155, 55, 55, 125)'),

                    name = 'Canada'

)

trace6 = go.Scatter(x = Japan_2016.teaching,

                    y = Japan_2016.world_rank,

                    mode = 'markers',

                    marker = dict(color = 'rgba(125, 55, 55, 55)'),

                    name = 'Japan'

)

trace7 = go.Scatter(x = Italy_2016.teaching,

                    y = Italy_2016.world_rank,

                    mode = 'markers',

                    marker = dict(color = 'rgba(55, 65, 5, 5)'),

                    name = 'Italy'

)

trace8 = go.Scatter(x = China_2016.teaching,

                    y = China_2016.world_rank,

                    mode = 'markers',

                    marker = dict(color = 'rgba(235, 55, 55, 55)'),

                    name = 'China'

)

trace9 = go.Scatter(x = Netherlands_2016.teaching,

                    y = Netherlands_2016.world_rank,

                    mode = 'markers',

                    marker = dict(color = 'rgba(18, 55, 95, 65)'),

                    name = 'Netherlands'

)

trace10 = go.Scatter(x = France_2016.teaching,

                    y = France_2016.world_rank,

                    mode = 'markers',

                    marker = dict(color = 'rgba(155, 55, 55, 55)'),

                    name = 'France'

)

trace11 = go.Scatter(x = Sweden_2016.teaching,

                    y = Sweden_2016.world_rank,

                    mode = 'markers',

                    marker = dict(color = 'rgba(55, 55, 55, 55)'),

                    name = 'Sweden'

)

trace12 = go.Scatter(x = Taiwan_2016.teaching,

                    y = Taiwan_2016.world_rank,

                    mode = 'markers',

                    marker = dict(color = 'rgba(255, 55, 55, 55)'),

                    name = 'Taiwan'

)

trace13 = go.Scatter(x = Spain_2016.teaching,

                    y = Spain_2016.world_rank,

                    mode = 'markers',

                    marker = dict(color = 'rgba(65, 75, 95, 55)'),

                    name = 'Spain'

)

trace14 = go.Scatter(x = south_korea2016.teaching,

                    y = south_korea2016.world_rank,

                    mode = 'markers',

                    marker = dict(color = 'rgba(205, 125, 155, 55)'),

                    name = 'South Korea'

)

trace15 = go.Scatter(x = Switzerland_2016.teaching,

                    y = Switzerland_2016.world_rank,

                    mode = 'markers',

                    marker = dict(color = 'rgba(155, 5, 5, 55)'),

                    name = 'Switzerland'

)





data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12, trace13, trace14, trace15]

layout = dict(title = 'The distribution of the ranking by the top 15 countries in 2016.',

             xaxis = dict(title = 'Teaching'),

             yaxis = dict(title = 'World rank'))

fig = go.Figure(data = data, layout = layout)

fig
trace1 = go.Box( y= times.teaching,

                name = 'teaching',

                marker = dict(color = 'rgba(12, 12, 140)')



)

trace2 = go.Box( y= times.total_score,

                name = 'total_score',

                marker = dict(color = 'rgb(225, 12, 140)')



)

trace3 = go.Box( y= times.international,

                name = 'international',

                marker = dict(color = 'rgb(95, 192, 250)')



)

trace4 = go.Box( y= times.research,

                name = 'research',

                marker = dict(color = 'rgb(95, 52, 20)')



)

trace5 = go.Box( y= times.income,

                name = 'income',

                marker = dict(color = 'rgb(110, 72, 240)')



)

data = [trace1, trace2, trace3, trace4, trace5]

layout = dict(width = 600, height = 600)

fig = go.Figure(data = data, layout = layout)

fig
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

countries = times['country']

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(countries))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()