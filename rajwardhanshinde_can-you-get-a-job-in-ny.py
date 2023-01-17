!pip install squarify

!pip install bubbly
import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import folium

from folium.plugins import FastMarkerCluster

from wordcloud import WordCloud

from nltk.corpus import stopwords

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



from bubbly.bubbly import bubbleplot

import squarify



import os

print(os.listdir("../input"))
df = pd.read_csv('../input/nyc-jobs.csv')

df.head()
print("There are {} rows and {} columns in our data".format(df.shape[0], df.shape[1]))
df.info()
def pie_plot(cnts, colors, title):

    labels = cnts.index

    values = cnts.values

    

    trace = go.Pie(labels=labels,

                   values=values,

                   title=title,

                   textinfo='value',

                   hoverinfo='label+percent',

                   hole=.77,

                   textposition='inside',

                   marker=dict(colors=colors,

                               line=dict(color='#000000', width=2)

                              )

                  )

    return trace
py.iplot([pie_plot(df['Posting Type'].value_counts(), ['gold', 'cyan'], 'Posting Types')])
plt.style.use('dark_background')

plt.figure(figsize=(20,20))



vacancies = df.groupby(by='Civil Service Title')['# Of Positions'].sum().sort_values(ascending=False).head(15)

plt.subplot(211)

color = plt.cm.spring(np.linspace(0, 1, 40))

vacancies.plot.bar(color=color)

plt.xticks(rotation=60)



vacancies = df.groupby(by='Business Title')['# Of Positions'].sum().sort_values(ascending=False).head(15)

plt.subplot(212)

color = plt.cm.autumn_r(np.linspace(0, 1, 40))

vacancies.plot.bar(color=color)

plt.xticks(rotation=60)



plt.subplots_adjust(hspace=0.4, wspace=0.4)

plt.tight_layout()

plt.show()
def createhmap(keys, vals):

    x = 0.

    y = 0.

    width = 100.

    height = 100.

    colcnt = 0

    values = vals



    normed = squarify.normalize_sizes(values, width, height)

    rects = squarify.squarify(normed, x, y, width, height)



    color_brewer = ['#f4c242']

    shapes = []

    annotations = []

    counter = 0



    for r in rects:

        shapes.append( 

            dict(

                type = 'rect', 

                x0 = r['x'], 

                y0 = r['y'], 

                x1 = r['x']+r['dx'], 

                y1 = r['y']+r['dy'],

                line = dict( width = 5, color="#fff" ),

                fillcolor = '#f4c242'

            ) 

        )

        annotations.append(

            dict(

                x = r['x']+(r['dx']/2),

                y = r['y']+(r['dy']/2),

                text = str(list(keys)[counter]) +" ("+ str(values[counter]) + ")",

                showarrow = False

            )

        )

        counter = counter + 1

        colcnt+=1

        if colcnt >= len(color_brewer):

            colcnt = 0



    # For hover text

    trace0 = go.Scatter(

        x = [ r['x']+(r['dx']/2) for r in rects ], 

        y = [ r['y']+(r['dy']/2) for r in rects ],

        text = [ str(v)+" ("+str(values[k])+" )" for k,v in enumerate(keys) ], 

        mode = 'text',

    )



    layout = dict(

        height=500, width=900,

        margin = dict(l=100),

        xaxis=dict(

                autorange=True,

                showgrid=False,

                zeroline=False,

                showline=False,

                ticks='',

                showticklabels=False

            ),

        yaxis=dict(

                autorange=True,

                showgrid=False,

                zeroline=False,

                showline=False,

                ticks='',

                showticklabels=False

            ),

        shapes=shapes,

        annotations=annotations,

        title="Top Job Categories"

    )

    

    figure = dict(data=[trace0], layout=layout)

    py.iplot(figure)



categories = df['Job Category'].value_counts().sort_values(ascending=False).head(5)

createhmap(list(categories.index), list(categories.values))
salaries = df.groupby('Civil Service Title')['Salary Range From'].sum().sort_values(ascending=False).head(15)



trace1 = go.Bar(x=salaries.values, 

                y=salaries.index, 

                width=0.6,

                marker=dict(

                    color='#54d1f7', 

                    line=dict(

                        color='#54d1f7', 

                        width=1.5)

                ),

                orientation='h', name='Highest Average Starting Salary')



layout = dict(showlegend=False,

              title='Highest Average Starting Salaries',

              yaxis=dict(

                  showgrid=False,

                  showline=False,

                  showticklabels=True,

              ),

             xaxis=dict(

                  title='Salaries',

                  zeroline=False,

                  showline=False,

                  showticklabels=True,

                  showgrid=False,

             ),

             margin = dict(l=300, r=20, t=50, b=50),

            )

fig = go.Figure(data=[trace1], layout=layout)

py.iplot(fig)
salaries = df.groupby('Civil Service Title')['Salary Range To'].mean().sort_values(ascending=False).head(15)



trace1 = go.Bar(y=salaries.index, x=salaries.values, width=0.6, 

                marker=dict(color='red',

                            opacity=0.6,

                            line=dict(color='red',

                                      width=1.5)

                           ),

               orientation='h', name='Highest Salary Jobs')



layout = dict(showlegend=False,

             title='Highest Salary Jobs',

             yaxis=dict(

                 showgrid=False,

                 showline=False,

                 showticklabels=True,

             ),

             xaxis=dict(

                 title='Salaries',

                 showgrid=False,

                 showline=False,

                 showticklabels=True

             ),

             margin=dict(l=300, r=20, t=50, b=50),

            )



fig = go.Figure(data=[trace1], layout=layout)

py.iplot(fig)
freq = df['Salary Frequency'].value_counts()

py.iplot([pie_plot(freq, colors=['#E1396C', '#96D38C', '#D0F9B1'], title='Salary Frequency')])
py.iplot([pie_plot(df['Full-Time/Part-Time indicator'].value_counts(), ['orange', 'purple'], 'Job Type')])
df['start'] = round(df['Salary Range From'])

df['start'] = df['start'].astype('int')



df['end'] = round(df['Salary Range To'])

df['end'] = df['end'].astype('int')





fig = bubbleplot(df, x_column='start', y_column='end', bubble_column='Agency', size_column='# Of Positions', x_title='Starting Salary', color_column='Agency', 

                 y_title='Salary Range till', title='Finding the right job', x_logscale=False, scale_bubble=3, height=650)



py.iplot(fig)
locations = df['Work Location'].value_counts().sort_values(ascending=False).head(15)



trace1 = go.Bar(x=locations.values, 

                y=locations.index, 

                width=0.6,

                marker=dict(

                    color='#8ddcf4', 

                    line=dict(

                        color='#54d1f7', 

                        width=1.5)

                ),

                orientation='h', name='Job Locations')



layout = dict(showlegend=False,

              title='Most Busy Locations',

              yaxis=dict(

                  showgrid=False,

                  showline=False,

                  showticklabels=True,

              ),

             xaxis=dict(

                  title='Jobs',

                  zeroline=False,

                  showline=False,

                  showticklabels=True,

                  showgrid=False,

             ),

             margin = dict(l=300, r=20, t=50, b=50),

            )

fig = go.Figure(data=[trace1], layout=layout)

py.iplot(fig)
from nltk.corpus import stopwords

plt.style.use('ggplot')

plt.figure(figsize=(15, 15))



plt.subplot(211)

wc = WordCloud(background_color='white', max_words=100, stopwords=stopwords.words('english'))

plt.imshow(wc.generate_from_text(str(df['Minimum Qual Requirements'])), interpolation='bilinear')

title = plt.title('Minimum Qualification', fontsize=20)

plt.setp(title, color='black')

plt.axis("off")



plt.subplot(212)

stopwords = list(stopwords.words('english'))

stopwords.extend(['NaN', 'Candidate'])

wc = WordCloud(background_color='white', max_words=100, stopwords=stopwords, colormap='rainbow')

plt.imshow(wc.generate_from_text(str(df['Preferred Skills'])), interpolation='bilinear')

title = plt.title('Preferred Skills', fontsize=20)

plt.setp(title, color='black')

plt.axis("off")



plt.show()