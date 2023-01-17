!pip install joypy -q

import joypy

import numpy as np # linear algebra

import plotly.express  as px

import plotly.graph_objects as go

fig = go.Figure()



import matplotlib.pyplot as plt

from matplotlib import cm

plt.style.use('ggplot')

import seaborn as sns

import plotly.io as pio

pio.templates.default = "plotly_dark"

sns.set_style('darkgrid')

%matplotlib inline



import cufflinks as cf

import plotly.offline

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)



import datetime as dt



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/udemy-courses/udemy_courses.csv')

data.sample(5).reset_index(drop=True).style.set_properties(**{'background-color': '#161717','color': '#30c7e6','border-color': '#8b8c8c'})
data[data['num_lectures']==0]
## Removing Undesired Rows

data.drop([892], inplace = True)
Subject = pd.DataFrame(data['subject'].value_counts()).reset_index(drop = False)

fig = px.pie(Subject, values =Subject.subject, names = Subject['index'],

             title='Distribution of Various Courses on Udemy!')

fig.show()
#changing the 'published_timestamp' to correct Datatye

data['published_timestamp'] = pd.to_datetime(data['published_timestamp'])

data['year'] = data['published_timestamp'].dt.year



Year_wise = data.groupby('year')['course_id'].count().sort_values().reset_index()

Year_wise.rename({'course_id':'Number of Courses'},axis = 1, inplace = True)

fig = px.bar(Year_wise, y = 'Number of Courses', x = 'year', color = 'year')

fig.show()
#plt.figure(figsize = (12,10))



fig = px.box(data,

       x='content_duration',

       y='is_paid',

       orientation='h',

       color='is_paid',

       title='Duration Distribution Across Type of Course',

       color_discrete_sequence=['#03cffc','#eb03fc']

      )



fig.update_layout(showlegend=False)

fig.update_xaxes(title='Content Duration')

fig.update_yaxes(title='Paid Course')

fig.show()
fig = px.box(data,     

       x='content_duration',

       y='subject',

       orientation='h',

       color='is_paid',

       title='Duration Distribution Across Subject and Type of Course',

       color_discrete_sequence=['#03cffc','#eb03fc']

      )





fig.update_xaxes(title='Content Duration')

fig.update_yaxes(title='Course Subject')

fig.show()
fig = px.box(data,

      x = 'subject',

      y = 'price',

      hover_name = 'course_title',

      color = 'subject',

      title = 'Course Prices x Subject'

)

fig.show()
# Ridgeline Plot

fig = joypy.joyplot(data,

                    by      = 'subject',

                    column  = 'price',

                    figsize = (16,12),

                    grid    = 'both',

                    linewidth = 3,

                    colormap  = cm.winter,

                    fade      = True,

                    title     = 'Price Distribution Across Subjects',

                    overlap   = 2

                   )

plt.show()
top25_paid = data.sort_values("num_subscribers", ascending=False)[0:25].sort_values("num_subscribers", ascending=True).reset_index(drop=True).reset_index(drop =True)

fig = px.bar(top25_paid,

       y = 'course_title',

       x= 'num_subscribers',

       orientation = 'h',

       color='num_subscribers',

      hover_data=['is_paid','num_reviews','num_lectures'])





fig.update_layout(showlegend=False)

fig.update_xaxes(title='Number of Subscribers')

fig.update_yaxes(title='Course Title',showticklabels=False)

fig.show()
Unpaid = data[data['is_paid']==False]



top25_free = Unpaid.sort_values("num_subscribers", 

                                ascending=False)[0:25].sort_values("num_subscribers", ascending=True).reset_index(drop=True).reset_index()

fig = px.bar(top25_free,

       y = 'course_title',

       x= 'num_subscribers',

       orientation = 'h',

       color='num_subscribers',

      hover_data=['num_reviews','num_lectures','year'])





fig.update_layout(showlegend=False)

fig.update_xaxes(title='Number of Subscribers')

fig.update_yaxes(title='Course Title',showticklabels=False)

fig.show()
Web = data[data['subject']=='Web Development']

top_web = Web.sort_values("num_subscribers", 

                                ascending=False)[0:10].sort_values("num_subscribers", ascending=True).reset_index(drop=True).reset_index()



fig = px.bar(top_web,

       y = 'course_title',

       x= 'num_subscribers',

       orientation = 'h',

       color='num_subscribers',

      hover_data=['num_reviews','num_lectures','year','url'])





fig.update_layout(showlegend=False)

fig.update_xaxes(title='Number of Subscribers')

fig.update_yaxes(title='Course Title',showticklabels=False)

fig.show()
Bus = data[data['subject']=='Business Finance']



top_web = Bus.sort_values("num_subscribers", 

                                ascending=False)[0:10].sort_values("num_subscribers", ascending=True).reset_index(drop=True).reset_index()



fig = px.bar(top_web,

       y = 'course_title',

       x= 'num_subscribers',

       orientation = 'h',

       color='num_subscribers',

      hover_data=['num_reviews','num_lectures','year','url'])





fig.update_layout(showlegend=False)

fig.update_xaxes(title='Number of Subscribers')

fig.update_yaxes(title='Course Title',showticklabels=False)

fig.show()
Graphic = data[data['subject']=='Graphic Design']



top_graph = Graphic.sort_values("num_subscribers", 

                                ascending=False)[0:10].sort_values("num_subscribers", ascending=True).reset_index(drop=True).reset_index()



fig = px.bar(top_graph,

       y = 'course_title',

       x= 'num_subscribers',

       orientation = 'h',

       color='num_subscribers',

      hover_data=['num_reviews','num_lectures','year','url'])





fig.update_layout(showlegend=False)

fig.update_xaxes(title='Number of Subscribers')

fig.update_yaxes(title='Course Title',showticklabels=False)

fig.show()
Music = data[data['subject']=='Musical Instruments']



top_music= Music.sort_values("num_subscribers", 

                                ascending=False)[0:10].sort_values("num_subscribers", ascending=True).reset_index(drop=True).reset_index()



fig = px.bar(top_music,

       y = 'course_title',

       x= 'num_subscribers',

       orientation = 'h',

       color='num_subscribers',

      hover_data=['num_reviews','num_lectures','year','url'])





fig.update_layout(showlegend=False)

fig.update_xaxes(title='Number of Subscribers')

fig.update_yaxes(title='Course Title',showticklabels=False)

fig.show()
plt.figure(figsize = (10,7))

f = data[['num_reviews','price','num_subscribers','content_duration']].corr()

sns.heatmap(f, annot=True)
fig = px.scatter(data,x = data['price'], y = data['content_duration'],

           hover_data = ['course_title'],color=data["subject"])



fig.update_xaxes(title='Price')

fig.update_yaxes(title='Content Duration',showticklabels=False)

fig.show()
fig = px.scatter(data,x = data['num_reviews'], y = data['num_subscribers'],

           hover_data = ['course_title'],color=data["subject"])



fig.update_xaxes(title='Number of Reviews')

fig.update_yaxes(title='Number of Subscribers',showticklabels=False)

fig.show()
fig = px.scatter(data, x = data['price'], y = data['num_subscribers'],

                 hover_data = ['course_title'],

              color=data["subject"])



fig.update_xaxes(title='Price of a Course')

fig.update_yaxes(title='Number of Subscribers',showticklabels=False)

fig.show()