#!pip install bubbly
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

from __future__ import division

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib.pyplot as plt

import seaborn as sns





import plotly

import plotly.express as px

import plotly.graph_objects as go



from plotly.offline import init_notebook_mode,iplot



#from bubbly.bubbly import bubbleplot





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/udemy-courses/udemy_courses.csv')
def aboutdata(dataframe):

    df=dataframe

    print('About datatypes of columns and memory usage:')

    df.info()

    print('Shape of data frame:')

    print(df.shape)

    for col in df.columns:

        print("Unique number of values in ")

        print(col)

        print(df.loc[:,col].nunique())

    print("number of null values present in each column")

    print(df.isnull().sum())

    print(df.head(5))

aboutdata(df)
df['published_timestamp']=pd.to_datetime(df['published_timestamp'])
df['date']=df['published_timestamp'].dt.date
df['month']=df['published_timestamp'].dt.month
df['year']=df['published_timestamp'].dt.year
df=df.drop(['published_timestamp'],axis=1)
df1=df['subject'].value_counts().reset_index()

fig=go.Figure(data=[go.Pie(labels=df1['index'],

                          values=df1['subject'],

                          hole=.4,

                          title="Share of each Course")])

fig.update_layout(title="Udemy Course Subjects")

fig.show()
feature=['course_title','price','is_paid','subject']

df[feature].loc[df[feature].price > 0].sort_values(by=['price'],ascending=False).head(10)
a=df['price'].value_counts().reset_index()

fig=go.Figure(data=[go.Pie(labels=a['price'],

                          values=a['index'],

                          hole=.4,

                          title="Number of courses grouped by price")])

fig.update_layout(title="Udemy Course Prices")

fig.show()
df2=df[feature].loc[df[feature].price > 0].sort_values(by=['price'],ascending=True).head(15)

fig=px.bar(df2,

                           y='price',

                           x='course_title',

                           )

fig.update_layout(title="Udemy Course Subjects")

fig.show()
feature1=['num_reviews','price','subject']
df3=df[feature1].loc[df[feature1].price > 0].sort_values(by=['num_reviews'],ascending=False)

fig=px.scatter(df3,

                           y='price',

                           x='num_reviews',

                           color='subject',

                           animation_frame='num_reviews',

                           animation_group='price',

                           range_x =[0,27445],

                           range_y=[2011,2018],

                          

                           )



fig.update_layout(title="Free courses published in the year",paper_bgcolor='rgb(180,180,180)')

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 200

fig.show();

df3=df[feature1].loc[df[feature1].price == 0].sort_values(by=['num_reviews'],ascending=False)

fig=px.scatter(df3,

                           y='price',

                           x='num_reviews',

                           size='num_reviews',color='subject'

                           )

fig.update_layout(title="Number of reviews for the free courses")

fig.show()
df3=df[feature1].loc[df[feature1].price > 0].sort_values(by=['num_reviews'],ascending=False)

fig=px.scatter(df3,

                           y='subject',

                           x='num_reviews',

                           color='price',size='num_reviews'

                           )

fig.update_layout(title="Subject wise number of reviews for paid courses")

fig.show()
df3=df[feature1].loc[df[feature1].price == 0].sort_values(by=['num_reviews'],ascending=False)

fig=px.scatter(df3,

                           y='subject',

                           x='num_reviews',

                           size='num_reviews'

                           )

fig.update_layout(title="Subject wise number of reviews for free courses")

fig.show()
feature2=['num_subscribers','price','subject']
c=df[feature2].loc[df[feature2].price == 0].sort_values(by=['num_subscribers'],ascending=False)
fig=px.scatter(c,

                           y='price',

                           x='num_subscribers',

                           size="num_subscribers"

                           )

fig.update_layout(title="Udemy Course reviews")

fig.show()
d=df[feature2].loc[df[feature2].price > 0].sort_values(by=['num_subscribers'],ascending=False)
fig=px.scatter(d,

                           y='price',

                           x='num_subscribers',

                           size="num_subscribers",color='subject'

                           )

fig.update_layout(title="Udemy Course subscribers for paid courses")

fig.show()
df[feature1].loc[df[feature1].price > 0].sort_values(by=['num_reviews'],ascending=False)
feature2=['price','level','subject','course_title']
e=df[feature2].loc[df[feature2].price > 0].sort_values(by=['price'],ascending=False)
fig=px.scatter(e,

                           y='price',

                           x='level',

                           size="price",color='subject'

                           )

fig.update_layout(title="Udemy Course level for paid courses")

fig.show()
f=df[feature2].loc[df[feature2].price ==0].sort_values(by=['price'],ascending=False)
fig=px.scatter(f,

                           y='subject',

                           x='level',

                           size="price",color='subject'

                           )

fig.update_layout(title="Udemy Course level for free courses")

fig.show()
feature3=['num_lectures','price','course_title','subject']

g=df[feature3].loc[df[feature3].price == 0]

fig=px.scatter(g,

                           y='price',

                           x='num_lectures',

                           size="num_lectures",color='subject'

                           )

fig.update_layout(title="number of lecutures for free courses")

fig.show()



h=df[feature3].loc[df[feature3].price > 0].sort_values(by=['price'],ascending=False)

fig=px.scatter(h,

                           y='price',

                           x='num_lectures',

                           size="num_lectures",color='subject'

                           )

fig.update_layout(title="number of lecutures for paid courses")

fig.show()

feature4=['year','price','subject','course_title']

df6=df[feature4].loc[df[feature4].price > 0].sort_values(by=['year'],ascending=False)




g = [go.Scatter(

          y=df.year,

          x=df['course_title'],

          line=go.scatter.Line(color='green', width = 0.9),

          opacity=0.9)]

layout = go.Layout(height=600,width=800,title='Paid courses published in the year')



fig = go.Figure(data=g,layout=layout)

iplot(fig)



feature4=['year','price','subject','course_title']

i=df[feature4].loc[df[feature4].price == 0].sort_values(by=['year'],ascending=False)
g = [go.Scatter(

          y=i.year,

          x=i['course_title'],

          line=go.scatter.Line(color='green', width = 0.9),

          opacity=0.9)]

layout = go.Layout(height=600,width=800,title='Free courses published in the year')



fig = go.Figure(data=g,layout=layout)

iplot(fig)

feature5=['content_duration','price','subject','course_title']

l=df[feature5].loc[df[feature5].price == 0].sort_values(by=['content_duration'],ascending=False)
fig=px.scatter(l,

                           y='price',

                           x='course_title',

                           size='content_duration',color='subject'

                           )

fig.update_layout(title="content duration for free courses")

fig.show()

k=df[feature5].loc[df[feature5].price > 0].sort_values(by=['content_duration','price'],ascending=False)
fig=px.scatter(k,

                           y='price',

                           x='course_title',

                           size='content_duration',color='subject'

                           )

fig.update_layout(title="content duration for paid courses")

fig.show()

feature6=['year','course_title','num_subscribers','subject','month','price','course_id']

o=df[feature6].loc[df[feature6].price > 0].sort_values(by=['year','course_id'])

fig=px.scatter(o,

                           x='course_id',

                           y='year',

                           size='num_subscribers',color='month',

                           animation_frame='course_title',

                           animation_group='year',

                           range_x =[8324,1282064],

                           range_y=[2011,2018]

                           

                           )

fig.update_layout(title="Paid courses published in the year",paper_bgcolor='rgb(233,233,233)')

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 200

fig.show();

feature7=['year','course_title','num_subscribers','subject','month','price','course_id']

p=df[feature7].loc[df[feature7].price == 0].sort_values(by=['year','course_id'])





fig=px.scatter(p,

                           x='course_id',

                           y='year',

                           size='num_subscribers',

                           size_max=50,

                           color='subject',

                           animation_frame='course_title',

                           animation_group='year',

                           range_x =[17349,1268616],

                           range_y=[2011,2018],

                           )

fig.update_layout(title="Free courses published in the year",paper_bgcolor='rgb(180,180,180)')

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 200

fig.show();
