import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px



df = pd.read_csv('../input/udemy-courses/udemy_courses.csv',parse_dates=['published_timestamp'])
df['year']=df['published_timestamp'].dt.year

df['month']=df['published_timestamp'].dt.month

df['date']=df['published_timestamp'].dt.day
df.shape
df.head(5)
df.dtypes
df.isnull().sum()
df['course_title']=df['course_title'].astype(str)
df['is_paid']=df['is_paid'].astype(int)
#price





plt.figure(figsize=(8,6))

plt.hist('price', data = df, bins = 10,color='maroon')

plt.xlabel('price')

plt.show()

#majority price is in between price 25-60



sns.boxplot(x = 'price', data = df, orient = 'h', width = 0.8, 

                 fliersize = 3, showmeans=True)







plt.figure(figsize=(8,6))

plt.hist('num_subscribers', data = df, bins = 10,color='maroon')

plt.xlabel('num_subscribers')

plt.show()

#majority subscribers are in range 0-50000



sns.boxplot(x = 'num_subscribers', data = df, orient = 'h', width = 0.8, 

                 fliersize = 3, showmeans=True)



plt.figure(figsize=(8,6))

plt.hist('num_reviews', data = df, bins = 10,color='maroon')

plt.xlabel('num_reviews')

plt.show()

#majority reviews are in between 0-2500



sns.boxplot(x = 'num_reviews', data = df, orient = 'h', width = 0.8, 

                 fliersize = 3, showmeans=True)



plt.figure(figsize=(8,6))

plt.hist('content_duration', data = df, bins = 10,color='maroon')

plt.xlabel('content_duration')

plt.show()

#majority content duration are in between 0-10



sns.boxplot(x = 'content_duration', data = df, orient = 'h', width = 0.8, 

                 fliersize = 3, showmeans=True)

           

  

plt.figure(figsize=(8,6))

plt.hist('num_lectures', data = df, bins = 10,color='maroon')

plt.xlabel('num_lectures')

plt.show()

#majority lecture are in between 0-50 



sns.boxplot(x = 'num_lectures', data = df, orient = 'h', width = 0.8, 

                 fliersize = 3, showmeans=True)



fig=px.bar(df,x='price',y='num_subscribers',barmode='group')

fig.show()





fig=px.scatter(df,x='num_reviews',y='num_subscribers')

fig.show()



fig=px.scatter(df,x='num_lectures',y='num_subscribers')

fig.show()

fig=px.scatter(df,x='content_duration',y='num_subscribers')

fig.show()

fig=px.scatter(df,x='price',y='num_subscribers')

fig.show()



fig=px.scatter(df,x='price',y='content_duration')

fig.show()



course = df.loc[:, ['level','num_subscribers']]

course['count'] = course.groupby('level')['num_subscribers'].transform('sum')

course = course.sort_values('count', ascending=False)

course.drop('num_subscribers', axis=1, inplace=True)

course = course.drop_duplicates().reset_index(drop=True)

course = course.head()
course
yearcourse = df.loc[:, ['year','is_paid','num_subscribers']]



yearcourse['count'] = yearcourse.groupby(['year','is_paid'])['num_subscribers'].transform('sum')

#course = course.sort_values(['year','month'], ascending=True)

yearcourse.sort_values('year', inplace=True)

yearcourse.drop('num_subscribers', axis=1, inplace=True)

yearcourse = yearcourse.drop_duplicates().reset_index(drop=True)

print(course)

fig=px.bar(yearcourse,x='year',y='count',color='is_paid',barmode="group")

fig.show()
ucourses = df.loc[:, ['year','num_subscribers']]

ucourses['count'] = ucourses.groupby('year')['num_subscribers'].transform('sum')

#course = course.sort_values('count', ascending=True)

ucourses.drop('num_subscribers', axis=1, inplace=True)

ucourses = ucourses.drop_duplicates().reset_index(drop=True)

ucourses.sort_values('year', inplace=True)

ucourses

ucourses['Subscribercount'] = ucourses['count'].cumsum()

from plotly.subplots import make_subplots

import plotly.graph_objects as go



fig = make_subplots(

    rows=1, 

    cols=1,

    subplot_titles=("Time series plot of number of subscribers",

                    ))



fig.append_trace(go.Scatter(

    x=ucourses['year'],

    y=ucourses['Subscribercount'],

    name="All",

    mode='lines'),

    row=1, col=1)

fig.show()





course = df.loc[:, ['year','month','subject','num_subscribers']]

course['count'] = course.groupby(['year','month','subject'])['num_subscribers'].transform('sum')

#course = course.sort_values('count', ascending=True)

#course.drop('num_subscribers', axis=1, inplace=True)

course = course.drop_duplicates().reset_index(drop=True)

course.sort_values(['year','month'], inplace=True)



course2011=course.loc[course.year==2011]

course2012=course.loc[course.year==2012]

course2012Music=course2012.loc[course.subject=='Musical Instruments']

course2012Web=course2012.loc[course.subject=='Web Development']

course2012Design=course2012.loc[course.subject=='Graphic Design']

course2012Fin=course2012.loc[course.subject=='Business Finance']





course2013=course.loc[course.year==2013]

course2013Music=course2013.loc[course.subject=='Musical Instruments']

course2013Web=course2013.loc[course.subject=='Web Development']

course2013Design=course2013.loc[course.subject=='Graphic Design']

course2013Fin=course2013.loc[course.subject=='Business Finance']



course2014=course.loc[course.year==2014]

course2014Music=course2014.loc[course.subject=='Musical Instruments']

course2014Web=course2014.loc[course.subject=='Web Development']

course2014Design=course2014.loc[course.subject=='Graphic Design']

course2014Fin=course2014.loc[course.subject=='Business Finance']



course2015=course.loc[course.year==2015]

course2015Music=course2015.loc[course.subject=='Musical Instruments']

course2015Web=course2015.loc[course.subject=='Web Development']

course2015Design=course2015.loc[course.subject=='Graphic Design']

course2015Fin=course2015.loc[course.subject=='Business Finance']



course2016=course.loc[course.year==2016]

course2016Music=course2016.loc[course.subject=='Musical Instruments']

course2016Web=course2016.loc[course.subject=='Web Development']

course2016Design=course2016.loc[course.subject=='Graphic Design']

course2016Fin=course2016.loc[course.subject=='Business Finance']





course2017=course.loc[course.year==2017]

course2017Music=course2017.loc[course.subject=='Musical Instruments']

course2017Web=course2017.loc[course.subject=='Web Development']

course2017Design=course2017.loc[course.subject=='Graphic Design']

course2017Fin=course2017.loc[course.subject=='Business Finance']



from plotly.subplots import make_subplots

import plotly.graph_objects as go



fig1 = make_subplots(

    rows=1, 

    cols=1,

    subplot_titles=("Month Wise for the year 2012",))





fig1.append_trace(go.Scatter(

    x=course2012Music['month'], 

    y=course2012Music['num_subscribers'], 

    mode="lines+markers",

    name="Music",

    line=dict(color="#74597D", dash="longdashdot"),

),

    row=1, col=1)

fig1.append_trace(go.Scatter(

    x=course2012Web['month'], 

    y=course2012Web['num_subscribers'] , 

    

    mode="lines+markers",

    name="Web",

    line=dict(color="#C85A17", dash="dash")

),

    row=1, col=1)

fig1.append_trace(go.Scatter(

    x=course2012Design['month'], 

    y=course2012Design['num_subscribers'], 

    

    mode="lines+markers",

    name="Design",

  

    line=dict(color="#1884C7", dash="dashdot")

),

    row=1, col=1)



fig1.append_trace(go.Scatter(

    x=course2012Fin['month'], 

    y=course2012Fin['num_subscribers'], 

    

    mode="lines+markers",

    name="Finance",

  

    line=dict(color="#617C58", dash="dashdot")

),

    row=1, col=1)



fig1.update_layout(width=700, height=600)

fig1.show()



fig2 = make_subplots(

    rows=1, 

    cols=1,

    subplot_titles=("Month Wise for the year 2013",))

fig2.append_trace(go.Scatter(

    x=course2013Music['month'], 

    y=course2013Music['num_subscribers'], 

    mode="lines+markers",

    name="Music",

    line=dict(color="#74597D", dash="longdashdot"),

),

    row=1, col=1)

fig2.append_trace(go.Scatter(

    x=course2013Web['month'], 

    y=course2013Web['num_subscribers'] , 

    

    mode="lines+markers",

    name="Web",

    line=dict(color="#C85A17", dash="dash")

),

    row=1, col=1)

fig2.append_trace(go.Scatter(

    x=course2013Design['month'], 

    y=course2013Design['num_subscribers'], 

    

    mode="lines+markers",

    name="Design",

  

    line=dict(color="#1884C7", dash="dashdot")

),

    row=1, col=1)

fig2.append_trace(go.Scatter(

    x=course2013Fin['month'], 

    y=course2013Fin['num_subscribers'], 

    

    mode="lines+markers",

    name="Finance",

  

    line=dict(color="#617C58", dash="dashdot")

),

    row=1, col=1)

fig2.update_layout(width=700, height=600)

fig2.show()





fig3 = make_subplots(

    rows=1, 

    cols=1,

    subplot_titles=("Month Wise for the year 2014",))

fig3.append_trace(go.Scatter(

    x=course2014Music['month'], 

    y=course2014Music['num_subscribers'], 

    mode="lines+markers",

    name="Music",

    line=dict(color="#74597D", dash="longdashdot"),

),

    row=1, col=1)

fig3.append_trace(go.Scatter(

    x=course2014Web['month'], 

    y=course2014Web['num_subscribers'] , 

    

    mode="lines+markers",

    name="Web",

    line=dict(color="#C85A17", dash="dash")

),

    row=1, col=1)

fig3.append_trace(go.Scatter(

    x=course2014Design['month'], 

    y=course2014Design['num_subscribers'], 

    

    mode="lines+markers",

    name="Design",

  

    line=dict(color="#1884C7", dash="dashdot")

),

    row=1, col=1)

fig3.append_trace(go.Scatter(

    x=course2014Fin['month'], 

    y=course2014Fin['num_subscribers'], 

    

    mode="lines+markers",

    name="Finance",

  

    line=dict(color="#617C58", dash="dashdot")

),

    row=1, col=1)

fig3.update_layout(width=700, height=600)

fig3.show()



fig4 = make_subplots(

    rows=1, 

    cols=1,

    subplot_titles=("Month Wise for the year 2015",))

fig4.append_trace(go.Scatter(

    x=course2015Music['month'], 

    y=course2015Music['num_subscribers'], 

    mode="lines+markers",

    name="Music",

    line=dict(color="#74597D", dash="longdashdot"),

),

    row=1, col=1)

fig4.append_trace(go.Scatter(

    x=course2015Web['month'], 

    y=course2015Web['num_subscribers'] , 

    

    mode="lines+markers",

    name="Web",

    line=dict(color="#C85A17", dash="dash")

),

    row=1, col=1)

fig4.append_trace(go.Scatter(

    x=course2015Design['month'], 

    y=course2015Design['num_subscribers'], 

    

    mode="lines+markers",

    name="Design",

  

    line=dict(color="#1884C7", dash="dashdot")

),

    row=1, col=1)

fig4.append_trace(go.Scatter(

    x=course2015Fin['month'], 

    y=course2015Fin['num_subscribers'], 

    

    mode="lines+markers",

    name="Finance",

  

    line=dict(color="#617C58", dash="dashdot")

),

    row=1, col=1)

fig4.update_layout(width=700, height=600)

fig4.show()











fig5 = make_subplots(

    rows=1, 

    cols=1,

    subplot_titles=("Month Wise for the year 2016",))

fig5.append_trace(go.Scatter(

    x=course2016Music['month'], 

    y=course2016Music['num_subscribers'], 

    mode="lines+markers",

    name="Music",

    line=dict(color="#74597D", dash="longdashdot"),

),

    row=1, col=1)

fig5.append_trace(go.Scatter(

    x=course2016Web['month'], 

    y=course2016Web['num_subscribers'] , 

    

    mode="lines+markers",

    name="Web",

    line=dict(color="#C85A17", dash="dash")

),

    row=1, col=1)

fig5.append_trace(go.Scatter(

    x=course2016Design['month'], 

    y=course2016Design['num_subscribers'], 

    

    mode="lines+markers",

    name="Design",

  

    line=dict(color="#1884C7", dash="dashdot")

),

    row=1, col=1)

fig5.append_trace(go.Scatter(

    x=course2016Fin['month'], 

    y=course2016Fin['num_subscribers'], 

    

    mode="lines+markers",

    name="Finance",

  

    line=dict(color="#617C58", dash="dashdot")

),

    row=1, col=1)

fig5.update_layout(width=700, height=600)

fig5.show()



fig6 = make_subplots(

    rows=1, 

    cols=1,

    subplot_titles=("Month Wise for the year 2017",))

fig6.append_trace(go.Scatter(

    x=course2017Music['month'], 

    y=course2017Music['num_subscribers'], 

    mode="lines+markers",

    name="Music",

    line=dict(color="#74597D", dash="longdashdot"),

),

    row=1, col=1)

fig6.append_trace(go.Scatter(

    x=course2017Web['month'], 

    y=course2017Web['num_subscribers'] , 

    

    mode="lines+markers",

    name="Web",

    line=dict(color="#C85A17", dash="dash")

),

    row=1, col=1)

fig6.append_trace(go.Scatter(

    x=course2017Design['month'], 

    y=course2017Design['num_subscribers'], 

    

    mode="lines+markers",

    name="Design",

  

    line=dict(color="#1884C7", dash="dashdot")

),

    row=1, col=1)

fig6.append_trace(go.Scatter(

    x=course2017Fin['month'], 

    y=course2017Fin['num_subscribers'], 

    

    mode="lines+markers",

    name="Finance",

  

    line=dict(color="#617C58", dash="dashdot")

),

    row=1, col=1)

fig6.update_layout(width=700, height=600)

fig6.show()











course = df.loc[:, [ 'subject','num_subscribers']]



course['count'] = course.groupby('subject')['num_subscribers'].transform('sum')

course = course.sort_values('count', ascending=True)

course.drop('num_subscribers', axis=1, inplace=True)

course = course.drop_duplicates().reset_index(drop=True)

#course['Subscribercount'] = course['count'].cumsum()
fig = px.pie(course, names='subject', values='count', width=500)

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
course = df.loc[:, [ 'subject','is_paid','num_subscribers']]



course['count'] = course.groupby(['subject','is_paid'])['num_subscribers'].transform('sum')

course.drop('num_subscribers', axis=1, inplace=True)

course = course.drop_duplicates().reset_index(drop=True)

course