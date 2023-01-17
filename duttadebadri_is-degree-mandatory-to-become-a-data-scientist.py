# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
color = sns.color_palette()
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import plotly.tools as tls

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df1=pd.read_csv('../input/SurveySchema.csv')
df1.head()
df2=pd.read_csv('../input/multipleChoiceResponses.csv')
df2.head()
df3=pd.read_csv('../input/freeFormResponses.csv')
df3.head()
country=df2[1:]['Q3'].value_counts()
country.sort_index(inplace=True)

py.offline.init_notebook_mode(connected=True)

country=pd.DataFrame(country)
country['country']=country.index
count = [ dict(
        type = 'choropleth',
        locations = country['country'],
        locationmode='country names',
        z = country['Q3'],
        text = country['country'],
        colorscale = 'Viridis',
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick =False,
            title = 'Number of Respondents(In Thousands)'),
      ) ]
layout = dict(
    title = 'Number of Respondent from across the Globe',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
fig = dict( data=count, layout=layout )
py.offline.iplot( fig, validate=False, filename='d3-world-map' )
#reference taken from subham lekhwar's kernel
cnt_srs = df2['Q4'][1:].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Highest Qualification of Data Scientists'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="ds1")

print("We see that",round((df2['Q4'].value_counts()[0]/df2['Q4'][1:].dropna().shape[0])*100,2),"% Data Scientists have a master's degree,\
",round((df2['Q4'].value_counts()[1]/df2['Q4'][1:].dropna().shape[0])*100,2),"% have bachelors degree and",round((df2['Q4'].value_counts()[2]/df2['Q4'][1:].dropna().shape[0])*100,2)\
    ,"% having a doctorate degree" )
cnt_srs = df2['Q2'][1:].value_counts().head()
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1]
    ),
)

layout = dict(
    title='Age distribution of Data Scientists',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="ds2")

trace2 = go.Bar(
    x=df2['Q3'][1:],
    y=df2['Q3'][1:].value_counts(),
    name='Countries',
    marker=dict(
        color='rgb(26, 118, 255)'
    )
)
data = [trace2]
layout = go.Layout(
    title='Top Countries where Data Science is a trend!',
    xaxis=dict(
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='No. of people',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='style-bar')
ax=pd.DataFrame(df2['Q5'][1:].value_counts()).plot.bar(figsize=(12,8))
ax.set_ylabel('No. of people')
ax.set_title('Undergraduate Courses')
ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
col1 = 'Q4'   
col2 = 'Q3' 
data = []
for i in df2[1:][col1].unique():
    country = df2.loc[df2[col1] == i, col2].value_counts().sort_index().index
    valu_country = df2.loc[df2[col1] == i, col2].value_counts().sort_index().values
    size = []  
    for j in country :
        size.append(df2.loc[df2[col2] == j, col1].dropna().size)
    z = (valu_country/size)*100
    trace = go.Bar(
            x=country,
            y=z,
            name=i,            
            )
    data.append(trace)
    layout = go.Layout(width=900, height=600, 
                       barmode='stack',
                       legend=dict(x=0,y=2))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
y = df2['Q6'][1:].value_counts()
x=y.index

trace0 = go.Bar(
    x=x,
    y=y,
    text=x,
    marker=dict(
        color='rgb(210,118,118)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),
    opacity=0.6
)

data = [trace0]
layout = go.Layout(
    title='Current Occupation of Respondants',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='occ')
import re
from matplotlib.ticker import FuncFormatter

df2=df2[1:]
df2 = df2[pd.notnull(df2['Q9'])]
df2 = df2[~df2.Q9.isin(['I do not wish to disclose my approximate yearly compensation','500,000+'])]
df2 = df2[df2.Q6 != 'Student']
df2 = df2[df2.Q7 != 'I am a student']
df2['Q3'] = df2['Q3'].str.replace('(\(|,| or | and | of ).+$','', regex=True)
df2['Q4'] = df2['Q4'].str.replace('(/).+$','', regex=True)
df2['Q5'] = df2['Q5'].str.replace('(\(|,| or | and | of ).+$','', regex=True)

def extract_avg_pay(compensation):
    result = re.split('-|,',compensation)
    return 1000*(int(result[0]) + int(result[1]))/2
    
df2['pay'] = df2['Q9'].apply(extract_avg_pay)
mycolor=['#34495e','#9b59b6']
sns.set()
sns.color_palette(mycolor)
sns.set(font_scale=1)
px = df2[df2.Q1.isin(['Male','Female'])].groupby(['Q4','Q1'])['pay'].mean().unstack().plot(kind="bar", figsize=(16,7))
px.set(xlabel='Educational Degree', ylabel='Average Annual Income (USD)')
px.legend().set_title('Gender')
px.set_xticklabels(px.get_xticklabels(), rotation=30)
comma_fmt = FuncFormatter(lambda x, p: format(int(x), ','))
px.yaxis.set_major_formatter(comma_fmt)

## I have taken reference for this graph from Doha's kernel
fig = {
  "data": [
    {
      "values": df2['Q26'].value_counts(),
      "labels": df2['Q26'].unique(),
        'marker': {'colors': ['rgb(58, 21, 56)',
                                  'rgb(33, 180, 150)']},
      "name": "Confident Data Scientists",
      "hoverinfo":"label+percent+name",
      "hole": .5,
      "type": "pie"
    }],
     "layout": {
        "title":"How many respondants consider themseleves as Data Scientists?"
     }
}
py.iplot(fig, filename='donut')
plt.figure(figsize=(10,8))
ax=sns.countplot(df2['Q12_MULTIPLE_CHOICE'].dropna())
ax.set_xticklabels(labels=df2['Q12_MULTIPLE_CHOICE'].dropna().unique(),rotation=90)
ax.set_ylabel(ylabel='No. of Users',fontsize=17)
ax.axes.set_title('Most used primary tool at work',fontsize=17)
ax.tick_params(labelsize=13)
df2.fillna('',inplace=True)
df2['important_roles']=df2['Q11_Part_1']+df2['Q11_Part_2']+df2['Q11_Part_3']+df2['Q11_Part_4']+df2['Q11_Part_5']+df2['Q11_Part_6']+df2['Q11_Part_7']
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(df2['important_roles'].dropna())
df2=pd.read_csv('../input/multipleChoiceResponses.csv')
df2 = df2[1:]

df2 = df2[~pd.isnull(df2['Q35_Part_1'])]

count_dict = {
    'Self-taught & Online Courses' : (df2['Q35_Part_1'].astype(float)>0).sum()+(df2['Q35_Part_2'].astype(float)>0).sum(), #considering online courses as self-taught
    'Work' : (df2['Q35_Part_3'].astype(float)>0).sum(),
    'University' : (df2['Q35_Part_4'].astype(float)>0).sum(),
    'Kaggle competitions' : (df2['Q35_Part_5'].astype(float)>0).sum(),
    'Other' : (df2['Q35_Part_6'].astype(float)>0).sum()
}

cnt_srs = pd.Series(count_dict)

trace = go.Pie(labels=np.array(cnt_srs.index),values=np.array(cnt_srs.values))

layout = go.Layout(
    title='How the respondants learnt Data Science?',
    height=750,
    width=750
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="learningcategory")


count_dict = {
    'Coursera' : (df2['Q36_Part_2'].count()),
    'Udemy' : (df2['Q36_Part_9'].count()),
    'DataCamp' : (df2['Q36_Part_4'].count()),
    'Kaggle Learn' : (df2['Q36_Part_6'].count()),
    'Udacity' : (df2['Q36_Part_1'].count()),
    'edX' : (df2['Q36_Part_3'].count()),
    'Online University Courses' : (df2['Q36_Part_11'].count()),
    'Fast.AI' : (df2['Q36_Part_7'].count()),
    'Developers.google.com' : (df2['Q36_Part_8'].count()),
    'DataQuest' : (df2['Q36_Part_5'].count()),
    'The School of AI' : (df2['Q36_Part_10'].count())
}


cnt_srs = pd.Series(count_dict)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color='gray',
    ),
)

layout = go.Layout(
    title='Online Platforms with max. users'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="platform")
