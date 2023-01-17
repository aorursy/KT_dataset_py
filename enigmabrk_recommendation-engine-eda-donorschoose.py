import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
from geopy.geocoders import Nominatim
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
import squarify
import mpl_toolkits
from numpy import array
from matplotlib import cm
import folium
from folium.plugins import MarkerCluster
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

%matplotlib inline
donations = pd.read_csv('../input/Donations.csv',nrows=10000)
donors = pd.read_csv('../input/Donors.csv', low_memory=False,nrows=10000)
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False,nrows=10000)
teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False,parse_dates=['Teacher First Project Posted Date'],nrows=10000)
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False, parse_dates=["Project Posted Date","Project Fully Funded Date"])
resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False,nrows=10000)
donations.head()
donations['Donation Amount'].describe()
x = donations['Donation Amount']

data = [go.Histogram(x=x)]


layout = go.Layout(
    title='Donation Amount Distribution',
    xaxis=dict(
        title='Value'
    ),
    yaxis=dict(
        title='Count'
    ),
    bargap=0.1,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='styled histogram')
x = donations['Donation Amount']

data = [go.Histogram(x=np.log(x))]


layout = go.Layout(
    title='Log Distribution Of Donation Amount',
    xaxis=dict(
        title='Log Value'
    ),
    yaxis=dict(
        title='Count'
    ),
    bargap=0.1,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='styled histogram')
plt.figure(figsize = (12, 8))
plt.scatter(range(donations.shape[0]), np.sort(donations['Donation Amount'].values))
plt.xlabel('Donation Amount', fontsize=12)
plt.title("Distribution of Donation Amount")
plt.show()
optionl=donations['Donation Included Optional Donation'].value_counts()
tag = (np.array(optionl.index))
sizes = (np.array((optionl / optionl.sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Does Donation include optional Donation ?')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Activity Distribution")
plt.figure(figsize=(12,10))

cart_count=donations['Donor Cart Sequence'].value_counts().head(10)

sns.barplot(cart_count.index,cart_count.values)

plt.xlabel('Cart Number',fontsize=12)
plt.ylabel('Number Of Donors',fontsize=12)
plt.title('Top Checked Cart of Donors',fontsize=18)
plt.show()
tag = (np.array(cart_count.index))
sizes = (np.array((cart_count / cart_count.sum())*100))
plt.figure(figsize=(15,8))


trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Checked Cart Of Donors')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Activity Distribution")
donors.head()
don_city=donors['Donor City'].value_counts().head(50).reset_index()
don_city.columns=['City','Count']
data = [go.Bar(
            x=don_city.City, 
            y=don_city.Count
    )]

layout = go.Layout(
    title='Top Cities Of Donors',
    xaxis=dict(
        title='City Name'
    ),
    yaxis=dict(
        title='Count'
    ),
  
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='styled histogram')
don_state=donors['Donor State'].value_counts().head(50).reset_index()
don_state.columns=['State','Count']
data = [go.Bar(
            x=don_state.State, 
            y=don_state.Count
    )]

layout = go.Layout(
    title='Top State Of Donors',
    xaxis=dict(
        title='State'
    ),
    yaxis=dict(
        title='Count'
    ),
  
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='styled histogram')
x=donors.dropna()
don_state=donors['Donor State'].value_counts().reset_index()
don_state.columns=['State','Count']
state_codes = {
    'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 
    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 
    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 
    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 
    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 
    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 
    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 
    'Nevada': 'NV', 'Maine': 'ME', 'other': ''}

don_state['code'] = don_state['State'].apply(lambda x : state_codes[x])


data = [ dict(
        type='choropleth',
        #colorscale = scl,
        autocolorscale = False,
        locations = don_state['code'],
        z = don_state['Count'],
        locationmode = 'USA-states',
        text = don_state['State'],
        marker = dict(
            line = dict (
                color = 'rgb(125,205,250)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Number Of Donations")
        ) ]

layout = dict(
        title = 'Number Of Donors Per State<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )
teacher=donors['Donor Is Teacher'].value_counts()

tag = (np.array(teacher.index))
sizes = (np.array((teacher / teacher.sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Is Donor a Teacher ??')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
teacher=donors[donors['Donor Is Teacher']=='Yes']
nope=donors[donors['Donor Is Teacher']=='No']
val=teacher['Donor State'].value_counts().reset_index()
val.columns=['City','Count']
val1=nope['Donor State'].value_counts().reset_index()
val1.columns=['City','Count']
trace1 = go.Bar(
    x=val.City,
    y=val.Count ,
    name='Donor Is Teacher'
)
trace2 = go.Bar(
    x=val1.City,
    y=val1.Count ,
    name='Donor Is not a Teacher'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')
projects.head()
prefix=projects['Teacher Project Posted Sequence'].dropna().value_counts().head(10)
tag = (np.array(prefix.index))
sizes = (np.array((prefix / prefix.sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Teacher Project Posted Sequence')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
pro_type= projects["Project Subject Category Tree"].dropna().value_counts().head(30)
data = [go.Bar(
            x=pro_type.index,
            y=pro_type.values, marker=dict(
        color='rgb(118,22,15)',
        line=dict(
            color='rgb(18,48,107)',
            width=1.5,
        )
    )
    )]
layout = go.Layout(
    title='Project Subject Category',
    xaxis=dict(
    
            title='Project Subject'
    )
    
    ,
    yaxis=dict(
        title='Number Of Projects',
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='style-bar')
pro_type= projects["Project Subject Subcategory Tree"].dropna().value_counts().head(10)
data = [go.Bar(
            x=pro_type.index,
            y=pro_type.values, marker=dict(
        color='rgb(118,212,15)',
        line=dict(
            color='rgb(18,48,107)',
            width=1.5,
        )
    )
    )]
layout = go.Layout(
    title='Project Type',
    xaxis=dict(
    
            title='Project Type'
    )
    
    ,
    yaxis=dict(
        title='Number Of Projects',
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='style-bar')
pro_grade=projects['Project Grade Level Category'].value_counts().head(4).reset_index()
pro_grade.columns=['Grade','Count']
fig = {
  "data": [
    {
      "values": pro_grade.Count,
      "labels": pro_grade.Grade.values,
      "domain": {"x": [.2, .78]},
      "name": "GHG Emissions",
      "hoverinfo":"label+percent+name",
      "hole": .5,
      "type": "pie"
    }],
      "layout": {
        "title":"Different Grade Level Of Students",
        "annotations": [
            {
                "font": {
                    "size": 15
                },
                "showarrow": False,
                "text": "Grade Level Of Student",
                "x": 0.49,
                "y": 0.5
            }]}}

py.iplot(fig)
mask =projects['Project Posted Date'] <'2010-10-10'
pro_2010=projects.loc[mask]
pro_2010_val=pro_2010['Project Resource Category'].value_counts().head(3).reset_index()
pro_2010_val.columns=['Resource','Count']
fig = {
  "data": [
    {
      "values": pro_2010_val.Count,
      "labels": pro_2010_val.Resource.values,
      "domain": {"x": [.2, .78]},
      "name": "Project Resource",
      "hoverinfo":"label+percent+name",
      "hole": .5,
      "type": "pie"
    }],
      "layout": {
        "title":"Top Project Resources before 10/10/2010",
        "annotations": [
            {
                "font": {
                    "size": 15
                },
                "showarrow": False,
                "text": "Top Project Resources",
                "x": 0.49,
                "y": 0.5
            }]}}

py.iplot(fig)
mask =projects['Project Posted Date'] >'2010-10-10'
after_2010=projects.loc[mask]
after_2010_val=after_2010['Project Resource Category'].value_counts().head(15).reset_index()
after_2010_val.columns=['Resource','Count']
fig = {
  "data": [
    {
      "values": after_2010_val.Count,
      "labels": after_2010_val.Resource.values,
      "domain": {"x": [.2, .78]},
      "name": "Project Resource",
      "hoverinfo":"label+percent+name",
      "hole": .5,
      "type": "pie"
    }],
      "layout": {
        "title":"Top Project Resources after 10/10/2010",
        "annotations": [
            {
                "font": {
                    "size": 15
                },
                "showarrow": False,
                "text": "Top Project Resources",
                "x": 0.49,
                "y": 0.5
            }]}}

py.iplot(fig)
projects['Posted Year']=projects['Project Posted Date'].dt.year
year_posted=projects['Posted Year'].value_counts().reset_index()
year_posted.columns=['Year','Count']
x_data=year_posted.Year.values
year_2013=projects[projects['Posted Year']==2013]
y0=year_2013['Project Cost'].head(50)
year_2014=projects[projects['Posted Year']==2014]
y1=year_2014['Project Cost'].head(50)
year_2015=projects[projects['Posted Year']==2015]
y2=year_2015['Project Cost'].head(50)
year_2016=projects[projects['Posted Year']==2016]
y3=year_2016['Project Cost'].head(50)
year_2017=projects[projects['Posted Year']==2017]
y4=year_2017['Project Cost'].head(50)
year_2018=projects[projects['Posted Year']==2018]
y5=year_2018['Project Cost'].head(50)

y_data = [y0,y1,y2,y3,y4,y5]

colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']

traces = []

for xd, yd, cls in zip(x_data, y_data, colors):
        traces.append(go.Box(
            y=yd,
            name=xd,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=cls,
            marker=dict(
                size=2,
            ),
            line=dict(width=1),
        ))

layout = go.Layout(
    title='Barplot For Project Cost Of First 50 Projects of Every-Year',
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        dtick=5,
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2,
    ),
    margin=dict(
        l=40,
        r=30,
        b=80,
        t=100,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    showlegend=False
)

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)
plt.figure(figsize=(25,8))
projects['Project Posted Date'].value_counts().plot()
teachers.head()
prefix=teachers['Teacher Prefix'].value_counts()
tag = (np.array(prefix.index))
sizes = (np.array((prefix / prefix.sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Teacher Prefix')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Teacher Distribution")
teachers['Years']=teachers['Teacher First Project Posted Date'].dt.year

year=teachers['Years'].value_counts().reset_index().sort_values(by='index')
year.columns=['Year','Count']
trace = go.Scatter(
    x = year.Year,
    y = year.Count,
    name=year.Year)
layout = go.Layout(
    title='Over The Year Growth in First Project Posted<br>(Hover To Get Exact Number Of Projects) ',
    xaxis=dict(
        title='Year'
    ),
    yaxis=dict(
        title='Number Of Projects'
    ),
  
)
data1=[trace]

fig = go.Figure(data=data1, layout=layout)
py.iplot(fig)
pre=teachers['Teacher Prefix'].value_counts().reset_index()
pre.columns=['Prefix','Count']
x=pre.Prefix
data=[]
for i in x:
    teacher_pre=teachers[teachers['Teacher Prefix']==i]
    year=teacher_pre['Years'].value_counts().reset_index().sort_values(by='index')
    year.columns=['Year','Count']
    trace = go.Scatter(
    x = year.Year,
    y = year.Count,
    name = i)
    data.append(trace)
layout = go.Layout(
    title='Over The Year Growth in First Project Posted Prefix-Wise',
    xaxis=dict(
        title='Year'
    ),
    yaxis=dict(
        title='Number Of Projects'
    ),
  
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
def get_type(x):
    name=x.split(' ')
    return ' '.join(name[-2:])
   

schools['Type']=schools['School Name'].apply(lambda x:get_type(x))
plt.figure(figsize=(12,10))

school_type_count=schools['Type'].value_counts().head(10)

sns.barplot(school_type_count.index,school_type_count.values)

plt.xlabel('School Type',fontsize=12)
plt.xticks(rotation=45)
plt.ylabel('Number Of Schools Of that Type',fontsize=12)
plt.title('Different Types Of School(Top 10)',fontsize=18)
plt.show()
tag = (np.array(school_type_count.index))
sizes = (np.array((school_type_count / school_type_count.sum())*100))
plt.figure(figsize=(15,8))


trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Top Different Types Of School')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Activity Distribution")
plt.figure(figsize=(12,10))

school_metro_count=schools['School Metro Type'].value_counts().head(10)

sns.barplot(school_metro_count.index,school_metro_count.values)

plt.xlabel('School Metro Type',fontsize=12)
plt.xticks(rotation=45)
plt.ylabel('Number Of Schools Of that Type',fontsize=12)
plt.title('Differnt Metro Type Of Schools',fontsize=18)
plt.show()
tag = (np.array(school_metro_count.index))
sizes = (np.array((school_metro_count / school_metro_count.sum())*100))
plt.figure(figsize=(15,8))


trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Different Metro type Of Schools')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Activity Distribution")
x = schools['School Percentage Free Lunch']

data = [go.Histogram(x=np.log(x))]


layout = go.Layout(
    title='Log Distribution Of School Percent Free Lunch',
    xaxis=dict(
        title='Log Value'
    ),
    yaxis=dict(
        title='Count'
    ),
    bargap=0.1,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='styled histogram')
percent_rural=schools[schools['School Metro Type']=='rural']
x0 = percent_rural['School Percentage Free Lunch']

percent_urban=schools[schools['School Metro Type']=='urban']
x1 = percent_urban['School Percentage Free Lunch']

percent_urban=schools[schools['School Metro Type']=='suburban']
x2 = percent_urban['School Percentage Free Lunch']

percent_urban=schools[schools['School Metro Type']=='town']
x3 = percent_urban['School Percentage Free Lunch']




trace0 = go.Histogram(
    x=x0,
    name='Rural'
)
trace1 = go.Histogram(
    x=x1,
    name='Urban'
)
trace2 = go.Histogram(
    x=x2,
    name='Suburban'
)
trace3 = go.Histogram(
    x=x3,
    name='Town'
)

data = [trace0, trace1,trace2,trace3]
layout = go.Layout(barmode='stack')
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='stacked histogram')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate

import warnings; warnings.simplefilter('ignore')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
projects = pd.read_csv('../input/Projects.csv',nrows=10000)
projects.shape
textfeats = ["Project Title","Project Essay"]
for cols in textfeats:
    projects[cols] = projects[cols].astype(str) 
    projects[cols] = projects[cols].astype(str).fillna('') # FILL NA
    projects[cols] = projects[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
 
text = projects["Project Title"] + ' ' + projects["Project Essay"]
vectorizer = TfidfVectorizer(strip_accents='unicode',
                             analyzer='word',
                             lowercase=True, # Convert all uppercase to lowercase
                             stop_words='english', # Remove commonly found english words ('it', 'a', 'the') which do not typically contain much signal
                             max_df = 0.9, # Only consider words that appear in fewer than max_df percent of all documents
                             # max_features=5000 # Maximum features to be extracted                    
                            ) 
project_ids = projects['Project ID'].tolist()
tfidf_matrix = vectorizer.fit_transform(text)
tfidf_feature_names = vectorizer.get_feature_names()
tfidf_matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[0]
projects=projects.reset_index()
titles=projects['Project Title']
indices=pd.Series(projects.index,index=projects['Project Title'])
projects.head()
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]
get_recommendations('learning in color!').head(10)