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
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False, parse_dates=["Project Posted Date","Project Fully Funded Date"],nrows=10000)
resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False,nrows=10000)
donations.head()
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
    name=ye)
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