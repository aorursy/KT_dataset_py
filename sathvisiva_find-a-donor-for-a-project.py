# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# linear algebra
import numpy as np 

# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd 

#visualization
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected = True)
from wordcloud import WordCloud, STOPWORDS

from scipy.stats import kurtosis
from scipy.stats import skew
stopwords = set(STOPWORDS)
from textblob import TextBlob



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
resource = pd.read_csv('../input/Resources.csv')
schools = pd.read_csv('../input/Schools.csv')
donors = pd.read_csv('../input/Donors.csv')
donations = pd.read_csv('../input/Donations.csv')
teachers = pd.read_csv('../input/Teachers.csv')
projects = pd.read_csv('../input/Projects.csv')
#function for displaying top 20 values using plotly bar plot

def vert_bar_plot(df, col, title):
    trace = go.Bar(
    x = df[col].value_counts()[:20].index,
    y = df[col].value_counts()[:20].values,
    text = df[col].value_counts()[:20].values,
    textposition = 'auto',
    marker = dict(
        color = 'rgb(153,153,255)',
        line = dict(
            color = 'rgb(8,48,107)',
            width = 1.5            
        ),
    ),
    opacity = 0.6  
    )
    layout = dict(
    title=title,
    )
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
#state codes list which will be used for displaying state stats in maps
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
resource.head()
schools.head()
donors.head()
donations.head()
teachers.head()
projects.head()
msno.matrix(resource)
plt.show()
print("Missing datas\n" , pd.isnull(resource).sum())
vert_bar_plot(resource, 'Resource Vendor Name', 'Top 20 Resource Vendors')
plt.figure(figsize = (12,8))
plt.scatter(range(resource.shape[0]), np.sort(resource['Resource Unit Price'].values))
plt.xlabel('Resource unit Price')
plt.show()

plt.figure(figsize = (16,8))
sns.distplot(resource['Resource Unit Price'].dropna().apply(np.sqrt))
plt.show()

print("Skewness ", skew(resource['Resource Unit Price'].dropna()))
print("Kurtosis ", kurtosis(resource['Resource Unit Price'].dropna()))
msno.matrix(schools)
plt.show()
print("Shape of Schools " , schools.shape)
print("Missing datas\n" , pd.isnull(schools).sum())
data = [go.Pie( labels = schools['School Metro Type'].value_counts().index, values = schools['School Metro Type'].value_counts().values  )]
layout = dict(
    title='Distribution of School Metro Type',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)

vert_bar_plot(schools, 'School City', 'No.of Schools in each City')
vert_bar_plot(schools, 'School District', 'No .of Schools in different Districts')
schools_75_free_lunch = schools[schools['School Percentage Free Lunch'] > 75.0]
metro_free_lunch = schools_75_free_lunch.groupby('School Metro Type').agg({'School Metro Type' : 'count', 'School Percentage Free Lunch' : 'count'})
data = [go.Pie( labels = metro_free_lunch.index, values = metro_free_lunch['School Percentage Free Lunch']  )]
layout = dict(
    title='Schools providing free lunch with more than 75% based on Metro Type',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)

vert_bar_plot(schools_75_free_lunch, 'School City', 'No.of Schools providing free lunch with more than 75% based on city')
schools_25_free_lunch = schools[schools['School Percentage Free Lunch'] < 25.0]
metro_free_lunch = schools_25_free_lunch.groupby('School Metro Type').agg({'School Metro Type' : 'count', 'School Percentage Free Lunch' : 'count'})
data = [go.Pie( labels = metro_free_lunch.index, values = metro_free_lunch['School Percentage Free Lunch']  )]
layout = dict(
    title='Schools providing free lunch with less than 25% based on Metro Type',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)
vert_bar_plot(schools_25_free_lunch, 'School City', 'No.of Schools providing less than 25% free lunch  based on city')
school_free_lunch = schools.groupby('School State').agg({'School Name' : 'count', 'School Percentage Free Lunch' : 'mean'}).reset_index()
school_free_lunch.columns = ['state' , 'school', 'free_lunch']
for col in school_free_lunch.columns:
    school_free_lunch[col] = school_free_lunch[col].astype(str)
school_free_lunch['text'] = school_free_lunch['state'] + '<br>' + '% of free lunch provided :' + school_free_lunch['free_lunch']
school_free_lunch['code'] = school_free_lunch['state'].map(state_codes)

data = [dict( 
            type = 'choropleth',
            autocolorscale = True,
            locations = school_free_lunch['code'],
            z = school_free_lunch['free_lunch'].astype(float),
            locationmode = 'USA-states',
            text = school_free_lunch['text'].values,
            colorbar = dict(
                title = '% of free lunch'
            )
            )]

layout = dict(
        title = "Percentage of free lunch by states",
        geo = dict(
        scope = 'usa',
        projection = dict( type='albers usa' ),
        showlakes = True,
        lakecolor = 'rgb(255, 255, 255)'),
        )

fig = dict(data=data, layout=layout)
iplot(fig)
school_free_lunch['text'] = school_free_lunch['state'] + '<br>' + '# of schools :' + school_free_lunch['school']
data = [dict( 
            type = 'choropleth',
            autocolorscale = True,
            locations = school_free_lunch['code'],
            z = school_free_lunch['school'].astype(float),
            locationmode = 'USA-states',
            text = school_free_lunch['text'].values,
            colorbar = dict(
                title = '# of Schools'
            )
            )]

layout = dict(
        title = "Number of Schools in Different state",
        geo = dict(
        scope = 'usa',
        projection = dict( type='albers usa' ),
        showlakes = True,
        lakecolor = 'rgb(255, 255, 255)'),
        )

fig = dict(data=data, layout=layout)
iplot(fig)
msno.matrix(donors)
plt.show()
print("Shape of Donors " , donors.shape)
print("Missing datas\n" , pd.isnull(donors).sum())
vert_bar_plot(donors, 'Donor City', 'Top Donors by cities')
data = [go.Pie( labels = donors['Donor Is Teacher'].value_counts().index, values = donors['Donor Is Teacher'].value_counts().values  )]
layout = dict(
    title='Distribution of Donor teachers',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)
msno.matrix(donors)
plt.show()
print("Shape of Donations " , donations.shape)
print("Missing datas\n" , pd.isnull(donations).sum())
vert_bar_plot(donations, 'Donor ID', 'Top Donors')
repeating_donors = donations.groupby('Donor ID').agg({'Donor ID' : 'count', 'Donation Amount': 'sum'})
non_repeating = repeating_donors[repeating_donors['Donor ID'] == 1].shape[0]
two_times = repeating_donors[(repeating_donors['Donor ID']>2) & (repeating_donors['Donor ID'] < 5)].shape[0]
five_times = repeating_donors[(repeating_donors['Donor ID']>= 5) & (repeating_donors['Donor ID'] < 10)].shape[0]
ten_times = repeating_donors[repeating_donors['Donor ID']>= 10].shape[0]


labels = ['Non repeating','2 times','5 times','10 times']
values = [non_repeating,two_times,five_times,ten_times]

data = [go.Pie(labels=labels, values=values)]
layout = dict(
    title='Distribution of Repeating Donors',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)

repeating_donors['Donation Amount'].sum()
non_repeating_donations = repeating_donors['Donation Amount'][repeating_donors['Donor ID'] == 1].sum()
two_times_donations = repeating_donors['Donation Amount'][(repeating_donors['Donor ID']>2) & (repeating_donors['Donor ID'] < 5)].sum()
five_times_donations = repeating_donors['Donation Amount'][(repeating_donors['Donor ID']>= 5) & (repeating_donors['Donor ID'] < 10)].sum()
ten_times_donations = repeating_donors['Donation Amount'][repeating_donors['Donor ID']>= 10].sum()

labels = ['Non repeating','2 times','5 times','10 times']
values = [non_repeating_donations,two_times_donations,five_times_donations,ten_times_donations]

trace = go.Bar(
    x = labels,
    y = values,
    text = values,
    textposition = 'auto',
    marker = dict(
        color = 'rgb(153,153,255)',
        line = dict(
            color = 'rgb(8,48,107)',
            width = 1.5            
        ),
    ),
    opacity = 0.6  
    )
layout = dict(
    title='Donation Amount by repeating Donors',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)

data = [go.Pie(labels=labels, values=values)]
layout = dict(
    title='Donation Amount by repeating Donors',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)
optional_donations = donations.groupby('Donation Included Optional Donation').agg({'Donation Amount': 'sum'}).reset_index()

data = [go.Pie( labels = optional_donations['Donation Included Optional Donation'], values = optional_donations['Donation Amount'])]
layout = dict(
    title='Distribution of donation Amount based on optional donations',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)

data = [go.Pie( labels = donations['Donation Included Optional Donation'].value_counts().index, values = donations['Donation Included Optional Donation'].value_counts().values  )]
layout = dict(
    title='Distribution of optional donations',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)
plt.figure(figsize = (12,8))
plt.scatter(range(donations.shape[0]), np.sort(donations['Donation Amount'].values))
plt.xlabel('Distribution of donation amount')
plt.show()

plt.figure(figsize = (16,8))
sns.distplot(donations['Donation Amount'].apply(np.log))
plt.show()
#Donor Cart Sequence
vert_bar_plot(donations, 'Donor Cart Sequence', 'Donor sequences')
data = [go.Pie( labels = teachers['Teacher Prefix'].value_counts().index, values = teachers['Teacher Prefix'].value_counts().values  )]
layout = dict(
    title='Distribution of Teacher Prefixes',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)
female = teachers[teachers['Teacher Prefix'] == 'Mrs.']['Teacher ID'].count() + teachers[teachers['Teacher Prefix'] == 'Ms.']['Teacher ID'].count()
male = teachers[teachers['Teacher Prefix'] == 'Mr.']['Teacher ID'].count()
x = ['female' , 'male']
y = [female, male]
plt.figure(figsize = (12,8))
sns.barplot(x=x, y=y)
plt.title('Teacher Gender Distribution')
plt.show()
teachers['Teacher First Project Posted Date'] = pd.to_datetime(teachers['Teacher First Project Posted Date'])
teachers['posted_year'] = teachers['Teacher First Project Posted Date'].dt.year
teachers['posted_month'] = teachers['Teacher First Project Posted Date'].dt.month
teachers['posted_day'] = teachers['Teacher First Project Posted Date'].dt.dayofweek

vert_bar_plot(teachers, 'posted_month', 'No of projects posted by month')
vert_bar_plot(teachers, 'posted_day', 'No of projects posted by day')
year_wise = teachers.groupby('posted_year').agg({'Teacher ID' : 'count'}).reset_index()
data = [go.Scatter(x=year_wise.posted_year, y=year_wise['Teacher ID'])]
layout = dict(
    title='Teacher projects posted by year',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)

projects['Project Cost'] = projects['Project Cost'].str.replace(',', '')
projects['Project Cost'] = projects['Project Cost'].str.replace('$', '')
projects['Project Cost'] = projects['Project Cost'].astype(float)
vert_bar_plot(projects, 'Project Type', 'Project type Distribution')

data = [go.Pie( labels = projects['Project Type'].value_counts().index, values = projects['Project Type'].value_counts().values  )]
layout = dict(
    title='Distribution of Project Type',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)
vert_bar_plot(projects, 'Project Subject Category Tree', 'Project Subject Category')
### Distribution of Project Subject Subcategory
vert_bar_plot(projects, 'Project Subject Subcategory Tree', 'Project Subject Subcategory')
#Project Grade Level Category
vert_bar_plot(projects, 'Project Grade Level Category', 'Project Grade Level Category')
#Project Resource Category
vert_bar_plot(projects, 'Project Resource Category', 'Project Resource Category')
projects['Project Posted Date'] = pd.to_datetime(projects['Project Posted Date'])
projects['posted_year'] = projects['Project Posted Date'].dt.year
projects['Project Fully Funded Date'] = pd.to_datetime(projects['Project Fully Funded Date'])
projects['funded_year'] = projects['Project Fully Funded Date'].dt.year
posted = projects.groupby('posted_year').agg({'Teacher ID' : 'count'}).reset_index()
funded = projects.groupby('funded_year').agg({'Teacher ID' : 'count'}).reset_index()
posted = go.Scatter(x=posted.posted_year, y=posted['Teacher ID'])
funded = go.Scatter(x=funded.funded_year, y=funded['Teacher ID'])
layout = dict(
    title='Projects posted and funded',
    )
data = [posted, funded]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
#Project Current Status
data = [go.Pie( labels = projects['Project Current Status'].value_counts().index, values = projects['Project Current Status'].value_counts().values  )]
layout = dict(
    title='Distribution of Project Current Status',
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)
t = projects.groupby('Project Current Status').agg({'Project Cost' : 'mean'}).reset_index()
plt.figure(figsize = (12,8))
sns.barplot(x="Project Current Status", y="Project Cost", data=t)
plt.show()
#Project Grade Level Category
t = projects.groupby('Project Grade Level Category').agg({'Project Cost' : 'mean'}).reset_index()
plt.figure(figsize = (12,8))
sns.barplot(x="Project Grade Level Category", y="Project Cost", data=t)
plt.show()

#Project Resource Category
t = projects.groupby('Project Resource Category').agg({'Project Cost' : 'mean'}).reset_index()
plt.figure(figsize = (12,8))
sns.barplot(x="Project Cost", y="Project Resource Category", data=t)

plt.show()
projects.info()
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(projects['Project Title']))

fig = plt.figure(figsize = (14,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(projects['Project Essay']))

fig = plt.figure(figsize = (14,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
