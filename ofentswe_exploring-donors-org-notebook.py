import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os as os
import squarify
import plotly.graph_objs as go
import plotly.tools as tools
import plotly.offline as ply
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from mpl_toolkits.mplot3d import axes3d
from wordcloud import WordCloud, STOPWORDS
from plotnine import *

# default notebook setting 
color = sns.color_palette()
ply.init_notebook_mode(connected=True)
pd.options.display.max_columns = 1000
%matplotlib inline
os.listdir('../input/')
resources = pd.read_csv('../input/Resources.csv', low_memory=False)
teachers = pd.read_csv('../input/Teachers.csv', low_memory=False)
donors = pd.read_csv('../input/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/Schools.csv', low_memory=False)
donations = pd.read_csv('../input/Donations.csv', low_memory=False)
projects = pd.read_csv('../input/Projects.csv', low_memory=False)
# error_bad_lines=False
teachers.head()
print("\x1b[1;31m We have {0} records \x1b[0m ".format(teachers.shape[0]))
min_date = teachers['Teacher First Project Posted Date'].min()
max_date = teachers['Teacher First Project Posted Date'].max()
num_teachers = len(pd.unique(teachers['Teacher ID']))
print('\x1b[1;31m' + ' From {0} to {1} we have '.format(min_date, max_date) + '\x1b[0m')
_ = venn2((num_teachers,0,0), set_labels=('Teachers',''))
_.get_label_by_id('01').set_text('')
titles = teachers['Teacher Prefix'].value_counts()
trace = go.Table(
    header=dict(values=['Title','Total'],
                fill = dict(color='#cd5c5c'),
                align = ['left']),
    cells=dict(values=[titles.index, titles.values],
               fill = dict(color='#ffb6c1'),
               align = ['left']))

data = [trace] 
ply.iplot(data, filename = 'table')
(ggplot(teachers.dropna()) + 
   aes(x='Teacher Prefix', fill='Teacher Prefix') +
   geom_bar() + 
   ggtitle("Number of Teachers by Title/Prefix") +
   theme(axis_text_x=element_text(rotation=90, hjust=1))
)
X=teachers['Teacher Prefix'].value_counts()
trace = go.Pie(labels=X.index, values=X.values)
ply.iplot([trace], filename='basic_pie_chart')
gender_identifier = {"Mrs.": "Female", "Ms.":"Female", "Mr.":"Male", "Teacher":"Not Specified", 
                     "Dr.":"Not Specified", "Mx.":"Not Specified" }

teachers["gender"] = teachers['Teacher Prefix'].map(gender_identifier)
X=teachers['gender'].value_counts()
colors = ['#F08080', '#1E90FF', '#FFFF99']

plt.pie(X.values, labels=X.index, colors=colors,
        startangle=90,
        explode = (0, 0, 0),
        autopct = '%1.2f%%')
plt.axis('equal')
plt.show()
females = len(teachers[teachers["gender"]=='Female'])
males = len(teachers[teachers["gender"]=='Male'])
unspecified = len(teachers[teachers["gender"]=='Not Specified'])
_ = venn2((females,unspecified,males), set_labels=('Females','Not specified', 'Males'))
teachers['count'] = 1
long_date_name = 'Teacher First Project Posted Date'
teachers_register = teachers[[long_date_name, 'count']].groupby([long_date_name]).count().reset_index()
teachers_register[long_date_name] = pd.to_datetime(teachers_register[long_date_name])
teachers_register = teachers_register.set_index(long_date_name)
max_date = max(teachers_register.index)
min_date = min(teachers_register.index)

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 8)
ax = teachers_register.plot(color='blue')
ax.axvspan(min_date, '2002-12-31', color='red', alpha=0.3)
ax.axvspan('2018-01-01', max_date, color='red', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Number of First Project posted by Date')
ax.set_title('All time Projects posted by Date')
plt.show()
increase_number = teachers_register.reset_index()
start = increase_number[increase_number[long_date_name].dt.year == 2013]
max_start = start[start['count']==max(start['count'])][long_date_name].values[0]
print('\x1b[1;31m' + ' The projects Started increasing with over 500 since 2013 on {0} '.format(max_start.astype(str)[:10]) + '\x1b[0m')
ax = teachers_register.plot(color='blue')
ax.axvspan(max_start, max_date, color='green', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Number of First Project posted by Date')
ax.set_title('(2013 - 2018 snip) Projects Posted by Date ')
plt.show()
ax = teachers_register[teachers_register.index>='2013-10-27'].plot(color='blue')
ax.axhline(500, color='green', linestyle='--')
ax.axvspan('2018-01-01', max_date, color='red', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Number of First Project posted by Date')
ax.set_title('Closer look (2013 -2018 snip) Projects posted by Date')
plt.show()
ax = teachers_register[(teachers_register.index>=min_date) & 
                       (teachers_register.index<='2002-12-31')].plot(color='blue', style='.-')
ax.axhline(6, color='green', linestyle='--')
ax.axvspan(min_date, '2002-12-31', color='red', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Number of First Project posted by Date')
ax.set_title('Closer look (2002 snip) Projects posted by Date')
plt.show()
ax = teachers_register[teachers_register.index>='2018-01-01'].plot(color='blue')
ax.axhline(500, color='green', linestyle='--')
ax.axvspan('2018-01-01', max_date, color='red', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Number of First Project posted by Date')
ax.set_title('Closer look (2018 snip) Projects posted by Date')
plt.show()
teachers_register['Year'] = teachers_register.reset_index()[long_date_name].dt.year
per_year = teachers[[long_date_name, 'count']].groupby(long_date_name).count().reset_index()
per_year[long_date_name] = pd.to_datetime(per_year[long_date_name])
per_year['Year'] = per_year[long_date_name].dt.year

ax = per_year[['Year', 'count']].groupby('Year').agg({'count':'sum'}).plot(color='blue', style='.-')
ax.set_xlabel('Date')
ax.set_ylabel('Number of First Project posted by Year')
ax.set_title('Projects poted by Year')
plt.show()
(ggplot(teachers_register.reset_index())
 + geom_point(aes(long_date_name, 'count', fill='count'))
 + labs(y='number of Projects posted ')
 + ggtitle("Projects posted by Date") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)
gender_based = teachers[[long_date_name, 'gender', 'count']].groupby([long_date_name, 'gender']).count().reset_index()
gender_based[long_date_name] = pd.to_datetime(gender_based[long_date_name])

(ggplot(gender_based) 
 + aes(long_date_name, 'count', color='factor(gender)') 
 + geom_line()
 + ggtitle("Projects posted by Date - Gender Based ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)
(ggplot(gender_based[gender_based.gender=='Male'])
 + aes(long_date_name, 'count', color='gender') 
 + geom_line()
 + ggtitle("Projects posted by Date - Males ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)
(ggplot(gender_based[gender_based.gender=='Female']) 
 + aes(long_date_name, 'count', color='gender') 
 + geom_line()
 + ggtitle("Projects posted by Date - Females ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)
(ggplot(gender_based[gender_based.gender=='Not Specified']) 
 + aes(long_date_name, 'count', color='gender') 
 + geom_line()
 + ggtitle("Projects posted by Date - Not Specified ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)
teachers[long_date_name] = pd.to_datetime(teachers[long_date_name])
teachers['Year'] = teachers[long_date_name].dt.year
gender_based_year = teachers[['Year', 'gender', 'count']].groupby(['Year', 'gender']).count().reset_index()

(ggplot(gender_based_year) 
 + aes('Year', 'count', color='factor(gender)') 
 + geom_point()
 + geom_line()
 + ggtitle("Projects posted by Year - Gender Based ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)
(ggplot(gender_based) 
  + aes('count', fill='gender', color='gender') 
  + ggtitle("Projects posted on Gender ") 
  + geom_density(alpha=0.2)  
)
(ggplot(gender_based[gender_based.gender=='Male']) 
   + aes('count', fill='gender', color='gender') 
   + ggtitle("Projects posted for Males") 
   + geom_density(alpha=0.2)  
)
(ggplot(gender_based[gender_based.gender=='Female']) 
  + aes('count', fill='gender', color='gender')
  +  ggtitle("Projects posted for Females ") 
  + geom_density(alpha=0.2)  
)
(ggplot(gender_based[gender_based.gender=='Not Specified'])
  + aes('count', fill='gender', color='gender')
  + ggtitle("Projects posted for Unknown ") 
  + geom_density(alpha=0.2)  
)
schools.head()
num_schools = len(pd.unique(schools['School ID']))
print('\x1b[1;31m' + 'We have {0}'.format(num_schools) + ' Schools \x1b[0m')
_ = venn2((num_schools,0,0), set_labels=('Schools',''))
_.get_label_by_id('01').set_text('')
metro_types = schools['School Metro Type'].value_counts()
trace = go.Table(
    header=dict(values=['Metro Type','Total'],
                fill = dict(color='#cd5c5c'),
                align = ['left']),
    cells=dict(values=[metro_types.index, metro_types.values],
               fill = dict(color='#ffb6c1'),
               align = ['left']))

data = [trace] 
ply.iplot(data, filename = 'metro_table')
X=schools['School Metro Type'].value_counts()
trace = go.Pie(labels=X.index, values=X.values)
ply.iplot([trace], filename='school_pie_chart')
(ggplot(schools) + 
   aes(x='School Metro Type', fill='School Metro Type') +
   geom_bar() + 
   ggtitle("Number of Schoos by Metro type") +
   theme(axis_text_x=element_text(rotation=90, hjust=1))
)
(ggplot(schools.dropna()) 
  + aes(x='School Percentage Free Lunch', fill='School Metro Type')
  + ggtitle("Distribution of School Percentage Free Lunch ") 
  + geom_histogram(binwidth=5)
)
sns.kdeplot(schools['School Percentage Free Lunch'].dropna())
plt.title('School Percentage Free Lunch distribution')
plt.xlabel('Number of schools')
plt.ylabel('Percentage')
plt.show()
sns.kdeplot(schools['School Percentage Free Lunch'].dropna(), cumulative=True)
plt.title('School Percentage Free Lunch - Cummulative distribution')
plt.xlabel('Number of schools')
plt.ylabel('Percentage')
plt.show()
schools[schools['School Metro Type'] == 'rural']['School District'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 rural Districts")
plt.ylabel("Number of schools")
plt.xlabel("District Name")
schools[schools['School Metro Type'] == 'urban']['School District'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 Urban Districts")
plt.ylabel("Number of schools")
plt.xlabel("District Name")
schools[schools['School Metro Type'] == 'suburban']['School District'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 Suburban Districts")
plt.ylabel("Number of schools")
plt.xlabel("District Name")
schools[schools['School Metro Type'] == 'town']['School District'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 Town Districts")
plt.ylabel("Number of schools")
plt.xlabel("District Name")
schools[schools['School Metro Type'] == 'unknown']['School District'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 Unknown Districts")
plt.ylabel("Number of schools")
plt.xlabel("District Name")
schools['School District'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 Overall Districts")
plt.ylabel("Number of schools")
plt.xlabel("District Name")
schools['School City'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 Cities")
plt.ylabel("Number of schools")
plt.xlabel("City Name")
schools['School County'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 Counties")
plt.ylabel("Number of schools")
plt.xlabel("County Name")
(ggplot(schools) + 
   aes(x='School State', fill='School State') +
   geom_bar() + 
   coord_flip() +
   ggtitle("Number of Schoos by School State") +
   theme(figure_size=(12, 20), axis_text_x=element_text(rotation=90, hjust=1))
)
states = schools['School State'].value_counts()
x = 0.
y = 0.
width = 50.
height = 50.
type_list = states.index
values = states.values

normed = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(normed, x, y, width, height)

color_brewer = ['#41B5A3','#FFAF87','#FF8E72','#ED6A5E','#377771','#E89005','#C6000D','#000000','#05668D','#028090','#9FD35C','#02C39A','#F0F3BD','#41B5A3','#FF6F59','#254441','#B2B09B','#EF3054','#9D9CE8','#0F4777','#5F67DD','#235077','#CCE4F9','#1748D1','#8BB3D6','#467196','#F2C4A2','#F2B1A4','#C42746','#330C25'] * 2
shapes = []
annotations = []
counter = 0

for r in rects:
    shapes.append( 
        dict(
            type = 'rect', 
            x0 = r['x'], 
            y0 = r['y'], 
            x1 = r['x'] + r['dx'], 
            y1 = r['y'] + r['dy'],
            line = dict(width = 1),
            fillcolor = color_brewer[counter]
        ) 
    )
    annotations.append(
        dict(
            x = r['x'] + (r['dx']/2),
            y = r['y'] + (r['dy']/2),
            text = "{}".format(type_list[counter]),
            showarrow = False
        )
    )
    counter = counter + 1
    if counter >= len(color_brewer):
        counter = 0

# For hover text
trace0 = go.Scatter(
    x = [ r['x'] + (r['dx']/2) for r in rects ], 
    y = [ r['y'] + (r['dy']/2) for r in rects ],
    text = [ str(v) for v in values ], 
    mode = 'text',
)

layout = dict(
    height = 1000, 
    width = 1000,
    xaxis = dict(showgrid=False,zeroline=False),
    yaxis = dict(showgrid=False,zeroline=False),
    shapes = shapes,
    annotations = annotations,
    hovermode = 'closest',
    font = dict(color="#FFFFFF"),
    margin = go.Margin(
            l=0,
            r=0,
            pad=0
        )
)

# With hovertext
figure = dict(data=[trace0], layout=layout)
ply.iplot(figure, filename='state-treemap')
# Get the coordinates of the states from the link using pandas 
usa_codes = [['AL', 'Alabama'],
       ['AK', 'Alaska'], 
       ['AZ', 'Arizona'],
       ['AR', 'Arkansas'],
       ['CA', 'California'],
       ['CO', 'Colorado'],
       ['CT', 'Connecticut'],
       ['DE', 'Delaware'],
       ['FL', 'Florida'],
       ['GA', 'Georgia'],
       ['HI', 'Hawaii'],
       ['ID', 'Idaho'],
       ['IL', 'Illinois'],
       ['IN', 'Indiana'],
       ['IA', 'Iowa'],
       ['KS', 'Kansas'],
       ['KY', 'Kentucky'],
       ['LA', 'Louisiana'],
       ['ME', 'Maine'],
       ['MD', 'Maryland'],
       ['MA', 'Massachusetts'],
       ['MI', 'Michigan'],
       ['MN', 'Minnesota'],
       ['MS', 'Mississippi'],
       ['MO', 'Missouri'],
       ['MT', 'Montana'],
       ['NE', 'Nebraska'],
       ['NV', 'Nevada'],
       ['NH', 'New Hampshire'],
       ['NJ', 'New Jersey'],
       ['NM', 'New Mexico'],
       ['NY', 'New York'],
       ['NC', 'North Carolina'],
       ['ND', 'North Dakota'],
       ['OH', 'Ohio'],
       ['OK', 'Oklahoma'],
       ['OR', 'Oregon'],
       ['PA', 'Pennsylvania'],
       ['RI', 'Rhode Island'],
       ['SC', 'South Carolina'],
       ['SD', 'South Dakota'],
       ['TN', 'Tennessee'],
       ['TX', 'Texas'],
       ['UT', 'Utah'],
       ['VT', 'Vermont'],
       ['VA', 'Virginia'],
       ['WA', 'Washington'],
       ['WV', 'West Virginia'],
       ['WI', 'Wisconsin'],
       ['WY', 'Wyoming']]
us_states = pd.DataFrame(data=usa_codes, columns=['Code', 'State'])
us_states = us_states.rename({'State': 'School State'})
counts = pd.DataFrame({'State': schools['School State'].value_counts().index, 
                       'Total': schools['School State'].value_counts().values})
maps_df = counts.merge(us_states, on='State', how='inner')
maps_df['text'] = maps_df['State'] + '<br>  ' + (maps_df['Total']).astype(str)+' donations'
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = maps_df['Code'],
        z = maps_df['Total'].astype(float),
        locationmode = 'USA-states',
        text = maps_df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Millions USD")
        ) ]

layout = dict(
        title = 'DonorsChoose.org Donations <br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
ply.iplot( fig, filename='d3-cloropleth-map' )
projects.head(2)
print("\x1b[1;31m We have {0} records \x1b[0m ".format(projects.shape[0]))
min_date = pd.to_datetime(projects['Project Fully Funded Date']).min()
max_date = pd.to_datetime(projects['Project Fully Funded Date']).max()
num_projects = len(pd.unique(projects['Project ID']))
print('\x1b[1;31m' + ' From {0} to {1} we have '.format(min_date, max_date) + '\x1b[0m')
_ = venn2((num_projects, 0, 0), set_labels=('Projects',''))
_.get_label_by_id('01').set_text('')
(ggplot(projects) + 
   aes(x='Project Current Status', fill='Project Current Status') +
   geom_bar() + 
   ggtitle("Number of Projects by Project Current Status") +
   theme(axis_text_x=element_text(rotation=90, hjust=1))
)
(ggplot(projects) + 
   aes(x='Project Type', fill='Project Type') +
   geom_bar() + 
   ggtitle("Number of Projects by Type") +
   theme(axis_text_x=element_text(rotation=90, hjust=1))
)
expired = len(projects[projects["Project Current Status"]=='Expired'])
fully = len(projects[projects["Project Current Status"]=='Fully Funded'])
live = len(projects[projects["Project Current Status"]=='Live'])
_ = venn2((expired,fully,live), set_labels=('Expired','Fully Funded', 'Live'))
X=projects['Project Current Status'].value_counts()
trace = go.Pie(labels=X.index, values=X.values)
ply.iplot([trace], filename='basic_pie_gender')
(ggplot(projects.dropna()) + 
   aes(x='Project Resource Category', fill='Project Resource Category') +
   geom_bar() + 
   coord_flip() + 
   ggtitle("Number of Projects by Project Resource Category") +
   theme(axis_text_x=element_text(rotation=90, hjust=1), figure_size=(12, 10))
)
(ggplot(projects.dropna()) + 
   aes(x='Project Grade Level Category', fill='Project Grade Level Category') +
   geom_bar() + 
   coord_flip() + 
   ggtitle("Number of Projects by Grade ") +
   theme(axis_text_x=element_text(rotation=90, hjust=1))
)
X=projects['Project Grade Level Category'].value_counts()
trace = go.Pie(labels=X.index, values=X.values)
ply.iplot([trace], filename='basic_pie_grade')
(ggplot(projects.dropna()) + 
   aes(x='Project Subject Category Tree', fill='Project Subject Category Tree') +
   geom_bar() + 
   coord_flip() + 
   ggtitle("Number of Projects by Subject ") +
   theme(axis_text_x=element_text(rotation=90, hjust=1), figure_size=(10, 30))
)
sns.kdeplot(projects['Teacher Project Posted Sequence'].dropna())
plt.title('Teachers Posted Sequence distribution')
sns.kdeplot(projects['Project Cost'], cumulative=True)
plt.xlabel('Project Cost')
plt.ylabel('Percentage of projects')
plt.show()
project_date = 'Project Posted Date'
cost = 'Project Cost'

projects_cost = projects[[project_date, cost]].groupby([project_date]).agg({cost:'sum'}).reset_index()
projects_cost[project_date] = pd.to_datetime(projects_cost[project_date])
projects_cost = projects_cost.set_index(project_date)
max_date = max(projects_cost.index)

ax = projects_cost.plot(color='blue')
ax.axvspan('2018-01-01', max_date, color='red', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Project Cost in dollars ($)')
ax.set_title('Projects Cost by Date')
plt.show()
projects_cost['Year'] = projects_cost.reset_index()[project_date].dt.year
per_year = projects[[project_date, cost]].groupby(project_date).agg({cost:'sum'}).reset_index()
per_year[project_date] = pd.to_datetime(per_year[project_date])
per_year['Year'] = per_year[project_date].dt.year

ax = per_year[['Year', cost]].groupby('Year').agg({cost:'sum'}).plot(color='blue', style='.-')
ax.axhspan(200000, 250000, color='green', alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('Project Cost in dollars ($)')
ax.set_title('Projects Cost by Year')
plt.show()
grade = 'Project Grade Level Category'
grade_based = projects[[project_date, grade, cost]].groupby([project_date, grade]).agg({cost:'sum'}).reset_index()
grade_based[project_date] = pd.to_datetime(grade_based[project_date])

(ggplot(grade_based) 
 + aes(project_date, cost, color='Project Grade Level Category') 
 + geom_line()
 + ggtitle("Projects Cost by Date - Grade Based ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)
fully = 'Project Fully Funded Date'
grade_based = projects[[fully, grade, cost]].groupby([fully, grade]).agg({cost:'sum'}).reset_index()
grade_based[fully] = pd.to_datetime(grade_based[fully])

(ggplot(grade_based.dropna()) 
 + aes(fully, cost, color='Project Grade Level Category') 
 + geom_line()
 + ggtitle("Projects Cost by Fully Funded Date - Grade Based ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)
projects[project_date] = pd.to_datetime(projects[project_date])
projects['Year'] = projects[project_date].dt.year
grade_based_year = projects[['Year', grade, cost]].groupby(['Year', grade]).agg({cost:'sum'}).reset_index()

(ggplot(grade_based_year) 
 + aes('Year', cost, color='Project Grade Level Category') 
 + geom_point()
 + geom_line()
 + ggtitle("Projects Cost by Year - Grade Based ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)
status = 'Project Current Status'
projects[project_date] = pd.to_datetime(projects[project_date])
projects['Year'] = projects[project_date].dt.year
grade_based_year = projects[['Year', status, cost]].groupby(['Year', status]).agg({cost:'sum'}).reset_index()

(ggplot(grade_based_year) 
 + aes('Year', cost, color=status) 
 + geom_point()
 + geom_line()
 + ggtitle("Projects Cost by Year - Status Based ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)
p_type = 'Project Type'
projects[project_date] = pd.to_datetime(projects[project_date])
projects['Year'] = projects[project_date].dt.year
grade_based_year = projects[['Year', p_type, cost]].groupby(['Year', p_type]).agg({cost:'sum'}).reset_index()

(ggplot(grade_based_year) 
 + aes('Year', cost, color=p_type) 
 + geom_point()
 + geom_line()
 + ggtitle("Projects Cost by Year - Project Type ") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)
stop_words = set(STOPWORDS)
new_words = ("DONOTREMOVEESSAYDIVIDER", "go", "bk")
new_stops = stop_words.union(new_words)
wordcloud = WordCloud(width=1440, height=1080, stopwords=new_stops).generate(" ".join(projects['Project Need Statement'].astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(wordcloud)
plt.axis('off')
wordcloud = WordCloud(width=1440, height=1080, stopwords=new_stops).generate(" ".join(projects['Project Title'].astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(wordcloud)
plt.axis('off')
wordcloud = WordCloud(width=1440, height=1080, stopwords=new_stops).generate(" ".join(projects['Project Essay'].sample(50000).astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(wordcloud)
plt.axis('off')
wordcloud = WordCloud(width=1440, height=1080, stopwords=new_stops).generate(" ".join(projects['Project Short Description'].sample(100000).astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(wordcloud)
plt.axis('off')
resources.head()
vendor_resource = resources["Resource Vendor Name"].value_counts()
trace = go.Table(
    header=dict(values=['Vendor Name','Total Resources'],
                fill = dict(color='#cd5c5c'),
                align = ['left']),
    cells=dict(values=[vendor_resource.index, vendor_resource.values],
               fill = dict(color='#ffb6c1'),
               align = ['left']))

data = [trace] 
ply.iplot(data, filename = 'metro_table')
(ggplot(resources.dropna()) + 
   aes(x="Resource Vendor Name", fill="Resource Vendor Name") +
   geom_bar(size=20) + 
   coord_flip() +
   ggtitle("Number of elements by Phase") +
   theme(axis_text_x=element_text(rotation=90, hjust=1), figure_size=(12, 20))
)
sns.kdeplot(resources['Resource Quantity'].dropna(), cumulative=True)
plt.title("Distribution of Resource quatity")
wordcloud = WordCloud(width=1440, height=1080, stopwords=new_stops).generate(" ".join(resources['Resource Item Name'].sample(60000).astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(wordcloud)
plt.axis('off')
donors.head()
X=donors['Donor Is Teacher'].value_counts()
trace = go.Pie(labels=X.index, values=X.values)
ply.iplot([trace], filename='basic_pie_grade')
(ggplot(donors) + 
   aes(x='Donor Is Teacher', fill='Donor Is Teacher') +
   geom_bar() + 
   ggtitle("Number of Donors that are Teachers and Not Teachers") +
   theme(axis_text_x=element_text(rotation=90, hjust=1))
)
donors['Donor City'].value_counts()[:10].plot(kind='bar')
plt.title('City by number of Donors')
plt.xlabel('City Name')
plt.ylabel('Number of Donors')
donors['Donor State'].value_counts()[:10].plot(kind='bar')
plt.title('State by number of Donors')
plt.xlabel('State')
plt.ylabel('Number of Donors')
# Get the coordinates of the states from the link using pandas 
counts = pd.DataFrame({'State': donors['Donor State'].value_counts().index, 
                       'Total': donors['Donor State'].value_counts().values})
maps_df = counts.merge(us_states, on='State', how='inner')

maps_df['text'] = maps_df['State'] + '<br>  ' + (maps_df['Total']).astype(str)+' donations'
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = maps_df['Code'],
        z = maps_df['Total'].astype(float),
        locationmode = 'USA-states',
        text = maps_df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Millions USD")
        ) ]

layout = dict(
        title = 'DonorsChoose.org Donors <br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
ply.iplot( fig, filename='d3-cloropleth-map' )
donations.head()
print(donations['Donation Amount'].min())
print(donations['Donation Amount'].max())
X=donations['Donation Included Optional Donation'].value_counts()
trace = go.Pie(labels=X.index, values=X.values)
ply.iplot([trace], filename='basic_pie_grade')
(ggplot(donations) + 
   aes(x='Donation Included Optional Donation', fill='Donation Included Optional Donation') +
   geom_bar() + 
   ggtitle("Number of Donations included additional donations/not") +
   theme(axis_text_x=element_text(rotation=90, hjust=1))
)
donations['Date'] = pd.to_datetime(donations['Donation Received Date'])
donations['Hour'] = donations['Date'].dt.hour
donations.Hour.plot(kind='density')
donations.Hour.value_counts().sort_index().plot(kind='bar')
plt.title('Donations by Time in Hours')
plt.ylabel('Number of Donations')
plt.xlabel('Hour')
sns.distplot(donations.Hour.value_counts(sort=True).values, hist=False, norm_hist=True)
donation_date = 'Donation Received Date'
options = 'Donation Included Optional Donation'
donations[donation_date] = pd.to_datetime(donations[donation_date])
donations['Date'] = donations[donation_date].dt.date
donations_time_based = donations[['Date', options, 'Donation Amount']].groupby(['Date', options]).agg({'Donation Amount':'count'}).plot()

#Donation Included Optional Donation 	Donation Amount 	Donor Cart Sequence

donation_date = 'Donation Received Date'
options = 'Donation Included Optional Donation'
donations[donation_date] = pd.to_datetime(donations[donation_date])
donations['Year'] = donations[donation_date].dt.year
donations_time_based = donations[['Year', options, 'Donation Amount']].groupby(['Year', options]).agg({'Donation Amount':'count'}).reset_index()

(ggplot(donations_time_based)
 + aes('Year', 'Donation Amount', color=options) 
 + geom_point()
 + geom_line()
 + ggtitle("Donation Amount by Year") 
 + theme(axis_text_x=element_text(rotation=75, hjust=1))
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, train_test_split, validation_curve
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF 
from sklearn.metrics import accuracy_score, auc, classification_report, mean_squared_error, roc_curve
## Comming soon