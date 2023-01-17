import pandas as pd
import numpy as np
import random

import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

folder = "../input/io/"
schools = pd.read_csv(folder + "Schools.csv", error_bad_lines=False)
teachers = pd.read_csv(folder + "Teachers.csv", parse_dates=['Teacher First Project Posted Date'])
donors = pd.read_csv(folder + "Donors.csv", low_memory=False)
donations = pd.read_csv(folder + "Donations.csv", parse_dates=['Donation Received Date'])
state_data = pd.read_csv("../input/usa-state-area-and-2017-population-estimates/State Land Areas.csv")

pd.options.display.float_format = '{:,.1f}'.format
school_county = schools['School State'].value_counts().reset_index()
school_county.columns = ['state', 'schools']

for col in school_county.columns:
    school_county[col] = school_county[col].astype(str)

state_codes = {'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 'Nevada': 'NV', 'Maine': 'ME', 'other': ''}

school_county['code'] = school_county['state'].map(state_codes)
def grey_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(0, 100)

word_string = schools['School Name'].str.cat(sep=' ')

wordcloud = WordCloud(
    stopwords=STOPWORDS,
    background_color='white',
    width=3000,
    height=1000).generate(word_string)

plt.figure(figsize=(20,40))
plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),interpolation="bilinear")
plt.axis('off')
plt.show()
school_words = [x.strip() for x in word_string.split(' ')]
df_school_words = pd.DataFrame({'words':school_words})
word_counts = df_school_words['words'].value_counts().reset_index()

data = [go.Bar(
            x=word_counts['words'].head(20),
            y=word_counts['index'].head(20),
            orientation = 'h',
            marker = dict(color = 'rgba(1, 140, 221, 1)')
)]

layout = dict(
    title='Top 20 Words used in School Names',
    yaxis=dict(autorange='reversed'),
    width=500,
    height=600,
    paper_bgcolor="#f9f9f9",
    plot_bgcolor="#f9f9f9"
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='horizontal-bar')
scl = [[0, '#FF6633'],[.2, '#FF9933'],[0.4, '#FFCC33'],
            [0.6, '#FFFF33'],[.8, '#99FF33'],[1, '#66FF33']]

# https://plot.ly/python/choropleth-maps/
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = school_county['code'], # The variable identifying state
        z = school_county['schools'].astype(float), # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = school_county['state'], 
        colorbar = dict(  
            title = "# of Schools")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = '# of Schools by State',
        width=800,
        height=500,
        paper_bgcolor="#f9f9f9",
        plot_bgcolor="#f9f9f9",
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
iplot(fig)
temp = schools['School Name'].value_counts().reset_index().head(15)

data = [go.Bar(
            x=temp['School Name'],
            y=temp['index'],
            orientation = 'h',
            marker = dict(color = 'rgba(1, 140, 221, 1)'),
            text=temp['School Name'],
            textposition = 'auto',
            textfont=dict(
                family='sans serif',
                size=12,
                color='#ffffff'
            ),
            width = .8
)]

layout = dict(
    title='Top 15 School Names',
     width=500,
    height=700,
    paper_bgcolor="#f9f9f9",
    plot_bgcolor="#f9f9f9",
    margin=dict(
        l=220,
        r=20,
        t=70,
        b=70,
    ),
    yaxis=dict(autorange='reversed')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='horizontal-bar')
temp = schools['School Metro Type'].value_counts(normalize=True).reset_index()

colors = ['#4f92ff', '#4ffff0', '#4fff6f', '#f9ff4f', '#ffbe4f']
explode = (0.05, 0.05, 0.1, 0.1, 0.1)
 
plt.pie(temp['School Metro Type'], explode=explode, labels=temp['index'], colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=0)
 
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.suptitle('% Schools across Metro types', fontsize=20)
plt.rcParams['font.size'] = 16
plt.axis('equal')
plt.show()
def free_lunch_buckets(x):
    if x>=75:
        return "High (75-100)"
    if x<=25:
        return "Low (0-25)"
    else:
        return "Medium (26-74)"

schools['Free Lunch'] = schools['School Percentage Free Lunch'].apply(lambda x: free_lunch_buckets(x))

temp = schools.groupby('School State')['Free Lunch'].value_counts().unstack(fill_value=0).reset_index()

temp['total_schools'] = temp['High (75-100)'] + temp['Medium (26-74)'] + temp['Low (0-25)']

temp['High (75-100)'] = (temp['High (75-100)']*100) / temp['total_schools']
temp['Medium (26-74)'] = (temp['Medium (26-74)']*100) / temp['total_schools']
temp['Low (0-25)'] = (temp['Low (0-25)']*100) / temp['total_schools']

temp = temp.sort_values(['High (75-100)', 'Medium (26-74)', 'Low (0-25)'])

trace1 = go.Bar(
    x=temp['School State'],
    y=temp['High (75-100)'],
    name='Schools with more than or equal to 75% of students availing free lunch'
)

trace2 = go.Bar(
    x=temp['School State'],
    y=temp['Medium (26-74)'],
    name='Schools with 26% to 74% of students availing free lunch'
)

trace3 = go.Bar(
    x=temp['School State'],
    y=temp['Low (0-25)'],
    name='Schools with less than or equal to 25% of students availing free lunch'
)

data = [trace1, trace2, trace3]
layout = go.Layout(
    barmode='stack',
    title='% of schools by state based on the % free lunch',
    legend=dict(x=0, y=1.25),
    height=500,
    margin=go.Margin(
        l=25,
        r=5,
        b=150,
        t=120,
        pad=4
    ),
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')
temp = schools.groupby('School Metro Type')['Free Lunch'].value_counts().unstack(fill_value=0).reset_index()

temp['total_schools'] = temp['High (75-100)'] + temp['Medium (26-74)'] + temp['Low (0-25)']

temp['High (75-100)'] = (temp['High (75-100)']*100) / temp['total_schools']
temp['Medium (26-74)'] = (temp['Medium (26-74)']*100) / temp['total_schools']
temp['Low (0-25)'] = (temp['Low (0-25)']*100) / temp['total_schools']

temp = temp.sort_values(['High (75-100)', 'Medium (26-74)', 'Low (0-25)'])
#temp

trace = go.Heatmap(z=[temp['High (75-100)'], temp['Medium (26-74)'], temp['Low (0-25)']],
                   x=temp['School Metro Type'],
                   y=['High (75-100)', 'Medium (26-74)', 'Low (0-25)'],
                   colorscale='Portland'
                  )

layout = go.Layout(
    title='% proportion of schools based on % free lunch',
    margin=go.Margin(
        l=150,
        r=5,
        b=50,
        t=100,
        pad=4
    )
)

fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)
schools_state_wise = schools['School State'].value_counts().reset_index()
state_data['State'] = state_data['State'].apply(lambda x: x.strip())
state_data = pd.merge(schools_state_wise, state_data, how='left', left_on='index', right_on='State')
state_data = pd.merge(state_data, school_county, how='left', left_on='State', right_on='state')
state_data.drop('index', axis=1, inplace=True)
state_data['Land area per school'] = state_data['Land Area in Sq Miles'] / state_data['School State']
state_data['Population per school'] = state_data['2017 Population Estimate'] / state_data['School State']
state_data['Schools per million people'] = state_data['School State'] * 1000000 / state_data['2017 Population Estimate']
state_data['Schools per 1000 sq miles'] = state_data['School State'] * 1000 / state_data['Land Area in Sq Miles']
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = state_data['code'], # The variable identifying state
        z = state_data['Schools per million people'], # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = school_county['state'], 
        colorbar = dict(  
            title = "Schools per million people")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = 'Schools per million people',
        width=800,
        height=500,
        paper_bgcolor="#f9f9f9",
        plot_bgcolor="#f9f9f9",
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
iplot(fig)
state_data_1 = state_data.sort_values('Schools per 1000 sq miles').head(50)

#scl = [[0, '#FF3019'],[.1, '#FF611D'],[0.2, '#FF8F20'],[.3, '#FFBD24'],[0.4, '#FFE928'],
#            [0.5, '#EAFF2B'],[.6, '#C0FF2E'],[.7, '#99FF32'],[0.8, '#72FF36'],[.9, '#4DFF39'],[1, '#3DFF4F']]

scl = [[0, '#FF3019'],[0.1, '#FF8F20'],[0.2, '#FFE928'],
            [.4, '#C0FF2E'],[0.6, '#72FF36'],[1, '#3DFF4F']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = state_data_1['code'], # The variable identifying state
        z = state_data_1['Schools per 1000 sq miles'], # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = school_county['state'], 
        colorbar = dict(  
            title = "Schools per 1000 sq miles")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = 'Schools per 1000 sq miles',
        width=800,
        height=500,
        paper_bgcolor="#f9f9f9",
        plot_bgcolor="#f9f9f9",
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
iplot(fig)
# Create a trace
trace = go.Scatter(
    x = state_data_1['Schools per million people'],
    y = state_data_1['Schools per 1000 sq miles'],
    mode = 'markers',
    marker = dict(
        size = 10,
        #color = 'rgba(109, 255, 68, 1)',
        color = state_data_1['schools'],
        colorscale='Earth',
        showscale=True,
        colorbar=dict(
                title='Number of Schools'
            ),
        line = dict(
            width =0.5,
            color = 'rgb(0, 0, 0)'
        )
    ),
    text = state_data_1['State']
)

layout= go.Layout(
    title= 'Schools per million people vs per 1000 sq miles',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Schools per million people',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Schools per 1000 sq miles',
        ticklen= 5,
        gridwidth= 2,
    ),
    width = 800,
    height = 800,
    showlegend = False
)
fig= go.Figure(data=[trace], layout=layout)
py.iplot(fig)
temp = teachers['Teacher Prefix'].value_counts().reset_index()

trace = go.Pie(labels=temp['index'], 
               values=temp['Teacher Prefix'],
               textfont=dict(size=20),
               marker=dict(colors = ['#4f92ff', '#4ffff0', '#4fff6f', '#f9ff4f', '#ffbe4f', '#f04fff'])
              )

layout= go.Layout(
    title= '% of teachers by title'
)

fig= go.Figure(data=[trace], layout=layout)
py.iplot(fig)
teachers['weekdays'] = teachers['Teacher First Project Posted Date'].dt.dayofweek
teachers['month'] = teachers['Teacher First Project Posted Date'].dt.month 
teachers['year'] = teachers['Teacher First Project Posted Date'].dt.year

dmap = {0:'Monday',1:'Tueday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
teachers['weekdays'] = teachers['weekdays'].map(dmap)

month_dict = {1 :"H1",2 :"H1",3 :"H1",4 :"H1",5 : "H1",6 : "H1",7 : "H2",8 :"H2",9 :"H2",10 :"H2",11 :"H2",12 :"H2"}
teachers['half_year'] = teachers['month'].map(month_dict)

month_dict = {1 :"Jan",2 :"Feb",3 :"Mar",4 :"Apr",5 : "May",6 : "Jun",7 : "Jul",8 :"Aug",9 :"Sep",10 :"Oct",11 :"Nov",12 :"Dec"}
teachers['month'] = teachers['month'].map(month_dict)
temp = teachers['year'].value_counts().reset_index().sort_values('index')

data = [go.Scatter(
            x=temp['index'],
            y=temp['year'])]

layout = go.Layout(
    title="# of teachers posted in each year"
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
teachers['year_and_half'] = teachers["year"].map(str) + teachers["half_year"]
temp = teachers.groupby('year_and_half').count()['Teacher ID'].reset_index()

data = [go.Bar(
            y=temp['Teacher ID'],
            x=temp['year_and_half'],
            marker = dict(color = 'rgba(1, 140, 221, 1)'),
            width=0.4
)]

layout = dict(
    title='Teachers posted in each half of the year',     
    width=800,
    height=400,
    paper_bgcolor="#f9f9f9",
    plot_bgcolor="#f9f9f9"
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='horizontal-bar')
temp = donors['Donor State'].value_counts().reset_index()
temp['code'] = temp['index'].map(state_codes)

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = temp['code'], # The variable identifying state
        z = temp['Donor State'], # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = temp['index'], 
        colorbar = dict(  
            title = "# of donors in each state")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = '# of donors in each state',
        width=800,
        height=500,
        paper_bgcolor="#f9f9f9",
        plot_bgcolor="#f9f9f9",
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
iplot(fig)
temp = donors['Donor Is Teacher'].value_counts().reset_index()

trace = go.Pie(labels=temp['index'], 
               values=temp['Donor Is Teacher'],
               textfont=dict(size=20),
               marker=dict(colors = ['#ffbe4f', '#4ffff0'])
              )

layout= go.Layout(
    title= '% of donors who are teachers'
)

fig= go.Figure(data=[trace], layout=layout)
py.iplot(fig)
temp = donations['Donation Included Optional Donation'].value_counts().reset_index()

trace = go.Pie(labels=temp['index'], 
               values=temp['Donation Included Optional Donation'],
               textfont=dict(size=20),
               marker=dict(colors = ['#4ffff0', '#ffbe4f'])
              )

layout= go.Layout(
    title= '% of donations with optional donation'
)

fig= go.Figure(data=[trace], layout=layout)
py.iplot(fig)
def bucket(a):
    if a == 1:
        return "Only once"
    if a > 1 and a <=10:
        return "Occasional (2-10)"
    if a > 10 and a <= 100:
        return "Regular (11-100)"
    if a > 100:
        return "Very Frequent (>100)"

temp = donations['Donor ID'].value_counts().reset_index()
temp['donor bucket'] = temp['Donor ID'].apply(lambda x: bucket(x))
temp = temp['donor bucket'].value_counts().reset_index()

trace = go.Pie(labels=temp['index'], 
               values=temp['donor bucket'],
               textfont=dict(size=20),
               marker=dict(colors = ['#4f92ff', '#4ffff0', '#4fff6f', '#f9ff4f'])
              )

layout= go.Layout(
    title= '% of donors based # on donations made'
)

fig= go.Figure(data=[trace], layout=layout)
py.iplot(fig)
donations['weekdays'] = donations['Donation Received Date'].dt.dayofweek
donations['month'] = donations['Donation Received Date'].dt.month 
donations['year'] = donations['Donation Received Date'].dt.year

dmap = {0:'Monday',1:'Tueday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
donations['weekdays'] = donations['weekdays'].map(dmap)

month_dict = {1 :"H1",2 :"H1",3 :"H1",4 :"H1",5 : "H1",6 : "H1",7 : "H2",8 :"H2",9 :"H2",10 :"H2",11 :"H2",12 :"H2"}
donations['half_year'] = donations['month'].map(month_dict)

month_dict = {1 :"Jan",2 :"Feb",3 :"Mar",4 :"Apr",5 : "May",6 : "Jun",7 : "Jul",8 :"Aug",9 :"Sep",10 :"Oct",11 :"Nov",12 :"Dec"}
donations['mmm'] = donations['month'].map(month_dict)

temp = donations['year'].value_counts().reset_index().sort_values('index')

data = [go.Scatter(
            x=temp['index'],
            y=temp['year'])]

layout = go.Layout(
    title="# of donations received in each year"
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
temp = donations.groupby('year').mean().reset_index()[['year', 'Donation Amount']]

data = [go.Bar(
            y=temp['Donation Amount'],
            x=temp['year'],
            marker = dict(color = 'rgba(1, 140, 221, 1)'),
            width=0.4
)]

layout = dict(
    title='Average donation amount',     
    width=800,
    height=400,
    paper_bgcolor="#f9f9f9",
    plot_bgcolor="#f9f9f9"
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='horizontal-bar')
temp = donations.groupby('mmm').mean().reset_index().sort_values('month')[['mmm', 'Donation Amount']]

data = [go.Bar(
            y=temp['Donation Amount'],
            x=temp['mmm'],
            marker = dict(color = 'rgba(1, 140, 221, 1)'),
            width=0.4
)]

layout = dict(
    title='Average donation amount in each month',     
    width=800,
    height=400,
    paper_bgcolor="#f9f9f9",
    plot_bgcolor="#f9f9f9"
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='horizontal-bar')