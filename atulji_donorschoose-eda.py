# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import plotly.plotly as py1
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools

import cufflinks as cf
cf.go_offline()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
resources = pd.read_csv("../input/Resources.csv")
resources.head()
temp = resources["Resource Item Name"].value_counts()
data = [go.Bar(
            x = temp.index.values[0:15],
            y = temp.values[0:15],
            marker=dict(color="blue"),
)]
layout = go.Layout(
    title='Top Resource Requests',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename="basic-bar")
total_resources = resources.shape[0]
vendor_names = temp.index.values[0:15]
#print(vendor_names)
temp = temp.head(15)
vendor_percent = [(x) for x in temp]
#temp.head(10)
temp = temp.apply(lambda x:(x/total_resources)*100)
temp.head()
temp = resources["Resource Vendor Name"].value_counts()
data = [go.Bar(
            x = temp.index.values[0:15],
            y = temp.values[0:15],
            marker=dict(color="light blue"),
)]
layout = go.Layout(
    title='Top Resource Vendors ',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename="basic-bar")
total_resources = resources.shape[0]
temp = temp.apply(lambda x:(x/total_resources)*100)
label_index = list(temp.index[0:15])
label_values = list(temp.values[0:15])
label_index.append("Other vendors")
label_values.append(sum(temp[15:]))

df = pd.DataFrame({'labels': label_index,
                   'values': label_values})
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Distribution of Top vendors')
teachers = pd.read_csv("../input/Teachers.csv")
teachers.head()
temp = teachers["Teacher Prefix"].value_counts()

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Distribution of Teacher Prefix')

from datetime import datetime
teachers["date_year"] = teachers['Teacher First Project Posted Date'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d").year)
temp = teachers.groupby("date_year").count()
temp = temp["Teacher First Project Posted Date"]

fig  = {
    'data':[ {
        'x': temp.index,
        'y': temp.values
    }
    ],
    'layout' : {
       'title' : 'Trends of Number of Teachers Posted  their First Project (2002 - 2018)',
        'xaxis': {'title': 'Year'},
        'yaxis': {'title': "Number of Teachers Posted First Project"}
    }
    
    
}
#fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename="basic-line")
schools = pd.read_csv('../input/Schools.csv',error_bad_lines=False)
schools.head()
temp = schools['School Metro Type'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Distribution of School Metro Type')
projects = pd.read_csv('../input/Projects.csv')
#projects.head()
projects_schools = projects.merge(schools, on='School ID', how='inner')
cnt_srs = projects_schools['School City'].value_counts().head(20)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Distribution of School cities',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="CitySchools")
school_count = schools['School State'].value_counts().reset_index()
school_count.columns = ['state', 'schools']
for col in school_count.columns:
    school_count[col] = school_count[col].astype(str)
school_count['text'] = school_count['state'] + '<br>' + '# of schools: ' + school_count['schools']
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

school_count['code'] = school_count['state'].map(state_codes) 
# https://plot.ly/python/choropleth-maps/
data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = school_count['code'], # The variable identifying state
        z = school_count['schools'].astype(float), # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = school_count['text'], # Text to show when mouse hovers on each state
        colorbar = dict(  
            title = "# of Schools")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = 'Number of schools in different states<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
iplot(fig)
schools.groupby('School Metro Type')['School Percentage Free Lunch'].describe()
donors = pd.read_csv("../input/Donors.csv")
donors.head()
donor_is_teacher = donors['Donor Is Teacher'].value_counts()
df = pd.DataFrame({'labels': donor_is_teacher.index,
                   'values': donor_is_teacher.values })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Not Teacher vs Teacher')
temp =  donors['Donor City'].value_counts()
data = [go.Bar(
            x = temp.index.values[0:15],
            y = temp.values[0:15],
            marker=dict(color="light blue"),
)]
layout = go.Layout(
    title='Top No.of Donors from Cities',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename="basic-bar")
temp =  donors['Donor State'].value_counts()
data = [go.Bar(
            x = temp.index.values[0:15],
            y = temp.values[0:15],
            marker=dict(color="light blue"),
)]
layout = go.Layout(
    title='Top No.of Donors from States',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename="basic-bar")
temp = projects_schools['Project Subject Category Tree'].value_counts().head(30)
temp.iplot(kind='bar', xTitle = 'Project Subject Category', yTitle = "Count", title = 'Distribution of Project subject categories', color='green')
temp = projects_schools['Project Subject Subcategory Tree'].value_counts().head(50)
temp.iplot(kind='bar', xTitle = 'Project Subject Sub-Category', yTitle = "Count", title = 'Distribution of Project subject Sub-categories', color='blue')
temp = projects_schools['Project Resource Category'].value_counts().head(30)
temp.iplot(kind='bar', xTitle = 'Project Resource Category Name', yTitle = "Count", title = 'Distribution of Project Resource categories')
temp = projects['Project Grade Level Category'].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      "name": "Grade Level Category",
      #"hoverinfo":"label+percent+name",
      "hole": .5,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Distribution of Projects Grade Level Category",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Grade Level Categories",
                "x": 0.11,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
donations = pd.read_csv('../input/Donations.csv')
donations.head()
print('Minimum Donated Amount is: {0}$'.format(donations['Donation Amount'].min()))
print('Maximum Donated Amount is: {0}$'.format(donations['Donation Amount'].max()))
print('Average Donated Amount is: {0}$'.format(donations['Donation Amount'].mean()))
# Merge donation data with donor data 
donors_donations = donations.merge(donors, on='Donor ID', how='inner')
city_wise_donation = donors_donations.groupby('Donor City', as_index=False).agg({'Donation ID': 'count','Donation Amount':'sum'}).sort_index(by=['Donation Amount'],ascending=[False])
trace = go.Bar(
    y=city_wise_donation['Donation Amount'][:15],
    x=city_wise_donation['Donor City'][:15],
    marker=dict(
        color=city_wise_donation['Donation Amount'][:15][::-1],
        colorscale = 'reds',
        reversescale = True
    ),
)

layout = dict(
    title='Distribution Top 15 Donation Amount Cities wise',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="CityDonationAmount")
state_wise_donation = donors_donations.groupby('Donor State', as_index=False).agg({'Donation ID': 'count','Donation Amount':'sum'}).sort_index(by=['Donation Amount'],ascending=[False])
trace = go.Bar(
    y=state_wise_donation['Donation Amount'][:15],
    x=state_wise_donation['Donor State'][:15],
    marker=dict(
        color=state_wise_donation['Donation Amount'][:15][::-1],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Distribution Top 15 Donation Amount State wise',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="StateDonationAmount")
state_wise = donors_donations.groupby('Donor State', as_index=False).agg({'Donation ID': 'count','Donation Amount':'sum'})   
state_wise.columns = ["State","Donation_num", "Donation_sum"]
state_wise["Donation_avg"]=state_wise["Donation_sum"]/state_wise["Donation_num"]
del state_wise['Donation_num']
for col in state_wise.columns:
    state_wise[col] = state_wise[col].astype(str)
state_wise['text'] = state_wise['State'] + '<br>' +\
    'Average amount per donation: $' + state_wise['Donation_avg']+ '<br>' +\
    'Total donation amount:  $' + state_wise['Donation_sum']
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

state_wise['code'] = state_wise['State'].map(state_codes)

for col in state_wise.columns:
    state_wise[col] = state_wise[col].astype(str)
state_wise['text'] = state_wise['State'] + '<br>' +\
    'Average amount per donation: $' + state_wise['Donation_avg']+ '<br>' +\
    'Total donation amount:  $' + state_wise['Donation_sum']
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

state_wise['code'] = state_wise['State'].map(state_codes)  
# https://plot.ly/python/choropleth-maps/
data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = state_wise['code'], # The variable identifying state
        z = state_wise['Donation_sum'].astype(float), # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = state_wise['text'], # Text to show when mouse hovers on each state
        colorbar = dict(  
            title = "Donation in USD")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = 'Donations given by different States<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
iplot(fig)

donations_data=donations
donation_10 = donations_data[donations_data["Donation Amount"] <= 10].shape[0]
total= donations_data.shape[0]
donation_100 = donations_data[(donations_data["Donation Amount"] > 10) & (donations_data["Donation Amount"] <= 100)].shape[0] 
donation_1000 = donations_data[(donations_data["Donation Amount"] > 100) & (donations_data["Donation Amount"] <= 1000)].shape[0] 
donations_percent = {"less Than 10$":donation_10, "Between 10-100$":donation_100,"Between 100-1000$":donation_1000}
donation_amount = donations_data[(donations_data["Donation Amount"] > 1000) & (donations_data["Donation Amount"] <= 10000)].shape[0]
donations_percent["Between 1000-10000$"]=donation_amount
donation_amount = donations_data[(donations_data["Donation Amount"] > 10000) & (donations_data["Donation Amount"] <= 30000)].shape[0]
donations_percent["Between 10000-30000$"] = donation_amount
donation_amount = donations_data[(donations_data["Donation Amount"] > 30000) & (donations_data["Donation Amount"] <= 50000)].shape[0]
donations_percent["Between 30000-50000$"]=donation_amount
donation_amount = donations_data[(donations_data["Donation Amount"] > 50000)].shape[0]
donations_percent["More than 50000"]=donation_amount
total = donations_data.shape[0]
donations_percent
df = pd.Series(data=donations_percent)
#df.plot(kind="bar")
#print(df.head(10))
df = pd.DataFrame({'labels': list(donations_percent.keys()),
                   'values': list(donations_percent.values())
                  })
df.iplot(kind='pie',labels='labels',values='values', title='')
donations_merge_projects = donations.merge(projects,how="inner",on="Project ID")
temp = donations_merge_projects.groupby("Project Subject Category Tree").count()["Donation Amount"].plot(kind="Bar",figsize=(15,10),title="Donations Towards different Projects")
