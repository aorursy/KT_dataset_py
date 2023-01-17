import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # ploting
import seaborn as sns # ploting

import re # regular expression

import plotly.plotly as py1
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools

import os # system
print(os.listdir("../input"))
resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False)
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False, warn_bad_lines=False)
donors = pd.read_csv('../input/Donors.csv', error_bad_lines=False, warn_bad_lines=False)
donations = pd.read_csv('../input/Donations.csv', error_bad_lines=False, warn_bad_lines=False)
teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False, warn_bad_lines=False)
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False)
print(resources.shape)
print(schools.shape)
print(donors.shape)
print(donations.shape)
print(teachers.shape)
print(projects.shape)
resources.head()
schools.head()
donors.head()
donations.head()
teachers.head()
projects.head()
donor_state_counts = donors['Donor State'].value_counts()
donor_state_counts_sorted = donor_state_counts.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12,7.5))
sns.barplot(donor_state_counts_sorted.values, donor_state_counts_sorted.index, ax=ax)
ax.set(xlabel= 'Number of Donors', ylabel = 'State', title = "Number of Donors by State")
plt.show()
fig, ax = plt.subplots(figsize=(8,5))
plt.pie(donor_state_counts_sorted.values, labels=donor_state_counts_sorted.index, autopct='%1.1f%%', shadow=True)
ax.set(title = "Number of Donors by State")
donor_state_counts_frame = donor_state_counts.to_frame()
donor_state_counts_frame.reset_index(inplace=True)
donor_state_counts_frame = donor_state_counts_frame.rename(columns = {'index':'Donor State', 'Donor State':'#Donors'})

for col in donor_state_counts_frame.columns:
    donor_state_counts_frame[col] = donor_state_counts_frame[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

donor_state_counts_frame['Text'] = donor_state_counts_frame['Donor State'] + '<br>' +\
    '#Donors '+ donor_state_counts_frame['#Donors']
    
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

donor_state_counts_frame['Code'] = donor_state_counts_frame['Donor State'].map(state_codes)  

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = donor_state_counts_frame['Code'],
        z = donor_state_counts_frame['#Donors'].astype(float),
        locationmode = 'USA-states',
        text = donor_state_counts_frame['Text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Number of Donors")
        ) ]

layout = dict(
        title = 'Number of Donors by State<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )

fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )
donor_city_counts = donors['Donor City'].value_counts()
donor_city_counts_top = donor_city_counts.sort_values(ascending=False)[:20]

fig, ax = plt.subplots(figsize=(12,7.5))
sns.barplot(donor_city_counts_top.values, donor_city_counts_top.index, ax=ax)
ax.set(xlabel= 'Number of Donors', ylabel = 'City', title = "Top 20 cities with most donors")
plt.show()
donor_teacher_counts = donors['Donor Is Teacher'].value_counts()
fig, ax = plt.subplots(figsize=(8,5))
plt.pie(donor_teacher_counts.values, labels=donor_teacher_counts.index, autopct='%1.1f%%', shadow=True)
ax.set(title = "Teacher distribution among donors")
donors_donations = pd.merge(donors, donations, on='Donor ID')
donors_donations.head()
donations_state_sum = donors_donations.groupby('Donor State')['Donation Amount'].sum()
donations_state_sum_sorted = donations_state_sum.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12,7.5))
sns.barplot(donations_state_sum_sorted.values, donations_state_sum_sorted.index, ax=ax)
ax.set(xlabel= 'Donation Amount', ylabel = 'State', title = "Donation Amount by State")
plt.show()
donations_city_sum = donors_donations.groupby('Donor City')['Donation Amount'].sum()
donations_city_sum_sorted = donations_city_sum.sort_values(ascending=False)[:20]
fig, ax = plt.subplots(figsize=(12,7.5))
sns.barplot(donations_city_sum_sorted.values, donations_city_sum_sorted.index, ax=ax)
ax.set(xlabel= 'Number of Donors', ylabel = 'City', title = "Top 20 cities with most donors")
plt.show()
donors_donations_optional = donors_donations['Donation Included Optional Donation'].value_counts()
fig, ax = plt.subplots(figsize=(8,5))
plt.pie(donors_donations_optional.values, labels=donors_donations_optional.index, autopct='%1.1f%%', shadow=True)
ax.set(title = "Optional Donations")
project_type_counts = projects['Project Type'].value_counts()
fig, ax = plt.subplots(figsize=(8,5))
plt.pie(project_type_counts.values, labels=project_type_counts.index, autopct='%1.1f%%', shadow=True)
ax.set(title = "Number of Donors by State")
project_resource_category_counts = projects['Project Resource Category'].value_counts()
project_resource_category_counts_top = project_resource_category_counts.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12,7.5))
sns.barplot(project_resource_category_counts_top.values, project_resource_category_counts_top.index, ax=ax)
ax.set(xlabel= 'Number of Projects', ylabel = 'Project Resource Category')
plt.show()
projects_dropna = projects.dropna(axis=0, subset=['Project Cost'], how='any')
projects_dropna.loc[:, 'Project Cost'] = projects_dropna.loc[:,'Project Cost'].apply(lambda x: float(re.sub(r'[^\d\.]', '', x)))

project_resource_category_cost = projects_dropna.groupby('Project Resource Category')['Project Cost'].sum()
project_resource_category_cost_top = project_resource_category_cost.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12,7.5))
sns.barplot(project_resource_category_cost_top.values, project_resource_category_cost_top.index, ax=ax)
ax.set(xlabel= 'Project Cost', ylabel = 'Project Resource Category')
plt.show()
project_status_counts = projects['Project Current Status'].value_counts()
fig, ax = plt.subplots(figsize=(8,5))
plt.pie(project_status_counts.values, labels=project_status_counts.index, autopct='%1.1f%%', shadow=True)
is_outlier = np.abs(resources['Resource Unit Price'] -resources['Resource Unit Price'].mean()) > (3 * resources['Resource Unit Price'].std())
resources_normal = resources[~is_outlier]

fig, ax = plt.subplots(figsize=(12,7.5))
resources_normal['Resource Unit Price'].hist(bins = 20, ax = ax)
ax.set(xlabel= 'Resource Unit Price', ylabel = 'Number of Resource Units', title = "Resource Unit Price Distribution")
plt.show()
resources_vendor_counts = resources['Resource Vendor Name'].value_counts()
resources_vendor_counts_sorted = resources_vendor_counts.sort_values(ascending=False)[:100]

fig, ax = plt.subplots(figsize=(12,7.5))
sns.barplot(resources_vendor_counts_sorted.values, resources_vendor_counts_sorted.index, ax=ax)
ax.set(xlabel= 'Donation Amount', ylabel = 'State', title = "Donation Amount by State")
plt.show()
school_state_count = schools['School State'].value_counts()
school_state_count_top = school_state_count.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12,7.5))
sns.barplot(school_state_count_top.values, school_state_count_top.index, ax=ax)
plt.show()
school_metro_count = schools['School Metro Type'].value_counts()
school_metro_count_top = school_metro_count.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(school_metro_count_top.index, school_metro_count_top.values, ax=ax)
plt.show()
teachers_growth = teachers['Teacher First Project Posted Date'].value_counts()
teachers_growth_cum = teachers_growth.sort_index(ascending=True).cumsum()
teachers_growth_cum.index = pd.to_datetime(teachers_growth_cum.index)

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(teachers_growth_cum)
ax.set(xlabel= 'Year', ylabel = 'Number of techer', title = "Number of teacher with posted projects")
plt.show()
teachers_projects = pd.merge(teachers, projects, on='Teacher ID')

teachers_projects_counts = teachers_projects['Teacher ID'].value_counts()
teachers_projects_counts = teachers_projects_counts.to_frame() 
teachers_projects_counts.reset_index(inplace=True)
teachers_projects_counts = teachers_projects_counts.rename(columns = {'index':'Teacher ID', 'Teacher ID':'#Projects'})
is_outlier = np.abs(teachers_projects_counts['#Projects'] -teachers_projects_counts['#Projects'].mean()) > (3 * teachers_projects_counts['#Projects'].std())
teachers_projects_counts_normal = teachers_projects_counts[~is_outlier]

fig, ax = plt.subplots(figsize=(12,7.5))
teachers_projects_counts_normal['#Projects'].hist(bins = 20, ax = ax)
ax.set(xlabel= 'Number of Projects', ylabel = 'Number of Teachers', title = "Number of posted projects per Teacher")
plt.show()