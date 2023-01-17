# General libraries
import os
import re
import urllib
from collections import Counter
from itertools import cycle, islice
import warnings

# Data analysis and preparation libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.offline import iplot
import plotly.graph_objs as go
import cufflinks as cf
from statsmodels.tsa.seasonal import seasonal_decompose

# Set configuration
warnings.filterwarnings("ignore")
cf.go_offline()
%%time
input_dir = '../input/io/'
#input_dir = './io/'
all_projects = pd.read_csv(input_dir + 'Projects.csv')
all_donations = pd.read_csv(input_dir + 'Donations.csv')
all_donors = pd.read_csv(input_dir + 'Donors.csv')
all_schools = pd.read_csv(input_dir + 'Schools.csv')
all_resources = pd.read_csv(input_dir + 'Resources.csv')
all_teachers = pd.read_csv(input_dir + 'Teachers.csv')

# Pull the state codes from another dataset and load in a dataframe
#filename = './us-state-county-name-codes/states.csv'
filename = '../input/us-state-county-name-codes/states.csv'
df_statecode = pd.read_csv(filename, delimiter=',')

print('Done loading files')
# Describe dataset
display(all_projects.head(5))
print('Overall {} Rows and {} columns'.format(all_projects.shape[0], all_projects.shape[1]))
display(all_projects.nunique())
proj_stat = all_projects['Project Current Status'].value_counts().sort_values().to_frame()
proj_stat['count'] = proj_stat['Project Current Status']
proj_stat['Project Current Status'] = proj_stat.index
proj_stat.iplot(kind='pie', labels='Project Current Status', values='count',
                title = 'Funding Status of Projects',
                pull=.1,
                hole=.4,
                textposition='outside',
                textinfo='value+percent')

proj_type = all_projects['Project Type'].value_counts().sort_values().to_frame()
proj_type['count'] = proj_type['Project Type']
proj_type['Project Type'] = proj_type.index
proj_type.iplot(kind='pie',labels='Project Type',values='count',
                title = 'Types of Projects',
                pull=.1,
                hole=.4,
                textposition='outside',
                textinfo='value+percent',
               )
# Monthly trends of project funding over the years
proj_posted = all_projects['Project Posted Date'].str.slice(start=0, stop=7)
proj_posted = proj_posted.value_counts().sort_index()

proj_funded = all_projects['Project Fully Funded Date'].str.slice(start=0, stop=7)
proj_funded = proj_funded.value_counts().sort_index()

proj = pd.concat([proj_posted, proj_funded], axis=1)
proj[(proj['Project Posted Date']) < (proj['Project Fully Funded Date'])]

proj.iplot([{'x': proj.index, 'y': proj[col], 'mode': 'line','name': col}
            for col in proj.columns],
           xTitle='Year and Month the Project was Posted / Funded',
           yTitle = "Total Count",
           title ="Monthly Observerd Volume of Projects Over Time"
          )
# proj.iplot(kind = 'scatter', xTitle='Year and Month the Project was Posted',  yTitle = "Total Count", 
#                 title ="Volume of Monthly Projects Over Time", width=5)

# Let's decomponse the time series
proj.index = pd.to_datetime(proj.index )
decomp_post = seasonal_decompose(proj['Project Posted Date'].to_frame(), model='multiplicative')
decomp_fund = seasonal_decompose(proj['Project Fully Funded Date'].to_frame(), model='multiplicative')
trend = decomp_post.trend.join(decomp_fund.trend)
seasonal = decomp_post.seasonal.join(decomp_fund.seasonal)
resid = decomp_post.resid.join(decomp_fund.resid)

trend.iplot([{'x': trend.index, 'y': trend[col], 'mode': 'line','name': col}
            for col in trend.columns],
           xTitle='Year and Month the Project was Posted / Funded',
           yTitle = "Total Count",
           title ="Decomposed Trend of Projects Over Time"
          )
        
seasonal.iplot([{'x': seasonal.index, 'y': seasonal[col], 'mode': 'line','name': col}
            for col in seasonal.columns],
           xTitle='Year and Month the Project was Posted / Funded',
           yTitle = "Total Count",
           title ="Decomposed Seasonality of Projects Over Time"
          )
        
resid.iplot([{'x': resid.index, 'y': resid[col], 'mode': 'line','name': col}
            for col in resid.columns],
           xTitle='Year and Month the Project was Posted / Funded',
           yTitle = "Total Count",
           title ="Decomposed Randomness in Project Volume Over Time"
          )
# Let's analyze the project cost
print('Lowest project cost - {}'.format(all_projects['Project Cost'].min()))
print('Highest project cost - {}'.format(all_projects['Project Cost'].max()))

# The cost of classroom requests vary a lot; between $35 to $255K
# Now split the data in some buckets such that it represents a normal distribution
# This is of course a simulated distribution, but it does provides some perspective on project costs.
custom_bucket = [0, 179, 299, 999, 2500, 100000]
custom_bucket_label = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
proj_cost = pd.cut(all_projects['Project Cost'], custom_bucket, labels=custom_bucket_label)
proj_cost = proj_cost.value_counts().sort_index()

proj_cost.iplot(kind='bar', xTitle = 'Project Cost', yTitle = "Project Count", 
                title = 'Distribution on Project Cost', color='violet')
# Project Category and Subcategory are stacked columns. A classromm request can span across multiple categories.
# I will start by exploding the columns and then analyze the trend over the years
def stack_attributes(df, target_column, separator=', '):
    df = df.dropna(subset=[target_column])
    df = (df.set_index(df.columns.drop(target_column,1).tolist())
          [target_column].str.split(separator, expand=True)
          .stack().str.strip()
          .reset_index()
          .rename(columns={0:target_column})
          .loc[:, df.columns])
    df = (df.groupby([target_column, 'Project Posted Date'])
          .size()
          .to_frame(name ='Count')
          .reset_index())
    
    return df

def plot_trend(df, target_column, chartType=go.Scatter,
              datecol='Project Posted Date', 
              ytitle='Number of relevant classroom requests'):
    trend = []
    for category in list(df[target_column].unique()):
        temp = chartType(
            x = df[df[target_column]==category][datecol],
            y = df[df[target_column]==category]['Count'],
            name=category
        )
        trend.append(temp)
    
    layout = go.Layout(
        title = 'Trend of ' + target_column,
        xaxis=dict(
            title='Year & Month',
            zeroline=False,
        ),
        yaxis=dict(
            title=ytitle,
        ),
    )
    
    fig = go.Figure(data=trend, layout=layout)
    iplot(fig)
    
proj = all_projects[['Project Subject Category Tree',
                     'Project Subject Subcategory Tree',
                     'Project Resource Category',
                     'Project Grade Level Category',
                     'Project Posted Date']].copy()
proj['Project Posted Date'] = all_projects['Project Posted Date'].str.slice(start=0, stop=4)

proj_cat = stack_attributes (proj, 'Project Subject Category Tree')
proj_sub_cat = stack_attributes (proj, 'Project Subject Subcategory Tree')
proj_res_cat = (proj.groupby(['Project Resource Category', 'Project Posted Date'])
                .size()
                .to_frame(name ='Count')
                .reset_index())
proj_grade_cat = (proj.groupby(['Project Grade Level Category', 'Project Posted Date'])
                .size()
                .to_frame(name ='Count')
                .reset_index())

plot_trend(proj_cat, 'Project Subject Category Tree')
plot_trend(proj_sub_cat, 'Project Subject Subcategory Tree')
plot_trend(proj_res_cat, 'Project Resource Category')
plot_trend(proj_grade_cat, 'Project Grade Level Category', chartType=go.Bar)
# Describe donation dataset
display(all_donations.head(5))
display('Overall {} Rows and {} columns'.format(all_donations.shape[0], all_donations.shape[1]))
display(all_donations.nunique())
# Describe donor dataset
display(all_donors.head(5))
print('Overall {} Rows and {} columns'.format(all_donors.shape[0], all_donors.shape[1]))
display(all_donors.nunique())
# Let's analyze the donation amoung
print('Lowest project cost - {}'.format(all_donations['Donation Amount'].min()))
print('Highest project cost - {}'.format(all_donations['Donation Amount'].max()))

# The Donation amount vary a lot; between $0.01 to $60,000
# Now the goal is to split the data in some buckets such that it represents a normal distribution
# This is of course a simulated distribution, but it will help provide some perspective on donation amount.
custom_bucket = [0, 0.99, 9.99, 99.99, 999.99, 1000000]
custom_bucket_label = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
don_amnt = all_donations[['Donation Amount', 'Donation Received Date']]
don_amnt['Donation Amount'] = pd.cut(don_amnt['Donation Amount'], custom_bucket, labels=custom_bucket_label)
don_amnt['Donation Received Date'] = don_amnt['Donation Received Date'].str.slice(start=0, stop=4)
don_amn = don_amnt['Donation Amount'].value_counts().sort_index()

don_amn.iplot(kind='bar', xTitle = 'Donation Amount', yTitle = 'Donation Count', 
                title = 'Simulated Distribution on Donation Amount')

don_amnt = (don_amnt.groupby(['Donation Amount', 'Donation Received Date'])
                .size()
                .to_frame(name ='Count')
                .reset_index())

plot_trend(don_amnt, 'Donation Amount', chartType=go.Scatter, datecol='Donation Received Date', 
           ytitle='Number of donations')
# Let's find number of donations per project
custom_bucket = [0, 1, 5, 10, 20, 1000000]
custom_bucket_label = ['Single Donor', '1-5 Donors', '6-10 Donors', '11-20 Donors', 'More than 20 Donors']
num_of_don = all_donations['Project ID'].value_counts().to_frame(name='Donation Count').reset_index()
num_of_don['Donation Cnt'] = pd.cut(num_of_don['Donation Count'], custom_bucket, labels=custom_bucket_label)
num_of_don = num_of_don['Donation Cnt'].value_counts().sort_index()

num_of_don.iplot(kind='bar', xTitle = 'Number of Donors', yTitle = 'Number of Projects', 
                title = 'Distribution on Number of Donors and Project Count')

# Let find how many time donors are donating to the classrooms
custom_bucket = [0, 1, 5, 10, 100, 1000000]
custom_bucket_label = ['Donated Once', 'Donated 1 - 5 times', 'Donated 6 - 10 time',
                       'Donated 11 & 100 times', 'Donated more than 100 times']
don_repeat = all_donations['Donor ID'].value_counts().to_frame(name='Donation Count')
display ('Maximum Repeat donations by a Donor - {}'.format(don_repeat['Donation Count'].max()))
display ('Minimum Repeat donations by a Donor - {}'.format(don_repeat['Donation Count'].min()))

don_repe = don_repeat.copy()
don_repe = don_repe['Donation Count'].value_counts().to_frame(name='Number of Donors')
don_repe['Number of Donations'] = don_repe.index
don_repe['Number of Donations'] = pd.cut(don_repe['Number of Donations'], 
                                         custom_bucket, labels=custom_bucket_label)
don_repe = {
  'data': [
    {
      'values': don_repe['Number of Donors'],
      'labels': don_repe['Number of Donations'],
      'name': 'Number of Donations',
      'hoverinfo':'name',
      'hole': .4,
      'pull': .01,
      'type': 'pie'
    }],
  'layout': {'title':'Share of number of Donations'}
}
iplot(don_repe, filename='donut')
# Lets find out the donor trends across state. 
# Let's start by finding out number of donors in each state
donor_per_state = all_donors.groupby('Donor State').size().to_frame(name='Total Donors in State')

donation_state = don_repeat.copy()
donation_state['Donor ID'] = donation_state.index
donation_state = donation_state.merge(all_donors, how='inner', on='Donor ID')

# This will give the repeat donors now
repeat_donors = donation_state[donation_state['Donation Count'] > 1]

# Let's find Number of donations per state and repeat donations per state 
repeat_donors_cnt = repeat_donors.groupby('Donor State').size().to_frame('Number of Repeat Donors')
repeat_donors_cnt = repeat_donors_cnt.merge(donor_per_state, left_index=True, right_index=True)
repeat_donors_cnt['Percentage of Repeat Donors'] = (repeat_donors_cnt['Number of Repeat Donors'] 
                                                * 100 / repeat_donors_cnt['Total Donors in State'])

(repeat_donors_cnt['Total Donors in State']
     .sort_values(ascending=False)
     .iplot(kind='bar', xTitle = 'States', yTitle = "Number of Donors", 
                title = 'Distribution on Donors Across State', color='Green'))

(repeat_donors_cnt['Percentage of Repeat Donors']
     .sort_values(ascending=False)
     .iplot(kind='bar', xTitle = 'States Cost', yTitle = "Number of Donors", 
                title = 'Distribution Repeat Donors Across States', color='Red'))
# Let's analyze if the city/state of classroom request impacts the donation
# Get the School state and the State of Donors, associated with classroom requests
don_info = all_donations[['Donor ID', 'Project ID']].copy()
don_info = don_info.merge(all_donors[['Donor ID', 'Donor State']], on='Donor ID', how='inner')
don_info = don_info.merge(all_projects[['Project ID', 'School ID']], on='Project ID', how='inner')
don_info = don_info.merge(all_schools[['School ID', 'School State']], on='School ID', how='inner')
don_info['In State Donation'] = np.where((don_info['School State']) == (don_info['Donor State']), 'Yes', 'No')

in_state = (don_info['In State Donation'].value_counts()
            .sort_values()
            .to_frame(name='Count')
            .reset_index()
           )
in_state['In State Donation'] = in_state.index
in_state.iplot(kind='pie',labels='In State Donation',values='Count',
                title = 'Are the Donors Donating Within the States They Live In?',
                pull=.01,
                hole=.01,
                colorscale='set3',
                textposition='outside',
                textinfo='value+percent')


# Let's see how this trend varies across states
in_stat = (don_info.groupby(['Donor State', 'In State Donation'])
                .size()
                .to_frame(name ='Count')
                .reset_index())
in_stat = in_stat.pivot(index='Donor State', columns='In State Donation', values='Count')
in_stat['In-State Donation Ratio'] = in_stat['Yes'] * 100 / (in_stat['Yes'] + in_stat['No'])

# (in_stat['In-State Donation Ratio']
#      .sort_values(ascending=True)
#      .iplot(kind='bar', xTitle = 'States', yTitle = "Pecentate of donations made within state", 
#             title = 'Percentage of in-state donations across states',
#             colorscale='-ylorrd', theme = 'pearl'))
in_stat['State'] = in_stat.index
temp = in_stat.merge(df_statecode, on='State')
temp.head()

data = [ dict(
        type='choropleth',
        autocolorscale = False,
        locations = temp['Abbreviation'],
        z = in_stat['In-State Donation Ratio'].astype(float),
        locationmode = 'USA-states',
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict()
        ) ]

layout = dict(
        title = 'Percentage of Donations Within Home State',
        geo = dict(
            scope='usa',
            projection=dict(type='albers usa'))
             )
    
fig = dict(data=data, layout=layout )
iplot(fig, filename='d3-cloropleth-map', validate=False)
# Let's see what percentage of the donors are teachers
teachers_in_donors = round((all_donors[all_donors['Donor Is Teacher'] == 'Yes'].shape[0]) 
                      * 100 / all_donors.shape[0])
teachers_in_donors = {"Teachers": teachers_in_donors, "Others": 100 - teachers_in_donors}

tchr_repeat_donors = round((repeat_donors[repeat_donors['Donor Is Teacher'] == 'Yes'].shape[0]) 
                      * 100 / repeat_donors.shape[0])
tchr_repeat_donors = {'Teachers': tchr_repeat_donors, 'Others': 100 - tchr_repeat_donors}


tchr = {
  'data': [
    {
        'values': list(teachers_in_donors.values()),
        'labels': list(teachers_in_donors.keys()),
        'domain': {"x": [0, .48]},
        'marker': {'colors': ['rgb(124, 173, 100)', 'rgb(215, 112, 100)']},
        'hoverinfo':'labels+percentage',
        'name': 'Percentage of Teachers Among Donors',
        'hole': .1,
        'pull': .0,
        'type': 'pie'
    },
    {
        'values': list(tchr_repeat_donors.values()),
        'labels': list(tchr_repeat_donors.keys()),
        'domain': {"x": [.52, 1]},
        'marker': {'colors': ['rgb(124, 173, 100)', 'rgb(200, 200, 100)']},
        'hoverinfo':'labels+percentage',
        'name': 'Percentage of Teachers Among Repeat Donors',
        'hole': .1,
        'pull': .0,
        'type': 'pie'
    }
  ],
    'layout': {'title':'Are Teachers Donating More That Others?'}
}

iplot(tchr, filename='styled_pie_chart')
# Describe teachers dataset
display(all_teachers.head(5))
display('Overall {} Rows and {} columns'.format(all_teachers.shape[0], all_teachers.shape[1]))
display(all_teachers.nunique())
# Describe resources dataset
display(all_resources.head(5))
display('Overall {} Rows and {} columns'.format(all_resources.shape[0], all_resources.shape[1]))
display(all_resources.nunique())
# Let's find the gender ratio of the teachers
tech_gend = all_teachers[['Teacher Prefix', 'Teacher ID', 'Teacher First Project Posted Date']].copy()
tech_gend['Gender'] = tech_gend['Teacher Prefix']
tech_gend.loc[tech_gend['Gender'] == 'Mrs.', 'Gender'] = 'Female'
tech_gend.loc[tech_gend['Gender'] == 'Ms.', 'Gender'] = 'Female'
tech_gend.loc[tech_gend['Gender'] == 'Mr.', 'Gender'] = 'Male'
tech_gend.loc[tech_gend['Gender'] == 'Teacher', 'Gender'] = 'Neutral'
tech_gend.loc[tech_gend['Gender'] == 'Dr.', 'Gender'] = 'Neutral'
tech_gend.loc[tech_gend['Gender'] == 'Mx.', 'Gender'] = 'Neutral'

gen = tech_gend.groupby('Gender').size().to_frame(name='Count')
gen['Gender'] = gen.index

# Average number of classroom request by gender
tech_gend_proj = tech_gend.merge(all_projects[['Teacher ID', 'Project Cost']], on='Teacher ID')
tech_gend_proj = tech_gend_proj.groupby('Gender').size().to_frame('Count')
tech_gend_proj['Gender'] = tech_gend_proj.index

tchr = {
  'data': [
    {
        'values': list(gen['Count']),
        'labels': list(gen['Gender']),
        'domain': {"x": [0, .48]},
        'marker': {'colors': ['rgb(124, 173, 100)', 'rgb(215, 112, 100)']},
        'hoverinfo':'labels+percentage',
        'name': 'Spread of Teachers Based on Gender',
        'hole': .70,
        'pull': .0,
        'type': 'pie'
    },
    {
        'values': list(tech_gend_proj['Count']),
        'labels': list(tech_gend_proj['Gender']),
        'domain': {"x": [.52, 1]},
        'marker': {'colors': ['rgb(124, 173, 200)', 'rgb(200, 200, 200)']},
        'hoverinfo':'labels+percentage',
        'name': 'Number of Clasroom Requests Based on Gender',
        'hole': .70,
        'pull': .0,
        'type': 'pie'
    }
  ],
    'layout': {'title':"Distribution Based on Gender"}
}
iplot(tchr, filename='styled_pie_chart')

# How many new teachers are joining DonorsChoose each month
tech_start_mnth = tech_gend['Teacher First Project Posted Date'].str.slice(start=0, stop=7)
tech_start_mnth = tech_start_mnth.value_counts().to_frame(name='Count').sort_index()
tech_start_mnth['Date'] = tech_start_mnth.index

trace = go.Scatter(
            x=tech_start_mnth.Date,
            y=tech_start_mnth.Count,
            name = "New Teachers Onboarding Every Month",
            line = dict(color = '#17BECF'),
            opacity = 0.8)

layout = dict(
    title = 'Monthly New Teachers Associating with DonorsChoose Over Time',
    xaxis = dict(
        title='Time Period',
    ),
    yaxis = dict(
        title='Number of Teachers',
    )
)

fig = dict(data=[trace], layout=layout)
iplot(fig, filename = 'time-series')

display(all_resources['Resource Item Name'].nunique())
display(all_resources['Resource Item Name'].value_counts().sort_values(ascending=False).head(10))
proj_proj = list(all_projects['Project ID'].unique())
proj_teacher = list(all_projects['Teacher ID'].unique())
proj_school = list(all_projects['School ID'].unique())

dontn_proj = list(all_donations['Project ID'].unique())
dontn_donor =list(all_donations['Donor ID'].unique())

donor_donor = list(all_donors['Donor ID'].unique())
resrcs_proj = list(all_resources['Project ID'].unique())
tech_teacher = list(all_teachers['Teacher ID'].unique())
sch_school = list(all_schools['School ID'].unique())

print('{} projects are there in total'.format(len(set(proj_proj))))
print('{} Teachers have classroom Projects but no records in Teachers dataset'
      .format(len(set(proj_teacher) - set(tech_teacher))))
print('{} Schools have a classroom request but no info in School dataset'
      .format(len(set(proj_school)-set(sch_school))))
print('{} Projects have no Resources associated with it'
      .format(len(set(proj_proj)-set(resrcs_proj))))
print('{} Projects have not received any Donoation'
      .format(len(set(proj_proj)-set(dontn_proj))))
print('{} people have donated to projects but missing in Donor dataset'
      .format(len(set(dontn_donor)-set(donor_donor))))
