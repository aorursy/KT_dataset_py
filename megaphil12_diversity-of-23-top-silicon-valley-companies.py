# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Interactive Data Visualization with plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Reveal_EEO1_for_2016.csv')
df.head()
df.info()
# Drop any null and 'na' in count column
df.dropna(inplace=True)
df = df[df['count']!='na']

# Change count column to int32
df['count'] = df['count'].astype(dtype='int32')

# I like using "Ethnicity" rather than "Race" in this context, so I'm going to rename that column
df.rename(columns={'race':'ethnicity'}, inplace=True)
all_sums = []

# Get totals for each company through "Totals" row in "job_category" column and summing them up
for comp in df.company.unique():
    s = df[(df.company==comp) & (df.job_category=='Totals')]['count'].sum()
    all_sums.append(s)

# Create new DataFrame for with totals for each company
sums = pd.DataFrame(columns=['Company', 'Total Employees'])
sums.Company = df.company.unique()
sums['Total Employees'] = all_sums

# Setup Data for Plotly chart
data = [
    go.Bar(
        x = sums['Total Employees'],
        y = sums.Company,
        orientation = 'h'
    )
]

# Setup Layout for Plotly chart
layout = go.Layout(
    title = 'Total Employees by Company',
    xaxis = {
        'title': '# of Employees'
    },
    yaxis = {
        'autorange': 'reversed',
        'title': 'Company'
    })

# Show Plotly Figure
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# Group by Company and Ethnicity, grab only the 'Totals' job_category rows and sum up the count for each ethnicity group into a new DataFrame
ethnicity = df[df.job_category=='Totals'].groupby(['company','ethnicity'], sort=False)['count'].sum().reset_index()

# Calculate percentage for each ethnicity group with the respective company total into a temporary DataFrame
tmp = ethnicity.apply(lambda row: (row.loc['count'] / sums[sums.Company==row.company]['Total Employees']) * 100, axis=1)

# Apply method created separate columns for each company so compress all columns into one
for i in range(1, len(tmp.columns)):
    tmp[0] = tmp[0].dropna().append(tmp[i].dropna()).reset_index(drop=True)
    
# Join percentage column with ethinicity dataframe
ethnicity = ethnicity.join(tmp[0])
ethnicity.rename(columns={0:'percentage'}, inplace=True)
ethnicity.head()
# Setup data for Plotly chart for each company into a list
trace = []
list_of_ethnicities = ethnicity.ethnicity.unique()

for comp in sums.Company:
    trace.append(go.Bar(
                    x = list_of_ethnicities,
                    y = ethnicity[ethnicity.company==comp]['percentage'],
                    name = comp
    ))

# Setup layout for Plotly Chart
layout = go.Layout(barmode = 'group',
                   title = 'Diversity Proportion Breakdown by Company',
                   xaxis = {
                       'title': 'Ethnicities'
                   },
                   yaxis = {
                       'title': 'Percentage %'
                   },
                   legend = {
                       'xanchor': 'auto'
                   })

# Show Plotly Figure
fig = go.Figure(data=trace, layout=layout)
iplot(fig)
for comp in sums.Company:

    # Setup data for each Plotly chart for each company
    data = [
        go.Pie(
            values = ethnicity[ethnicity.company==comp]['percentage'],
            labels = list_of_ethnicities,
            name = comp,
            hoverinfo = "label"
        )
    ]

    # Setup layout for Plotly Charts for each company
    layout = go.Layout(
        title = 'Diversity Breakdown for ' + comp
    )

    # Show Plotly Figure
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
gender = df[df.job_category=='Totals'].groupby(['company','gender'], sort=False)['count'].sum().reset_index()

# Calculate percentage for each ethnicity group with the respective company total
tmp = gender.apply(lambda row: (row.loc['count'] / sums[sums.Company==row.company]['Total Employees']) * 100, axis=1)

# Apply method created separate columns for each company so compress all columns into one
for i in range(1, len(tmp.columns)):
    tmp[0] = tmp[0].dropna().append(tmp[i].dropna()).reset_index(drop=True)
    
# Join percentage column with ethinicity dataframe
gender = gender.join(tmp[0])
gender.rename(columns={0:'percentage'}, inplace=True)

list_of_companies = sums.Company.unique()
data = [go.Bar(
              x = list_of_companies,
              y = gender[gender.gender=='male']['percentage'],
              marker = {
                  'color': 'rgb(55, 83, 109)'
              },
              name = 'Male'
        ),
        go.Bar(
              x = list_of_companies,
              y = gender[gender.gender=='female']['percentage'],
              marker = {
                  'color': 'rgb(26, 118, 255)'
              },
              name = 'Female'
        )]

layout = go.Layout(barmode = 'group',
                   title = 'Gender Percentage by Company',
                   xaxis = {
                       'title': 'Gender'
                   },
                   yaxis = {
                       'title': 'Percentage %',
                       'hoverformat': '.2f'
                   },
                   legend = {
                       'xanchor': 'auto'
                   })

fig = go.Figure(data=data, layout=layout)
iplot(fig)