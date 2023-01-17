# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #for visualisation 

import seaborn as sns #seaborn library

import plotly.graph_objects as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
degree = pd.read_csv('/kaggle/input/college-salaries/degrees-that-pay-back.csv')
region = pd.read_csv('/kaggle/input/college-salaries/salaries-by-region.csv')
college = pd.read_csv('/kaggle/input/college-salaries/salaries-by-college-type.csv')
degree.head(10)
degree.info()
degree.shape
college.head()
college.info()
region.head()
region.info()
region.columns
region['Starting Median Salary']=(region['Starting Median Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))

region['Mid-Career Median Salary']=(region['Mid-Career Median Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))

region['Mid-Career 10th Percentile Salary']=(region['Mid-Career 10th Percentile Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))

region['Mid-Career 25th Percentile Salary']=(region['Mid-Career 25th Percentile Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))

region['Mid-Career 75th Percentile Salary']=(region['Mid-Career 75th Percentile Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))

region['Mid-Career 90th Percentile Salary']=(region['Mid-Career 90th Percentile Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))
college['Starting Median Salary']=(college['Starting Median Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))

college['Mid-Career Median Salary']=(college['Mid-Career Median Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))

college['Mid-Career 10th Percentile Salary']=(college['Mid-Career 10th Percentile Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))

college['Mid-Career 25th Percentile Salary']=(college['Mid-Career 25th Percentile Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))

college['Mid-Career 75th Percentile Salary']=(college['Mid-Career 75th Percentile Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))

college['Mid-Career 90th Percentile Salary']=(college['Mid-Career 90th Percentile Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))
degree['Starting Median Salary']=(degree['Starting Median Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))

degree['Mid-Career Median Salary']=(degree['Mid-Career Median Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))

degree['Mid-Career 10th Percentile Salary']=(degree['Mid-Career 10th Percentile Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))

degree['Mid-Career 25th Percentile Salary']=(degree['Mid-Career 25th Percentile Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))

degree['Mid-Career 75th Percentile Salary']=(degree['Mid-Career 75th Percentile Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))

degree['Mid-Career 90th Percentile Salary']=(degree['Mid-Career 90th Percentile Salary'].replace( '[\$,)]','', regex=True )

                   .replace( '[(]','-',   regex=True ).astype(float))
region.describe()
college.groupby(['School Type']).mean()
region.groupby('Region').mean()
degree.head()
sns.set(style="whitegrid")

plt.figure(figsize=(20,8))

ax = sns.barplot(x="Undergraduate Major", y="Percent change from Starting to Mid-Career Salary", data=degree,palette='muted')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_title('Percent Change in Salary for Respective Majors')

ax.legend()
x0 = region['Starting Median Salary']

# Add 1 to shift the mean of the Gaussian distribution

x1 = region['Mid-Career Median Salary']



x2 = region['Mid-Career 90th Percentile Salary']

fig = go.Figure()

fig.add_trace(go.Histogram(x=x0,name='Starting Median Salary'))

fig.add_trace(go.Histogram(x=x1,name='Mid-Career Median Salary'))

fig.add_trace(go.Histogram(x=x2,name='Mid-Career 90th Percentile Salary'))

# Overlay both histograms

fig.update_layout(barmode='overlay')

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.7)



fig['layout'].update(title='Region vs Median Salary', legend=dict(x=0.65, y=0.8))

fig.show()
df = college.groupby('School Type').agg({'Starting Median Salary':'mean'})

df
df.reset_index(inplace = True)

df
y = df['Starting Median Salary']



x = df['School Type']



# Use textposition='auto' for direct text

fig = go.Figure(data=[go.Bar(

            x=x, y=y,

            text=y,

            textposition='auto',

       )])



# Customize aspect

fig.update_traces(marker_color=y, marker_line_color='rgb(8,48,107)',

                  marker_line_width=2, opacity=0.7)

fig.update_layout(title_text='Starting Median Salary Vs School Type')

fig.show()
df1 = college.groupby('School Type').agg({'Starting Median Salary':'mean'},{'Mid-Career Median Salary':'mean'})

df1.reset_index(inplace = True)

df1
school_type = college.groupby('School Type',as_index = False).mean()

ivy = school_type[school_type['School Type'] == 'Ivy League']

#ivy

school_type
ivy.info()
data = school_type['Mid-Career Median Salary']



fig = go.Figure(data=go.Scatter(x=school_type['School Type'],

                                y=data,

                                mode='markers',

                                #marker = ,

                                marker_color=data,

                                text=data,

                                 marker=dict(

                                    color=['rgb(93, 164, 24)', 'rgb(255, 144, 14)',  'rgb(44, 250, 10)', 'rgb(205, 65, 54)','rgb(10, 184, 2)'],

                                    size= 30,showscale=True)

                               )) # hover text goes here



fig.update_layout(title='School type Vs Mid-Career Salary')

fig.show()
fig = go.Figure(data=go.Scatter(x=degree['Undergraduate Major'],

                                y= degree['Starting Median Salary'],

                                mode='markers',

                                #marker = ,

                                marker_color=degree['Starting Median Salary'],

                                text=degree['Starting Median Salary'],

                                 marker=dict(

                                    color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',  'rgb(44, 160, 101)', 'rgb(255, 65, 54)','rgb(255, 104, 44)'],

                                    size= 12,showscale=True)

                               )) # hover text goes here



fig.update_layout(title='Major Vs Starting-Career Salary')

fig.show()
fig = go.Figure(data=go.Scatter(x=degree['Undergraduate Major'],

                                y= degree['Mid-Career Median Salary'],

                                mode='markers',

                                marker_color=degree['Mid-Career Median Salary'],

                                text=degree['Mid-Career Median Salary'],

                                marker=dict(

                                    color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',  'rgb(44, 160, 101)', 'rgb(255, 65, 54)','rgb(255, 104, 14)'],

                                    size= 15,showscale=True)

                               )) # hover text goes here



fig.update_layout(title='School type Vs Mid-Career Salary')

fig.show()