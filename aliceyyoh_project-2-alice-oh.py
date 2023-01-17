# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # matplotlib
import plotly.graph_objects as go


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read the file that I uploaded in Kaggle Data - the first try with the SEA data
dataset = pd.read_csv("../input/abdata/reviews_SEA.csv")
df = pd.DataFrame(dataset)

# print to check the data
df.tail()
# The dataset file is too large. I will delete listing_id, id, and review_id columns, and will only bring the rows with '/20'

# Remove columns
df = df.drop(columns=['listing_id', 'id', 'reviewer_id'])

# Extract year and months from 'date'
# Extract the rows only with the recent entries in Apr 2020, because I was worried if it wouldn't work for NY data due to the file size.
# But somehow these codes don't work.
# df['year'] = pd.DatetimeIndex(df['date']).year
# df['month'] = pd.DatetimeIndex(df['date']).month
# df = df[df['year'] == '2020']

# Second thought: just directly sort the data by date - this actually takes huge amount of time...
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by = 'date')
# drop the blank rows
df = df.dropna(how='any')
# get the last 100 & 75 datasets

# to train the model
smaller = df.tail(300)

# to analyze the data
smaller1 = df.tail(75)
smaller1
# save the new file

smaller.to_csv('Reviews_SEA_300.csv')
smaller1.to_csv('Reviews_SEA_75.csv')
# read the file
dataset2 = pd.read_csv("../input/abdata/reviews_NY_rev.csv")
df2 = pd.DataFrame(dataset2)

# Drop the unnecessary columns
df2 = df2.drop(columns=['listing_id', 'id', 'reviewer_id'])

# drop the rows with blank entries
df2 = df2.dropna(how='any')

# Sort the data by date - this actually takes huge amount of time...
df2['date'] = pd.to_datetime(df2['date'])
df2 = df2.sort_values(by = 'date')
# get the last 75 datasets

smaller2 = df2.tail(75)
smaller2
# save the new file

smaller2.to_csv('Reviews_NY_75.csv')
# read the processed files (analysis dataset) that I uploaded on the kaggle account
dataset3 = pd.read_csv("../input/abdata/processed_batch_SEA_All.csv")
df3 = pd.DataFrame(dataset3)

dataset4 = pd.read_csv("../input/abdata/processed_batch_NY_All.csv")
df4 = pd.DataFrame(dataset4)

# check
df3
# dataset manipulation

# Rename some columns
df3.rename(columns={"comments": "Comments", "Classification": "Sentiment", "Classification.1": "Aspect"}, inplace=True)
df4.rename(columns={"comments": "Comments", "Classification": "Sentiment", "Classification.1": "Aspect"}, inplace=True)

# split the Aspect column into multiple columns / leave comments, sentiment, aspects only
df3 = pd.concat([df3['Comments'], df3['Sentiment'], df3['Aspect'].str.split(':', expand=True)], axis=1)
df4 = pd.concat([df4['Comments'], df4['Sentiment'], df4['Aspect'].str.split(':', expand=True)], axis=1)

df3
# unpivot the aspects using melt to have them in one column

df3 = pd.melt(df3, id_vars =['Comments', 'Sentiment'], var_name='Order(no meaning)', value_name='Aspects')
df4 = pd.melt(df4, id_vars =['Comments', 'Sentiment'], var_name='Order(no meaning)', value_name='Aspects')

df3
# Clean the rows with no aspect data

# drop column 'Order(no meaning)'
df3 = df3.drop(columns=['Order(no meaning)'])
df4 = df4.drop(columns=['Order(no meaning)'])

# drop rows with 'None'
df4 = df4.dropna(how='any',axis=0) 
df3 = df3.dropna(how='any',axis=0) 

# try
df3
# Use plotly to visualize - seattle

x1 = df3.loc[df3['Sentiment'] == 'Positive', 'Aspects']
x2 = df3.loc[df3['Sentiment'] == 'Negative', 'Aspects']

trace1 = go.Histogram(x=x1, name='Positive',opacity=0.75)
trace2 = go.Histogram(x=x2, name = 'Negative',opacity=0.75)
data = [trace1, trace2]

fig = go.Figure(data=data, layout=layout)
fig.update_layout(
    xaxis={'categoryorder':'total descending'},
    title='Distribution of Aspects Based on Positive/Negative Sentiments - Seattle',
    barmode='overlay'
)

fig.show()
# Use plotly to visualize - New York

x3 = df4.loc[df4['Sentiment'] == 'Positive', 'Aspects']
x4 = df4.loc[df4['Sentiment'] == 'Negative', 'Aspects']

trace3 = go.Histogram(x=x3, name='Positive',opacity=0.75)
trace4 = go.Histogram(x=x4, name = 'Negative',opacity=0.75)
data2 = [trace3, trace4]

fig2 = go.Figure(data=data2, layout=layout)
fig2.update_layout(
    xaxis={'categoryorder':'total descending'},
    title='Distribution of Aspects Based on Positive/Negative Sentiments - New York',
    barmode='overlay'
)

fig2.show()
# Draw pie charts to check the portion of the aspects

# count the aspects 
aspect = df3['Aspects'].value_counts()

# plotting it on the pie chart
label_aspect = aspect.index
size_aspect = aspect.values

colors = ['Oxygen', 'Hydrogen', 'mediumturquoise', 'gold', 'crimson']

aspect_piechart = go.Pie(labels = label_aspect,
                         values = size_aspect,
                         marker = dict(colors = colors),
                         name = 'Aspects of Airbnb stay in Seattle', hole = 0.3)

dataset = [aspect_piechart]

fig3 = go.Figure(data = dataset,layout = layout)
fig3.update_layout(title_text='Aspects of Airbnb stay in Seattle', title_x=0.5)

fig3.show()
# count the aspects 
aspect2 = df4['Aspects'].value_counts()

# plotting it on the pie chart
label_aspect2 = aspect2.index
size_aspect2 = aspect2.values

colors = ['Oxygen', 'Hydrogen', 'mediumturquoise', 'gold', 'crimson']

aspect_piechart2 = go.Pie(labels = label_aspect2,
                         values = size_aspect2,
                         marker = dict(colors = colors),
                         name = 'Aspects of Airbnb stay in Seattle', hole = 0.3)

dataset2 = [aspect_piechart2]

fig4 = go.Figure(data = dataset2,layout = layout)
fig4.update_layout(title_text='Aspects of Airbnb stay in New York', title_x=0.5)

fig4.show()
