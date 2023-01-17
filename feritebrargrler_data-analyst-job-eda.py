# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid") # --> helps to visulize tools with grids. You can use another ones with looking plt.style.available

import seaborn as sns

from collections import Counter

import warnings

warnings.filterwarnings("ignore") # dont show warnings based on python

import plotly.graph_objs as go

# plotly

#import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/data-analyst-jobs/DataAnalyst.csv")
data.head()
data.info()
data = data.rename(columns={'Unnamed: 0':'Index', 'Job Title':'JobTitle', 'Salary Estimate':'SalaryEstimate', 'Job Description':'JobDescription', 'Company Name':'CompanyName', 'Type of ownership' : 'OwnershipType', 'Easy Apply':'EasyApply'})                                      
data.head()
data.columns[data.isnull().any()] # in which columns there are missing values? 
data.isnull().sum() # how many ?
data['CompanyName'] = data['CompanyName'].fillna('Unknown Company')
data.isnull().sum()
data['EasyApply'].value_counts()
(data['EasyApply'].replace('-1', False, inplace=True))
data["EasyApply"].value_counts()
data['Competitors'].value_counts()
data['Competitors'].replace('-1', np.nan, inplace=True)
data['Competitors'].value_counts()
data['Rating'].value_counts()
data['Rating'].replace(-1.0, "-")
data.head()
data['MinSalary'],data['MaxSalary']=data['SalaryEstimate'].str.split('-').str

data['MinSalary'] =  data['MinSalary'].str.strip().str.lstrip('$').str.rstrip('K').str.strip()

data['MinSalary'] =  pd.to_numeric(data['MinSalary'],downcast='float') # --> converting pandas series to numeric values

data['MaxSalary'] = data['MaxSalary'].str.strip().str.lstrip('$').str.rstrip('(Glassdoor est.)').str.rstrip('K ')

data['MaxSalary'] =  pd.to_numeric(data['MaxSalary'],downcast='float') # --> converting pandas series to numeric values
data['MinSalary'].value_counts()
data.head()
data['MinRevenue'], data['MaxRevenue'] = data['Revenue'].str.split('to').str
data['MinRevenue'].value_counts()
data['MinRevenue'] = data['MinRevenue'].str.strip()
data['MinRevenue'].replace('Unknown / Non-Applicable', np.nan, inplace=True)
data['MinRevenue'].replace('$100', 100000000, inplace=True)
data['MinRevenue'].replace('$50', 50000000, inplace=True)
data['MinRevenue'].replace('$1', 1000000, inplace=True)
data['MinRevenue'].replace('$10+ billion (USD)', 10000000000, inplace=True)
data['MinRevenue'].replace('-1', np.nan, inplace=True)
data['MinRevenue'].replace('$10', 10000000, inplace=True)
data['MinRevenue'].replace('$2', 2000000, inplace=True)
data['MinRevenue'].replace('$5', 5000000, inplace=True)
data['MinRevenue'].replace('$25', 25000000, inplace=True)
data['MinRevenue'].replace('Less than $1 million (USD)', 500000, inplace=True)
data['MinRevenue'].replace('$500 million', 500000000, inplace=True) 
data['MinRevenue'].value_counts()
# Unknown / Non-Applicable: changed with nan

# $100 : changed with 100.000.000

# $50 :changed with 50.000.00

# $1 : changed with 1.000.000

# $10+ billion (USD) : changed with 10.000.000.000 as a min value 

# -1 : changed with nan

# $10 : changed with 10.000.000

# $2 : 2.000.000

# $25 : 25.000.000

# For not missing values ı changed Less than 1 million (USD) as 500.000 (averagely)

# $500 million : changed with 500.000.000
data['MinRevenue'] = pd.to_numeric(data['MinRevenue'])

data['MaxRevenue'] = data['MaxRevenue'].str.strip()
data['MaxRevenue'].value_counts()
data['MaxRevenue'].replace('$500 million (USD)', 500000000, inplace=True)
data['MaxRevenue'].replace('$100 million (USD)', 100000000, inplace=True)
data['MaxRevenue'].replace('$25 million (USD)', 25000000, inplace=True)
data['MaxRevenue'].replace('$5 billion (USD)', 5000000000, inplace=True)
data['MaxRevenue'].replace('$5 million (USD)', 5000000, inplace=True)
data['MaxRevenue'].replace('$50 million (USD)', 50000000, inplace=True)
data['MaxRevenue'].replace('$2 billion (USD)', 2000000000, inplace=True)
data['MaxRevenue'].replace('$1 billion (USD)', 1000000000, inplace=True)
data['MaxRevenue'].replace('$10 million (USD)', 10000000, inplace=True)   
data['MaxRevenue'].replace('$10 billion (USD)', 10000000000, inplace=True)    
data['MaxRevenue'].value_counts()
data.tail()
bool_over_10b = data['Revenue']=='$10+ billion (USD)'

index_of_trues = [i for i, j in enumerate(bool_over_10b) if j == bool(True)]

data['MaxRevenue'][index_of_trues] =  (data['MaxRevenue'].fillna(20000000000)[index_of_trues])
#if you look at Revenue = $10+ billion (USD) there is a MinRevenue but there is no MaxRevenue

#so if you look at the ratio of the min and max you will find 2. So I predict that MaxRevenue $10+ billion (USD) must be 20 billion
data.tail()
data['MaxRevenue'].value_counts()
data.head()
job_title = data.JobTitle

plt.subplots(figsize=(10,10))

word_cloud = WordCloud(

    background_color = 'white',

    width = 512,

    height = 384,

).generate(' '.join(job_title))

plt.imshow(word_cloud)

plt.axis('off')

plt.show()
#data preparetion

new_index = data['MaxRevenue'].sort_values(ascending=False).head(10).index.values # values were sorted according to descending order.

sorted_data = data.reindex(new_index)

#visualization

plt.figure(figsize=(70,40))

sns.barplot(x=sorted_data['CompanyName'], y=sorted_data['MaxRevenue'])

plt.xticks(rotation=90)

plt.xlabel('Company Name')

plt.ylabel('Maximum Revenues')

plt.title('Maximum Revenues Given Companies')

plt.show()

data.head()
#data preparation

df=data.groupby('Sector')[['MaxRevenue','MinRevenue']].mean().sort_values(['MaxRevenue','MinRevenue'],ascending=False).head(10)

df_sector = []

for i in df.index:

    df_sector.append(i)

df['Sector'] = df_sector

df
#visualization

# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df.Sector,

                y = df.MaxRevenue,

                name = "Maximum Revenue",

                marker = dict(color = 'rgba(255, 174, 255, 1)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df.Sector)

# create trace2 

trace2 = go.Bar(

                x = df.Sector,

                y = df.MinRevenue,

                name = "Minimum Revenue",

                marker = dict(color = 'rgba(255, 255, 128, 1)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df.Sector)

df = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = df, layout = layout)

iplot(fig)
#data preparation

df=data.groupby('Industry')[['MaxRevenue','MinRevenue']].mean().sort_values(['MaxRevenue','MinRevenue'],ascending=False).head(12)

df_ındustry = []

for i in df.index:

    df_ındustry.append(i)

df['Industry'] = df_ındustry

df
# Revenue Rates of Top 12 Industries



# data preparation

pie = df.MaxRevenue



#Plotly calculates the rates automaticaly.



#figure

fig = {

    "data" : [

        {

            "values" : pie,

            "labels" : df.Industry,

            "domain": {"x": [0.5, .30]},

            "name" : "Industry Rates",

            "hoverinfo" : "label+percent+name", # cursor shows the rates and name of ındustry

            "hole" : 0.2, # Magnitude of white hole

            "type" : "pie"

        }

    ],

    "layout" : {

        "title" : "Revenue Rates of Top 12 Industries",

        "annotations" : [

            {"font" : { "size": 10}, #Magnitude of Generosity name at the top of pie chart

            "showarrow" : False,

             "text": "Industries",

            "x": 0.20,

            "y": 1

            }

        ]

    } 

}

iplot(fig)
# prepare data

max = data.MaxSalary

min = data.MinSalary



trace1 = go.Histogram(

    x=max,

    opacity=0.75,

    name = "MaxSalary",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))

trace2 = go.Histogram(

    x=min,

    opacity=0.75,

    name = "MinSalary",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



df = [trace1, trace2]

layout = go.Layout(barmode='overlay',

                   title='Ratio Of Minimum Salary and Maximum Salary',

                   xaxis=dict(title='Max-Min Salary'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=df, layout=layout)

iplot(fig)
#data MinSalary

df=data.groupby('Sector')[['MaxSalary','MinSalary']].mean().sort_values(['MaxSalary','MinSalary'],ascending=False).head(10)

df_sector = []

for i in df.index:

    df_sector.append(i)

df['Sector'] = df_sector

df
#visualization

# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df.Sector,

                y = df.MaxSalary,

                name = "Maximum Salary",

                marker = dict(color = 'rgba(102, 0, 102, 1)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df.Sector)

# create trace2 

trace2 = go.Bar(

                x = df.Sector,

                y = df.MinSalary,

                name = "Minimum Salary",

                marker = dict(color = 'rgba(160, 160, 160, 1)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df.Sector)

df = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = df, layout = layout)

iplot(fig)
#data preparation

df=data.groupby('Industry')[['MaxSalary','MinSalary']].mean().sort_values(['MaxSalary','MinSalary'],ascending=False).head(15)

df_ındustry = []

for i in df.index:

    df_ındustry.append(i)

df['Industry'] = df_ındustry

df
#visualization

# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df.Industry,

                y = df.MaxSalary,

                name = "Maximum Salary",

                marker = dict(color = 'rgba(255, 0, 0, 1)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df.Industry)

# create trace2 

trace2 = go.Bar(

                x = df.Industry,

                y = df.MinSalary,

                name = "Minimum Salary",

                marker = dict(color = 'rgba(255, 255, 0, 1)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df.Industry)

df = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = df, layout = layout)

iplot(fig)
#data preparation

df=data.groupby('JobTitle')[['MaxSalary','MinSalary']].mean().sort_values(['MaxSalary','MinSalary'],ascending=False).head(18)

df_JT = []

for i in df.index:

    df_JT.append(i)

df['JobTitle'] = df_JT

df
#visualization

# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df.JobTitle,

                y = df.MaxSalary,

                name = "Maximum Salary",

                marker = dict(color = 'rgba(153, 255, 255, 1)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df.JobTitle)

# create trace2 

trace2 = go.Bar(

                x = df.JobTitle,

                y = df.MinSalary,

                name = "Minimum Salary",

                marker = dict(color = 'rgba(128, 0, 128, 1)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df.JobTitle)

df = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = df, layout = layout)

iplot(fig)
df = data.groupby('Location')[['MaxSalary', 'MinSalary']].mean().sort_values(['MaxSalary', 'MinSalary'],ascending=False).head(15)

df_loc = []

for i in df.index:

    df_loc.append(i)

df['Location'] =df_loc

df
#visualization

# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df.Location,

                y = df.MaxSalary,

                name = "Maximum Salary",

                marker = dict(color = 'rgba(51, 255, 51, 1)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df.Location)

# create trace2 

trace2 = go.Bar(

                x = df.Location,

                y = df.MinSalary,

                name = "Minimum Salary",

                marker = dict(color = 'rgba(255, 255, 153, 1)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df.Location)

df = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = df, layout = layout)

iplot(fig)
data['EasyApply'].value_counts()
data['Rating'].value_counts()
df = data.groupby('EasyApply')[['Rating']].mean().sort_values(['Rating'],ascending=False).head(15)

df_rt = []

for i in df.index:

    df_rt.append(i)

df['EasyApply'] =df_rt

df