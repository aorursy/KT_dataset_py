# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.optimize import curve_fit
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as colors
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
college = pd.read_csv('../input/salaries-by-college-type.csv')
region = pd.read_csv('../input/salaries-by-region.csv')
majors = pd.read_csv('../input/degrees-that-pay-back.csv')
#Currently, all datasets are organised using the *variable_name* column by alphabetical order
#E.g. the CSV file stored in variable *region* is ordered alphabetically by the *school region*
college.head(10)
college.tail(10)
college.info()
region.head(10)
region.tail(10)
region.info()
majors.head(10)
majors.tail(10)
majors.info()
#renaming the columns in datasets so it's more succinct and easier to work with

college_columns = {
    "School Name" : "name",
    "School Type" : "type",
    "Starting Median Salary" : "start_p50",
    "Mid-Career Median Salary" : "mid_p50",
    "Mid-Career 10th Percentile Salary" : "mid_p10",
    "Mid-Career 25th Percentile Salary" : "mid_p25",
    "Mid-Career 75th Percentile Salary" : "mid_p75",
    "Mid-Career 90th Percentile Salary" : "mid_p90"
}
college.rename(columns=college_columns, inplace=True)

region_columns = {
    "School Name" : "name",
    "Region" : "region",
    "Starting Median Salary" : "start_p50",
    "Mid-Career Median Salary" : "mid_p50",
    "Mid-Career 10th Percentile Salary" : "mid_p10",
    "Mid-Career 25th Percentile Salary" : "mid_p25",
    "Mid-Career 75th Percentile Salary" : "mid_p75",
    "Mid-Career 90th Percentile Salary" : "mid_p90"
}
region.rename(columns=region_columns, inplace=True)

majors_columns = {
    "Undergraduate Major" : "name",
    "Starting Median Salary" : "start_p50",
    "Mid-Career Median Salary" : "mid_p50",
    "Percent change from Starting to Mid-Career Salary" : "increase",
    "Mid-Career 10th Percentile Salary" : "mid_p10",
    "Mid-Career 25th Percentile Salary" : "mid_p25",
    "Mid-Career 75th Percentile Salary" : "mid_p75",
    "Mid-Career 90th Percentile Salary" : "mid_p90"
}
majors.rename(columns=majors_columns, inplace=True)
#cleaning up the numerical figures in order to do mathematical graphing with them
selected_columns = ["start_p50", "mid_p50", "mid_p10", "mid_p25", "mid_p75", "mid_p90"]

# List of all datasets
datasets_list = [college, region, majors]

for dataset in datasets_list:
    for column in selected_columns:
        dataset[column] = dataset[column].str.replace("$","")
        dataset[column] = dataset[column].str.replace(",","")
        dataset[column] = pd.to_numeric(dataset[column])
group_by_type = college.groupby("type")

x_data = []
y_data = []

colors = ['rgba(159, 210, 238, 0.75)', 
          'rgba(177, 174, 244, 0.75)', 
          'rgba(205, 158, 214, 0.75)', 
          'rgba(232, 152, 150, 0.75)', 
          'rgba(255, 168, 138, 0.75)']

border_colors = ['rgba(148, 114, 110, 1)',
                 'rgba(127, 118, 116, 1)', 
                 'rgba(163, 135, 125, 1)', 
                 'rgba(127, 86, 78, 1)', 
                 'rgba(158, 146, 144, 1)']

for uni_type, uni_group in group_by_type:
    x_data.append(uni_type)
    y_data.append(uni_group["mid_p50"])

traces = []

for xd, yd, color, b_color in zip(x_data, y_data, colors, border_colors):
        traces.append(go.Box(
            y = yd,
            name = xd,
            boxpoints = 'all',
            jitter = 0.5,
            whiskerwidth = 0.2,
            fillcolor = color, #box color
            marker=dict(size=2),
           line = dict(color=b_color), #border color
        ))

layout = go.Layout(
    title='How\'s Your Prospect In Life Graduating From The Following Types of University?',
    margin=dict(
        l=40,
        r=30,
        b=80,
        t=100,
    ),
    paper_bgcolor='rgb(244, 238, 225)',
    plot_bgcolor='rgb(244, 238, 225)',
    showlegend=False
)

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)
majors_sort = majors.sort_values("mid_p50", ascending=False).head(20)

def cut_name(x):
    if len(x) <= 21:
        return x
    else:
        return x[0:18] + "..."

startingMedian = go.Bar(
    x = majors_sort["name"].apply(cut_name).tolist(),
    y = majors_sort["start_p50"].tolist(),
    name='Starting',
    marker=dict(
        color='rgba(102, 190, 178, 0.7)',
        line=dict(
            color='rgba(102, 190, 178, 1.0)',
            width=2,
        )
    )
)

midCareerMedian = go.Bar(
    x = majors_sort["name"].apply(cut_name).tolist(),
    y = majors_sort["mid_p50"].tolist(),
    name='Mid-Career',
    marker=dict(
        color='rgba(249, 113, 113, 0.7)',
        line=dict(
            color='rgba(249, 113, 113, 1.0)',
            width=2,
        )
    )
)

percentageChange = go.Scatter(
    x = majors_sort["name"].apply(cut_name).tolist(),
    y = majors_sort["increase"].tolist(),
    name='Percent change',
    mode = 'markers',
    marker=dict(
        color='rgba(67, 124, 140, 0.7)',
        line=dict(
            color='rgba(67, 124, 140, 1.0)',
            width=4,
        ),
        symbol="hexagon-dot",
        size=15
    ),
    yaxis='y2'
)

data = [startingMedian, midCareerMedian,percentageChange]

layout = go.Layout(
    barmode='group',
    title = 'Sometimes, You Just Gotta Be Patient...',
    
    width=850,  #the width of the graph chart region
    height=500, #the height of the graph chart region
    
    margin=go.Margin(
        l=75,   #left margin = 75px
        r=75,   #right margin = 75px
        b=120,  #bottom margin = 120px
        t=80,   #top margin = 80px
        pad=10  #padding = 10px
    ),
    
    paper_bgcolor='rgb(244, 238, 225)',
    plot_bgcolor='rgb(244, 238, 225)',
    
    yaxis = dict(
        title= 'Median Salary [$]',
        anchor = 'x',
        rangemode='tozero'
    ),  
    
    yaxis2=dict(
        title='Change [%]',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right',
        anchor = 'x',
        rangemode = 'tozero',
        dtick = 19.95
    ),

    legend=dict(x=0.1, y=0.05)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
# Sorted dataset
majors_sort_mid90 = majors.sort_values("mid_p90", ascending=True)

# Method that shortens long texts
def cut_name(x):
    if len(x) <= 25:
        return x
    else:
        return x[0:22] + "..."

# Prepared information
# traces = {column_name: dictionary{}}
traces = {
    "mid_p10" : {
        "name" : "Mid-Career 10th Percentile",
        "color" : "rgba(255, 114, 114, 0.7)",
        "line_color" : "rgba(255, 114, 114, 1.0)"
    },
    "mid_p25" : {
        "name" : "Mid-Career 25th Percentile",
        "color" : "rgba(255, 202, 120, 0.7)",
        "line_color" : "rgba(255, 202, 120, 1.0)"
    },
    "mid_p50" : {
        "name" : "Mid-Career 50th Percentile",
        "color" : "rgba(253, 255, 88, 0.7)",
        "line_color" : "rgba(253, 255, 88, 1.0)"
    },
    "mid_p75" : {
        "name" : "Mid-Career 75th Percentile",
        "color" : "rgba(153, 255, 45, 0.7)",
        "line_color" : "rgba(153, 255, 45, 1.0)"
    },
    "mid_p90" : {
        "name" : "Mid-Career 90th Percentile",
        "color" : "rgba(49, 255, 220, 0.7)",
        "line_color" : "rgba(49, 255, 220, 1.0)"
    }
}

# List that stores information about data traces
data = []

# Single trace 
for key, value in traces.items():
    trace = go.Scatter(
        x = majors_sort_mid90[key].tolist(),
        y = majors_sort_mid90["name"].apply(cut_name).tolist(),
        name = value["name"], #name of legend
        mode = 'markers',
        marker=dict(
            color = value["color"],
            line=dict(
                color = value["line_color"],
                width=2,
            ),
            symbol="hexagon-dot",
            size=10
        ),
    )
    data.append(trace)

# Chart layout
layout = go.Layout(
    title = 'Study Whatever You Want, Kids - Just Remember to Include Economics At All Costs :)',
    
    width=850,    #the width of the graph chart region
    height=1200,  #the height of the graph chart region
    
    margin=go.Margin(
        l=180, #left margin = 180px
        r=50,  #right margin = 50px
        b=80,  #bottom margin = 80px
        t=80,  #top margin = 80px
        pad=10 #padding = 10px
    ),
    
    paper_bgcolor='rgb(244, 238, 225)',
    plot_bgcolor='rgb(244, 238, 225)',
    
    yaxis = dict(
        anchor = 'x',
        rangemode='tozero',
        tickfont=dict(
            size=10
        ),
        ticklen=1
    ),  
    
    legend=dict(x=0.6, y=0.07)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)