# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import plotly.express as px

import plotly.graph_objects as go

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_subject = pd.read_csv('../input/us-college-graduates-wages/labor_market_college_grads.csv')
df_subject = df_subject[df_subject.Major != 'Overall']
df_subject.Major.unique()
df_subject.head(10)
display(df_subject[df_subject['Major']=='Fine Arts'])

display(df_subject[df_subject['Major'] == 'Mathematics'])
x = df_subject.Major

fig = go.Figure(data=[

    go.Bar(name='Early Career', x=x, y=df_subject['Median Wage Early Career']),

    go.Bar(name='Mid Career', x=x, y=df_subject['Median Wage Mid-Career'])

])

fig.update_layout(barmode='group')

fig.update_layout(

    font_family="Ariel",

    font_color="black",

    font_size = 8,

    title={

        'text': 'Wages by Degree Major',

        'font_size' : 20,

        'y':0.8,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis_title={

        'text' : "Degree Major",

        'font_size' : 16

    },

    yaxis_title={

        'text' : "Annual Wage",

        'font_size' : 16

    }

)

fig.show()
import plotly.graph_objs as go



xs = df_subject['Major']

ys = df_subject["Median Wage Early Career"]



data = [go.Bar(

    x=df_subject['Major'],

    y=df_subject["Median Wage Early Career"],

    marker={

        'color': ys,

        'colorscale': 'Viridis'

    }

)]

layout = {

    'xaxis': {

        'categoryorder': 'array',

        'categoryarray': [x for _, x in sorted(zip(ys, xs))]

    }

}

fig = go.FigureWidget(data=data, layout=layout)

fig.update_layout(

    font_family="Ariel",

    font_color="black",

    font_size = 8,

    title={

        'text': 'Early Career Median Wage by Major',

        'y':0.8,

        'x':0.5,

        'font_size' : 20,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis_title={

        'text' : "Degree Major",

        'font_size' : 16

    },

    yaxis_title={

        'text' : "Annual Wage",

        'font_size' : 16

    }

)

fig.show()
xs = df_subject['Major']

ys = df_subject["Median Wage Mid-Career"]



data = [go.Bar(

    x=df_subject['Major'],

    y=df_subject["Median Wage Mid-Career"],

    marker={

        'color': ys,

        'colorscale': 'Viridis'

    }

)]

layout = {

    'xaxis': {

        'categoryorder': 'array',

        'categoryarray': [x for _, x in sorted(zip(ys, xs))]

    }

}

fig = go.FigureWidget(data=data, layout=layout)

fig.update_layout(

    font_family="Ariel",

    font_color="black",

    font_size = 8,

    title={

        'text': 'Mid-Career Median Wage by Major',

        'y':0.8,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top',

        'font_size' : 20

    },

    xaxis_title={

        'text' : "Degree Major",

        'font_size' : 16

    },

    yaxis_title={

        'text' : "Annual Wage",

        'font_size' : 16

    }

)

fig.show()
dftop10_early = df_subject.sort_values(by=['Median Wage Early Career'], ascending=False).head(10)

dftop10_mid = df_subject.sort_values(by=['Median Wage Mid-Career'], ascending=False).head(10)

dftop10_early


fig = px.bar(dftop10_early, x="Major", y=["Median Wage Early Career"], color="Major", text =dftop10_early['Unemployment Rate'])

fig.update_layout(

    font_family="Ariel",

    font_color="black",

    font_size = 16,

    title={

        'text': 'Top 10 Early Career Wages by Degree Major with Unemployment Rate',

        'y':1,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis_title="Degree Major",

    yaxis_title="Annual Wage",

)

fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
fig = px.bar(dftop10_mid, x="Major", y=["Median Wage Mid-Career"], color="Major", text =dftop10_mid['Unemployment Rate'])

fig.update_layout(

    font_family="Ariel",

    font_color="black",

    font_size = 16,

    title={

        'text': 'Top 10 Mid Career Wages by Degree Major with Unemployment Rate',

        'y':1,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis_title="Degree Major",

    yaxis_title="Annual Wage",

)

fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
dfbottom10_early = df_subject.sort_values(by=['Median Wage Early Career'], ascending=True).head(10)

dfbottom10_mid = df_subject.sort_values(by=['Median Wage Mid-Career'], ascending=True).head(10)

dfbottom10_early
fig = px.bar(dfbottom10_early, x="Major", y=["Median Wage Early Career"], color="Major", text =dfbottom10_early['Unemployment Rate'])

fig.update_layout(

    font_family="Ariel",

    font_color="black",

    font_size = 16,

    title={

        'text': 'Bottom 10 Early Career Wages by Degree Major with Unemployment Rate',

        'y':1,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis_title="Degree Major",

    yaxis_title="Annual Wage",

)

fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
fig = px.bar(dfbottom10_mid, x="Major", y=["Median Wage Mid-Career"], color="Major", text =dfbottom10_mid['Unemployment Rate'])

fig.update_layout(

    font_family="Ariel",

    font_color="black",

    font_size = 16,

    title={

        'text': 'Bottom 10 Mid Career Wages by Degree Major with Unemployment Rate',

        'y':1,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis_title="Degree Major",

    yaxis_title="Annual Wage",

)

fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
df_ue = df_subject[['Major','Unemployment Rate', 'Underemployment Rate']]

df_ue.head()
df1 = df_ue.sort_values(by=['Unemployment Rate'])

fig = px.bar(df1, x="Major", y=["Unemployment Rate"], color="Major")

fig.update_layout(

    font_family="Ariel",

    font_color="black",

    font_size = 8,

    title={

        'text': "Unemployment Rate",

        'y':0.8,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top',

        'font_size' : 20},

    xaxis_title={

        'text' : 'Degree Major',

        'font_size' : 16

    },

    yaxis_title={

        'text' : 'Percentage',

        'font_size' : 16

    }

)

fig.show()
df2 = df_ue.sort_values(by=['Underemployment Rate'])

fig = px.bar(df2, x="Major", y=['Underemployment Rate'], color="Major", title="Underemployment Rate")

fig.update_layout(

    font_family="Times New Roman",

    font_color="black",

    font_size = 8,

    title={

        'text': 'Underemployment Rate',

        'y':0.8,

        'x':0.5,

        'font_size' : 20,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis_title={

        'text' : 'Degree Major',

        'font_size' : 16

    },

    yaxis_title={

        'text' : 'Percentage',

        'font_size' : 16

    }

)

fig.show()
df_wages = pd.read_csv('../input/us-college-graduates-wages/wages.csv')

df_wages.head()
df_wages['Date'] = pd.to_datetime(df_wages['Date'], format = '%m/%d/%Y')

df_wages = df_wages.set_index(df_wages['Date'])

df_wages = df_wages.drop(columns = ['Date'])
df_wages.head()
df_wages = df_wages[df_wages.index >= '1995-01-01']
fig = go.FigureWidget(data=[

    go.Scatter(x=df_wages.index, y=df_wages["Bachelor's degree: 25th percentile"], mode='lines', line={'dash': 'dash', 'color': 'red'}, name = "25th perc - Degree"),

    go.Scatter(x=df_wages.index, y=df_wages["Bachelor's degree: median"], mode='lines', line={'dash': 'solid', 'color': 'purple'}, name = "Median - Degree"),

    go.Scatter(x=df_wages.index, y=df_wages["Bachelor's degree: 75th percentile"], mode='lines', line={'dash': 'dash', 'color': 'blue'}, name  = "75th perc - Degree"),

    go.Scatter(x=df_wages.index, y=df_wages["High school diploma: median"], mode='lines', line={'dash': 'solid', 'color': 'green'}, name = "Median - H.S. Diploma")

])

fig.update_layout(

    font_family="Times New Roman",

    font_color="black",

    font_size = 16,

    title={

        'text': 'Annual Wages of College Graduates and High School Graduates',

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis_title="Year",

    yaxis_title="Annual Wages ($)",

    shapes = [

                    dict(

            type="rect",

            x0='2000-03-01',

            x1='2002-10-01',

            y0=24000,

            y1=70000,

            fillcolor="Red",

            opacity=0.4,

            layer="below",

            line_width=0

                    ),

                    dict(

            type="rect",

            x0="2007-12-01",

            x1="2009-06-01",

            y0=24000,

            y1=70000,

            fillcolor="Red",

            opacity=0.4,

            layer="below",

            line_width=0

                    ),

                    dict(

            type = 'line',

            x0 = '2008-10-01',

            x1 = '2008-10-01',

            y0 = 24000,

            y1 = 70000,

            line = dict(

            color = 'Black',

            dash = 'dashdot'))

        ],

    annotations=[

             dict(text="The Great Recession",x = '2007-12-01', y=70000),

             dict(text="Dot Com Bubble", x='2000-03-01', y=70000, hovertext = 'Market Height'),

             dict(text = "EESA Passed", x = '2008-10-01', y = 70000, showarrow=True, arrowhead=1, ax=80, ay=-50, hovertext = 'Emergency Economic Stabilization Act')

         ]

)

fig.show()
df2 = pd.read_csv('../input/us-college-graduates-wages/under_employment_college_grads.csv')
df2['Date'] = pd.to_datetime(df2['Date'], format = '%m/%d/%Y')

df2 = df2.set_index(df2['Date'])

df2 = df2.drop(columns = ['Date'])
df2.head()
df2 = df2[df2.index >= '1995-01-01']
fig = go.FigureWidget(data=[

    go.Scatter(x=df2.index, y=df2["Recent graduates"], mode='lines', line={'dash': 'solid', 'color': 'purple'}, name = "Recent grads"),

    go.Scatter(x=df2.index, y=df2["College graduates"], mode='lines', line={'dash': 'solid', 'color': 'blue'}, name = "College grads")

])

fig.update_layout(

    font_family="Times New Roman",

    font_color="black",

    font_size = 16,

    title={

        'text': 'Underemployment Rate for College Graduates',

        'y':0.95,

        'x':0.45,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis_title="Year",

    yaxis_title="Percentage",

    shapes = [

                    dict(

            type="rect",

            x0='2000-03-01',

            x1='2002-10-01',

            y0=30,

            y1=49,

            fillcolor="Red",

            opacity=0.4,

            layer="below",

            line_width=0

                    ),

                    dict(

            type="rect",

            x0="2007-12-01",

            x1="2009-06-01",

            y0=30,

            y1=49,

            fillcolor="Red",

            opacity=0.4,

            layer="below",

            line_width=0

                    ),

                    dict(

            type = "line",

            x0='2020-03-11',

            x1='2020-03-11',

            y0=30,

            y1=49,

            line=dict(

            color="Red",

            dash="dash"

            )

                    ),

                    dict(

            type = 'line',

            x0 = '2008-10-01',

            x1 = '2008-10-01',

            y0 = 30,

            y1 = 49,

            line = dict(

            color = 'Black',

            dash = 'dashdot'))

        ],

    annotations=[

             dict(text="The Great Recession",x = '2007-12-01', y=49, showarrow=True, arrowhead=1),

             dict(text="Dot Com Bubble", x='2000-03-01', y=49, hovertext = 'Market Height', showarrow=True, arrowhead=1),

             dict(text ="Covid-19", x='2020-03-11', y=49, hovertext = 'WHO declares Covid-19 a pandemic', showarrow=True, arrowhead=1),

             dict(text = "EESA Passed", x = '2008-10-01', y = 49, showarrow=True, arrowhead=1, ax=80, ay=-50, hovertext = 'Emergency Economic Stabilization Act')

         ]

)

fig.show()
df3 = pd.read_csv('../input/us-college-graduates-wages/Unemployment_rate.csv')
df3['Date'] = pd.to_datetime(df3['Date'], format = '%m/%d/%Y')

df3 = df3.set_index(df3['Date'])

df3 = df3.drop(columns = ['Date'])
df3 = df3[df3.index >= '1995-01-01']
fig = go.FigureWidget(data=[

    go.Scatter(x=df3.index, y=df3["Young workers"], mode='lines', line={'dash': 'solid', 'color': 'orange'}, name = "Young workers"),

    go.Scatter(x=df3.index, y=df3["All workers"], mode='lines', line={'dash': 'solid', 'color': 'green'}, name = "All workers"),

    go.Scatter(x=df3.index, y=df3["Recent graduates"], mode='lines', line={'dash': 'solid', 'color': 'purple'}, name = "Recent graduates"),

    go.Scatter(x=df3.index, y=df3["College graduates"], mode='lines', line={'dash': 'solid', 'color': 'blue'}, name = "College graduates")

])

fig.update_layout(

    font_family="Times New Roman",

    font_color="black",

    font_size = 16,

    title={

        'text': 'Unemployment Rate',

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis_title="Year",

    yaxis_title="Percentage",

    shapes = [

                    dict(

            type="rect",

            x0='2000-03-01',

            x1='2002-10-01',

            y0=0,

            y1=22,

            fillcolor="Red",

            opacity=0.4,

            layer="below",

            line_width=0

                    ),

                    dict(

            type="rect",

            x0="2007-12-01",

            x1="2009-06-01",

            y0=0,

            y1=22,

            fillcolor="Red",

            opacity=0.4,

            layer="below",

            line_width=0

                    ),

                    dict(

            type = "line",

            x0='2020-03-11',

            x1='2020-03-11',

            y0=0,

            y1=22,

            line=dict(

            color="Red",

            dash="dash"

            )

                    ),

                    dict(

            type = 'line',

            x0 = '2008-10-01',

            x1 = '2008-10-01',

            y0 = 0,

            y1 = 22,

            line = dict(

            color = 'Black',

            dash = 'dashdot'))

        ],

    annotations=[

             dict(text="The Great Recession",x = '2007-12-01', y=22, showarrow=True, arrowhead=1),

             dict(text="Dot Com Bubble", x='2000-03-01', y=22, hovertext = 'Market Height', showarrow=True, arrowhead=1),

             dict(text ="Covid-19", x='2020-03-11', y=22, hovertext = 'WHO declares Covid-19 a pandemic', showarrow=True, arrowhead=1),

             dict(text = "EESA Passed", x = '2008-10-01', y = 22, showarrow=True, arrowhead=1, ax=80, ay=-50, hovertext = 'Emergency Economic Stabilization Act')

         ]

)

fig.show()