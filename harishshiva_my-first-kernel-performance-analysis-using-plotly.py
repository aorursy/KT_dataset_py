import pandas as pd

import numpy as np

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go
df= pd.read_csv("../input/StudentsPerformance.csv")

df.head()
df.isnull().sum()
df.info()
df['test preparation course'].value_counts()
df_none= df[df['test preparation course']=='none']

df_completed=df[df['test preparation course']=='completed']

math_none=df_none['math score']

math_completed=df_completed['math score']

trace1 = go.Box(

     y = math_none,

     name = 'Not Completed the Course',

     marker = dict(color = 'rgba(254,24,0,0.8)')

)

trace2 = go.Box(

     y = math_completed,

     name = 'Completed the Course',

     marker = dict(color = 'rgba(5,140,20,0.8)')

)

data = [trace1,trace2]

layout= go.Layout(title='Math Score Comparison')

fig =go.Figure(data=data, layout=layout)

iplot(fig)
reading_none=df_none['reading score']

reading_completed=df_completed['reading score']

trace1 = go.Box(

     y = reading_none,

     name = 'Not Completed the Course',

     marker = dict(color = 'rgba(254,24,0,0.8)')

)

trace2 = go.Box(

     y = reading_completed,

     name = 'Completed the Course',

     marker = dict(color = 'rgba(5,140,20,0.8)')

)

data = [trace1,trace2]

layout= go.Layout(title='Reading Score Comparison')

fig =go.Figure(data=data, layout=layout)

iplot(fig)
writing_none=df_none['writing score']

writing_completed=df_completed['writing score']

trace1 = go.Box(

     y = writing_none,

     name = 'Not Completed the Course',

     marker = dict(color = 'rgba(254,24,0,0.8)')

)

trace2 = go.Box(

     y = writing_completed,

     name = 'Completed the Course',

     marker = dict(color = 'rgba(5,140,20,0.8)')

)

data = [trace1,trace2]

layout= go.Layout(title='Writing Score Comparison')

fig =go.Figure(data=data, layout=layout)

iplot(fig)
df.head()
df['race/ethnicity'].value_counts()
race=df.groupby('race/ethnicity')

racedf=race.mean()



trace1= go.Bar(x= racedf.index,

              y= racedf['math score'],

              name='Math')



trace2= go.Bar(x= racedf.index,

              y= racedf['reading score'],

              name='Reading')



trace3= go.Bar(x= racedf.index,

              y= racedf['writing score'],

              name='Writing')



data=[trace1,trace2,trace3]

layout=go.Layout(title='Average Score for different Race/Ethnicity')

fig= go.Figure(data=data, layout=layout)

iplot(fig)
df.head()
#scatter plot

df_marks = df.iloc[:,5:]

import plotly.figure_factory as ff

df_marks['index'] = np.arange(1,len(df_marks)+1)

fig = ff.create_scatterplotmatrix(df_marks, diag ='box', index = 'index' , colormap ='Portland',

                                 colormap_type = 'seq', height = 700, width = 700)

iplot(fig)
df.head()
gender_lunch=df.groupby(['gender','lunch'])

gender_lunch_count=gender_lunch.size()

trace= go.Pie(values=gender_lunch_count.values,

      labels=gender_lunch_count.index)

data=[trace]

iplot(data)
df.head()
gender_lunch_mean=gender_lunch.mean()

gender_lunch_mean.index[0]

new_data=[]

for i in gender_lunch_mean.index:

    new_data.append(' '.join(i))
new_data


trace1= go.Bar(x= new_data,

              y= gender_lunch_mean['math score'],

              name='Math')



trace2= go.Bar(x= new_data,

              y= gender_lunch_mean['reading score'],

              name='Reading')



trace3= go.Bar(x= new_data,

              y= gender_lunch_mean['writing score'],

              name='Writing')



data=[trace1,trace2,trace3]

layout=go.Layout(title='Average Score for Gender/Lunch Catagories')

fig= go.Figure(data=data, layout=layout)

iplot(fig)
df.head()
df['parental level of education'].value_counts()
trace1= go.Heatmap(x=df['parental level of education'], y=df['test preparation course'], z=df['math score'].values.tolist())

trace2= go.Heatmap(x=df['parental level of education'], y=df['test preparation course'], z=df['reading score'].values.tolist())

trace3= go.Heatmap(x=df['parental level of education'], y=df['test preparation course'], z=df['writing score'].values.tolist())



data=[trace1]

layout=go.Layout(title='Math Score')

fig= go.Figure(data=data, layout=layout)

iplot(fig)
data=[trace2]

layout=go.Layout(title='Reading Score')

fig= go.Figure(data=data, layout=layout)

iplot(fig)
data=[trace3]

layout=go.Layout(title='Writing Score')

fig= go.Figure(data=data, layout=layout)

iplot(fig)
df.head()
fig = {

    "data": [

        {

            "type": 'violin',

            "x": df['race/ethnicity'] [ df['lunch'] == 'standard' ],

            "y": df['math score'] [ df['lunch'] == 'standard' ],

            "legendgroup": 'Yes',

            "scalegroup": 'Yes',

            "name": 'Standard Lunch',

            "side": 'negative',

            "box": {

                "visible": True

            },

            "meanline": {

                "visible": True

            },

            "line": {

                "color": 'blue'

            }

        },

        {

            "type": 'violin',

            "x": df['race/ethnicity'] [ df['lunch'] == 'free/reduced' ],

            "y": df['math score'] [ df['lunch'] == 'free/reduced' ],

            "legendgroup": 'No',

            "scalegroup": 'No',

            "name": 'Free/Reduced Lunch',

            "side": 'positive',

            "box": {

                "visible": True

            },

            "meanline": {

                "visible": True

            },

            "line": {

                "color": 'green'

            }

        }

    ],

    "layout" : {"title":"Violin Plot For Math Scores",

        "yaxis": {

            "zeroline": False,

        },

        "violingap": 0,

        "violinmode": "overlay"

    }

}



iplot(fig)
fig = {

    "data": [

        {

            "type": 'violin',

            "x": df['race/ethnicity'] [ df['lunch'] == 'standard' ],

            "y": df['reading score'] [ df['lunch'] == 'standard' ],

            "legendgroup": 'Yes',

            "scalegroup": 'Yes',

            "name": 'Standard Lunch',

            "side": 'negative',

            "box": {

                "visible": True

            },

            "meanline": {

                "visible": True

            },

            "line": {

                "color": 'blue'

            }

        },

        {

            "type": 'violin',

            "x": df['race/ethnicity'] [ df['lunch'] == 'free/reduced' ],

            "y": df['reading score'] [ df['lunch'] == 'free/reduced' ],

            "legendgroup": 'No',

            "scalegroup": 'No',

            "name": 'Free/Reduced Lunch',

            "side": 'positive',

            "box": {

                "visible": True

            },

            "meanline": {

                "visible": True

            },

            "line": {

                "color": 'green'

            }

        }

    ],

    "layout" : {"title":"Violin Plot For Reading Scores",

        "yaxis": {

            "zeroline": False,

        },

        "violingap": 0,

        "violinmode": "overlay"

    }

}



iplot(fig)
fig = {

    "data": [

        {

            "type": 'violin',

            "x": df['race/ethnicity'] [ df['lunch'] == 'standard' ],

            "y": df['writing score'] [ df['lunch'] == 'standard' ],

            "legendgroup": 'Yes',

            "scalegroup": 'Yes',

            "name": 'Standard Lunch',

            "side": 'negative',

            "box": {

                "visible": True

            },

            "meanline": {

                "visible": True

            },

            "line": {

                "color": 'blue'

            }

        },

        {

            "type": 'violin',

            "x": df['race/ethnicity'] [ df['lunch'] == 'free/reduced' ],

            "y": df['writing score'] [ df['lunch'] == 'free/reduced' ],

            "legendgroup": 'No',

            "scalegroup": 'No',

            "name": 'Free/Reduced Lunch',

            "side": 'positive',

            "box": {

                "visible": True

            },

            "meanline": {

                "visible": True

            },

            "line": {

                "color": 'green'

            }

        }

    ],

    "layout" : {"title":"Violin Plot For Writing Scores",

        "yaxis": {

            "zeroline": False,

        },

        "violingap": 0,

        "violinmode": "overlay"

    }

}



iplot(fig)