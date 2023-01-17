# Data Processing

import numpy as np

import pandas as pd



# Basic Visualization tools

import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300

import seaborn as sns

sns.set_style("whitegrid")



data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
import missingno as msno

msno.matrix(data)
data.head()
# type 1 : default



import plotly.graph_objects as go



fig = go.Figure()



fig.add_trace(

    go.Table(header=dict(values=data.columns), 

             cells=dict(values=data.head(3).T))

)



fig.show()
# type 2 : change color



fig = go.Figure()



fig.add_trace(

    go.Table(header=dict(values=data.columns, fill_color='paleturquoise'),

            cells=dict(values=data.head(3).T, fill_color='lavender'))

)



fig.show()
fig, ax = plt.subplots(2,1,figsize=(16, 6))

# type 1 : use hue parameter

sns.countplot(x='math score', data=data, hue='gender', palette='Set1', alpha=0.7, ax=ax[0])

plt.legend()

plt.xticks(rotation='vertical')



#type 2: use value_counts + barplot / hard to divide like hue 

sns.barplot(x=data['math score'].value_counts().index, y=data['math score'].value_counts(), ax=ax[1])

plt.show()
import plotly.express as px



#type 1 : Stacked graph (default)



fig = px.histogram(data, x="math score", y="math score", color="gender")

fig.show()



#type 2 : group



fig = px.histogram(data, x="math score", y="math score", color="gender")

fig.update_layout(barmode='group')

fig.show()
fig = px.histogram(data, x="math score", y="math score", color="gender", 

                   marginal="box", # or violin, rug

                  ) 

fig.show()
# type 1 : default view

fig, ax = plt.subplots(1,1,figsize=(16, 6))

sns.distplot(data['math score'])

plt.show()
# type 2 : draw 2 graph with label

fig, ax = plt.subplots(1,1,figsize=(16, 6))

sns.distplot(data[data['gender']=='female']['math score'], color='purple', ax=ax, label='female')

sns.distplot(data[data['gender']=='male']['math score'], color='orange', ax=ax, label='male')

plt.xticks(rotation='vertical')

plt.legend()

plt.show()
# type 3 : add avg line

fig, ax = plt.subplots(1,1,figsize=(16, 6))

sns.distplot(data[data['gender']=='female']['math score'], color='purple', ax=ax, label='female')

sns.distplot(data[data['gender']=='male']['math score'], color='orange', ax=ax, label='male')



# avg line

plt.axvline(data[data['gender']=='female']['math score'].mean(), color='purple')

plt.axvline(data[data['gender']=='male']['math score'].mean(), color='orange')



plt.legend()

plt.xticks(rotation='vertical')

plt.show()

# type 1 : default

import plotly.figure_factory as ff



fig = ff.create_distplot([data[data['gender']=='male']['math score'], data[data['gender']=='female']['math score']], ['male', 'female'])

fig.show()
# type 2 : add color and change bin width



fig = ff.create_distplot([data[data['gender']=='male']['math score'], data[data['gender']=='female']['math score']], 

                         ['male', 'female'],

                         colors = ['#2BCDC1', '#F66095'],

                         bin_size = [2, 2]

                        )

fig.show()
# type 3 : change view (rug, hist, curve)



fig = ff.create_distplot([data[data['gender']=='male']['math score'], data[data['gender']=='female']['math score']], 

                         ['male', 'female'],

                         colors = ['#2BCDC1', '#F66095'],

                         bin_size = [2, 2],

                         show_rug=False, #rug 

                         show_hist=False, #hist

                         show_curve=True # curve

                        )

fig.show()
fig, ax = plt.subplots(2, 2, figsize=(20, 20))



# type 1 : default scatter plot 

sns.scatterplot(data=data, x='writing score', y='reading score', alpha=0.7, ax=ax[0][0])



# type 2 : with hue

sns.scatterplot(data=data, x='writing score', y='reading score', hue='parental level of education', alpha=0.7, ax=ax[0][1])



# type 3 : with style & color

sns.scatterplot(data=data, x='writing score', y='reading score',style='gender', color='royalblue', alpha=0.5, ax=ax[1][0])



# type 4 : with size & color 

sns.scatterplot(data=data, x='writing score', y='reading score',size='parental level of education', color='brown', alpha=0.5, ax=ax[1][1])



plt.show()
fig, ax = plt.subplots(1, 1, figsize=(12, 12))

sns.scatterplot(data=data, x='writing score', y='reading score', style='gender',hue='parental level of education',size='parental level of education', alpha=0.7)

plt.show()
# type 1 : default

fig = px.scatter(data, x='writing score', y='reading score')

fig.show()
# type 2 : use color as seaborn hue

fig = px.scatter(data, x='writing score', y='reading score', color='parental level of education', opacity=0.5)

fig.show()
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

data['parental level of education'] = LE.fit_transform(data['parental level of education'])
# type 3 : with size & color 

fig = px.scatter(data, x='writing score', y='reading score', 

                 color='parental level of education',

                 size='parental level of education',

                )

fig.show()
# type 1 : default boxplot & stripplot



fig, ax = plt.subplots(2, 2, figsize=(14, 12))



# boxplot

sns.boxplot(x='parental level of education', y='math score', data=data, ax=ax[0][0])



# stripplot

sns.stripplot(x='parental level of education', y='math score', data=data, ax=ax[0][1], alpha=0.5)



# swarmplot 

sns.swarmplot(x='parental level of education', y='math score', data=data, ax=ax[1][0], alpha=0.7)



# both

sns.boxplot(x='parental level of education', y='math score', data=data, ax=ax[1][1])

sns.stripplot(x='parental level of education', y='math score', data=data, ax=ax[1][1], alpha=0.3)



plt.tight_layout()

plt.show()
# type 2 : we can add hue parameter

fig, ax = plt.subplots(1, 1, figsize=(12,6))

sns.boxplot(x='race/ethnicity', y='math score', hue='gender', palette='Set2', data=data, ax=ax)

plt.show()
# type 1 : default boxplot

fig = px.box(data, x='parental level of education', y='math score')

fig.show()
# type 2 : boxplot with stripplot

fig = px.box(data, x='parental level of education', y='math score', points="all")

fig.show()
# type 3 : boxplot with stripplot + color

fig = px.box(data, x='parental level of education', y='math score', color='gender', points="all")

fig.show()
# type 1 : compare with boxplot



fig, ax = plt.subplots(1, 2, figsize=(28, 6))



# boxplot

sns.boxplot(x='parental level of education', y='math score', data=data, ax=ax[0])



# violinplot

sns.violinplot(x='parental level of education', y='math score', data=data, ax=ax[1])



plt.show()
# type 2 : hue parameter



fig, ax = plt.subplots(1, 2, figsize=(18, 6))

sns.violinplot(x='parental level of education', y='math score', hue='gender', data=data, ax=ax[0])





# type 3 : hue + split

sns.violinplot(x='parental level of education', y='math score', hue='gender', data=data, split=True, ax=ax[1])

plt.show()
# type 1 : default violinplot



fig = px.violin(data, x='parental level of education', y='math score')

fig.show()
# type 2 : color add violin

fig = px.violin(data, x='parental level of education', y='math score', 

                color='gender'

            )

fig.show()
# type 3 : color + violinmode

fig = px.violin(data, x='parental level of education', y='math score', 

                color='gender',

                violinmode='overlay'

            )

fig.show()
# type 4 : go.box usage

fig = go.Figure()



fig.add_trace(

    go.Violin(x=data['parental level of education'], y=data['math score'],

              box_visible=True, # with box plot

              line_color='black',

              meanline_visible=True, # with mean value

              fillcolor='lightseagreen', 

              opacity=0.6)

)



fig.show()

# type 1 : default

sns.set(style="white", color_codes=True) # suitable theme for jointplot

sns.jointplot(data=data, x='writing score', y='reading score', alpha=0.7)

plt.show()
# type 2 : many types (reg, hex, kde)

sns.jointplot(data=data, x='writing score', y='reading score', kind='reg', color='skyblue')

sns.jointplot(data=data, x='writing score', y='reading score', kind='hex', color='gold')

sns.jointplot(data=data, x='writing score', y='reading score', kind='kde', color='forestgreen' )

plt.show()
# type 1 : 2d histogram + such as heatmap



fig = go.Figure()



fig.add_trace(go.Histogram2d(

    x=data['writing score'], y=data['reading score'], 

    nbinsx=30, nbinsy=30,

    colorscale = 'Blues'

))



fig.show()
# type 2 : 2d histogram countour



fig = go.Figure()



fig.add_trace(go.Histogram2dContour(

    x=data['writing score'], y=data['reading score'], 

    nbinsx=30, nbinsy=30,

    colorscale = 'Blues'

))



fig.show()
# type 3 : Histogram2dContour + Scatter + Histogram 



fig = go.Figure()



fig.add_trace(go.Histogram2dContour(

    x=data['writing score'], y=data['reading score'], 

    nbinsx=30, nbinsy=30,

    colorscale = 'Blues',

    xaxis = 'x', yaxis= 'y'

))





fig.add_trace(go.Scatter(

        x=data['writing score'], y=data['reading score'], 

        xaxis = 'x',

        yaxis = 'y',

        mode = 'markers',

        marker = dict(

            color = 'rgba(0,0,0,0.3)',

            size = 3

        )

    ))



fig.add_trace(go.Histogram(

        y=data['reading score'], 

        xaxis = 'x2',

        marker = dict(

            color = 'rgba(0,0,0,1)'

        )

    ))



fig.add_trace(go.Histogram(

        x=data['writing score'],

        yaxis = 'y2',

        marker = dict(

            color = 'rgba(0,0,0,1)'

        )

    ))



fig.update_layout(

    autosize = False,

    xaxis = dict(

        zeroline = False,

        domain = [0,0.85],

        showgrid = False

    ),

    yaxis = dict(

        zeroline = False,

        domain = [0,0.85],

        showgrid = False

    ),

    xaxis2 = dict(

        zeroline = False,

        domain = [0.85,1],

        showgrid = False

    ),

    yaxis2 = dict(

        zeroline = False,

        domain = [0.85,1],

        showgrid = False

    ),

    height = 600,

    width = 600,

    bargap = 0,

    hovermode = 'closest',

    showlegend = False

)



fig.show()
#type4 : marginal_x & marginal_y

fig = px.scatter(data, x='writing score', y='reading score', marginal_y="rug", marginal_x="histogram")

fig.show()