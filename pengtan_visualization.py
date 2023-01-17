#Load the librarys

import plotly

import plotly.express as px

import pandas as pd #To work with dataset

import numpy as np #Math library

import seaborn as sns #Graph library that use matplot in background

import matplotlib.pyplot as plt #to plot some parameters in seaborn

# it's a library that we work with plotly

import plotly.offline as py 

py.init_notebook_mode(connected=True) # this code, allow us to work with offline plotly version

import plotly.graph_objs as go # it's like "plt" of matplot

import plotly.tools as tls # It's useful to we get some tools of plotly

import warnings # This library will be used to ignore some warnings

from collections import Counter # To do counter of some features
train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)

train
Data_df = pd.DataFrame([['train', len(train)], ['test', len(test)]], columns=['name', 'value'])

trace0 = go.Bar(

        x=[Data_df.name[0]],

        y=[Data_df.value[0]],

        name='train number',

        marker_color='red'

)

trace1 = go.Bar(

        x=[Data_df.name[1]],

        y=[Data_df.value[1]],

        name='test number',

        marker_color='blue'

)

trace2 = go.Bar(

        x = [f"{test['seq_length'].value_counts().index[0]}"],

        y = [test['seq_length'].value_counts().values[0]],

        name='Sequence 130 Number',

        marker_color='indianred'

)

trace3 = go.Bar(

        x = [f"{test['seq_length'].value_counts().index[1]}"],

        y = [test['seq_length'].value_counts().values[1]],

        name='Sequence 107 Number',

        marker_color='orange'

)



#Creating the grid

fig = plotly.subplots.make_subplots(rows=1, cols=2, specs=[[{}, {}]],

                          subplot_titles=('The amount of Train and Test','Sequence Length in public test set'))



#setting the figs

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 2)



fig['layout'].update(showlegend=True, title='Data Distribution', bargap=0.05)

py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')
train['mean_reactivity'] = train['reactivity'].apply(lambda x: np.mean(x))

train['mean_deg_Mg_pH10'] = train['deg_Mg_pH10'].apply(lambda x: np.mean(x))

train['mean_deg_Mg_50C'] = train['deg_Mg_50C'].apply(lambda x: np.mean(x))

train['mean_deg_pH10'] = train['deg_pH10'].apply(lambda x: np.mean(x))

train['mean_deg_50C'] = train['deg_50C'].apply(lambda x: np.mean(x))

#First plot

trace0 = go.Histogram(

    x=train['mean_reactivity'],

    histnorm='probability',

    name="reactivity"

)

#Second plot

trace1 = go.Histogram(

    x=train['mean_deg_Mg_pH10'],

    histnorm='probability',

    name="deg_Mg_pH10"

)

#Third plot

trace2 = go.Histogram(

    x=train['mean_deg_Mg_50C'],

    histnorm='probability',

    name="deg_Mg_50C"

)

#Fourth plot

trace3 = go.Histogram(

    x=train['mean_deg_pH10'],

    histnorm='probability',

    name="deg_pH10"

)

#Fivth plot

trace4 = go.Histogram(

    x=train['mean_deg_50C'],

    histnorm='probability',

    name="deg_50C"

)



#Creating the grid

fig = tls.make_subplots(rows=3, cols=2, specs=[[{}, {}], [{}, {}], [{'colspan': 2}, None]],

                          subplot_titles=('Mean Reactivity',

                                          'deg_Mg_pH10', 

                                          'Mean deg_Mg_50C',

                                          'Mean deg_pH10',

                                          'Distribution of Mean deg_50C in training set'))



#setting the figs

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 2, 2)

fig.append_trace(trace4, 3, 1)

fig['layout'].update(showlegend=True, title='Predictive Values Distribuition', bargap=0.05)

py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')
trace0 = go.Histogram(

    x=train['signal_to_noise'],

    histnorm='probability',

    name="Good Credit"

)



data = [trace0]



layout = go.Layout(

    yaxis=dict(

        title='Count'

    ),

    xaxis=dict(

        title='Signal_to_noise'

    ),

    title='Signal_to_noise Distribution'

)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='grouped-bar')
sns.pairplot(data=train,

             vars=['mean_reactivity',

                   'mean_deg_Mg_pH10',

                   'mean_deg_Mg_50C',

                   'mean_deg_pH10',

                   'mean_deg_50C'],

            hue='SN_filter')

plt.show()
trace0 = go.Pie(

            labels=train['SN_filter'].value_counts().index, 

            values=train['SN_filter'].value_counts().values

)



data = [trace0]



layout = go.Layout(

    title='SN_filter Bar Chart'

)



fig = go.Figure(data=data, layout=layout)



fig.show()