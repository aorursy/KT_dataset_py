import pandas as pd

import numpy as np



data_df = pd.read_csv('../input/Team_Records.csv')



#A peek at what we've got

data_df.head(10)
#Remove the asterisk to keep team names consistent

data_df['Team'] = data_df['Team'].str.replace('*', '')



#Remove data for the ongoing season

data_df = data_df[data_df['Season'] != '2017-18']



#Add a 'date' so we can make a time series graph easily

data_df['Date'] = data_df['Season'].str[:4].astype(int)+1



#Fill in NaN playoff messages

data_df['Playoffs'].fillna('Did not make Playoffs', inplace=True)



#A peek of what we've got now

data_df.head(10)
#Behold a list of every team in the dataset

print(list(data_df['Team'].unique()))
import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode(connected=True)

import random



def randomColor():

    (r,g,b) = [str(random.randint(1,255)), str(random.randint(1,255)), str(random.randint(1,255))]

    color = 'rgb(' + r + ','+ g + ',' + b + ')'

    return color



def updateVisibility(selected):

    visibilityValues = []

    for team in list(data_df['Team'].unique()):

        if team == selected:

            visibilityValues.append(True)

        else:

            visibilityValues.append(False)

    return visibilityValues



data = []

buttons_data = []

for team in list(data_df['Team'].unique()):

    data.append(go.Scatter(

        x = np.array(data_df[data_df.Team == team]['Date']),

        y = np.array(data_df[data_df.Team == team]['W/L%']),

        mode='lines+markers',

        line=dict(

            color=randomColor(),

            width=1

        ),

        name=team,

        text=data_df[data_df.Team == team]['Playoffs'],

        visible=(team =='Boston Celtics')

    ))

    buttons_data.append(dict(

        label = (team + ' (' + data_df[data_df.Team == team]['Date'].apply(str).iloc[-1] + '-' + 

                 data_df[data_df.Team == team]['Date'].apply(str).iloc[0] + ')'),

        method = 'update',

        args = [{'visible': updateVisibility(team)}]

    ))

    



updatemenus = list([

    dict(active=0,

         buttons= buttons_data,

         direction = 'down',

         pad = {'r': 10, 't': 10},

         showactive = True,

         x = 0.65,

         xanchor = 'left',

         y = 1.18,

         yanchor = 'top'

    )

])

    

layout = dict(

    title='Team Record History',

    updatemenus = updatemenus,

    xaxis=dict(

        rangeslider=dict(),

        type='date',

        autorange=True

    ),

    annotations=go.Annotations([

        go.Annotation(

            x=0.5004254919715793,

            y=-0.46191064079952971,

            showarrow=False,

            text='Year',

            xref='paper',

            yref='paper',

            font=dict(

                size=16,

            ),

        ),

        go.Annotation(

            x=-0.06944728761514841,

            y=0.4714285714285711,

            showarrow=False,

            text='Season Win %',

            textangle=-90,

            xref='paper',

            yref='paper'

        )

    ]),

)



fig = dict(data=data, layout=layout)

iplot(fig)