import pandas as pd

import numpy as np



data = pd.read_csv('../input/nba-players-stats/Seasons_Stats.csv')



#Demo

data.tail(10)
#Quick maths

(data['MPG'], data['PPG'], data ['APG'], 

 data ['RPG'], data['SPG'], data['BPG'])= [data['MP']/data['G'], data['PTS']/data['G'], data['AST']/data['G'],

                                           data['ORB']/data['G']+data['DRB']/data['G'], data['STL']/data['G'],

                                           data['BLK']/data['G']]

#Demo Again

data.tail(10)
#Player-finding function

def getCorePlayers(df, team, yearStart, yearEnd, minCutoff=20):

    df_team = df[df['Tm']==team]

    corePlayerList = []

    for year in range(yearStart, yearEnd+1):

        corePlayerList += list(df_team[(df_team['Year']==year) & (df_team['MPG']>minCutoff)]['Player'])

    corePlayerList = list(set(corePlayerList))

    return corePlayerList



#Player data-finding function

def getCorePlayersData(df, team, yearStart, yearEnd, minCutoff=20):

    corePlayerList = getCorePlayers(df, team, yearStart, yearEnd, minCutoff)

    df_team = df[df['Tm']==team]

    corePlayersData = df_team[(df_team['Year']>=yearStart) & (df_team['Year']<=yearEnd) 

                              & (df_team['Player'].isin(corePlayerList))]

    return corePlayersData



#Season record data-finding function

def getSeasonRecords(df, team, yearStart, yearEnd):

    df_team_records = df[df['Team'].str.contains(team)]

    df_team_records['Year'] = df_team_records['Season'].str[0:4].astype(int) + 1

    df_team_records = df_team_records[(df_team_records['Year']>=yearStart) & (df_team_records['Year']<=yearEnd)]

    return df_team_records



#Demo

getCorePlayersData(data, 'GSW', 2016, 2017, 32).sort_values(['Player', 'Year'], ascending=[True,True])
records = pd.read_csv('../input/nba-season-records-from-every-year/Team_Records.csv')

GSW = getCorePlayersData(data, 'GSW', 2011, 2017, 32).sort_values(['Player', 'Year'], ascending=[True,True])

MIA = getCorePlayersData(data, 'MIA', 2004, 2014, 32).sort_values(['Player', 'Year'], ascending=[True,True])

SAS = getCorePlayersData(data, 'SAS', 1997, 2017, 31).sort_values(['Player', 'Year'], ascending=[True,True])

LAL = getCorePlayersData(data, 'LAL', 1997, 2012, 35).sort_values(['Player', 'Year'], ascending=[True,True])

CHI = getCorePlayersData(data, 'CHI', 1985, 2000, 35).sort_values(['Player', 'Year'], ascending=[True,True])

LAL2 = getCorePlayersData(data, 'LAL', 1980, 1990, 35).sort_values(['Player', 'Year'], ascending=[True,True])

BOS = getCorePlayersData(data, 'BOS', 1980, 1988, 35).sort_values(['Player', 'Year'], ascending=[True,True])

(GSW.name, GSW.color1, GSW.color2, GSW.records) = ['GSW', 'rgb(255,205,52)', 'rgb(36,62,144)', 

                                                   getSeasonRecords(records, 'Golden State Warriors', 2011, 2017)]

(MIA.name, MIA.color1, MIA.color2, MIA.records) = ['MIA', 'rgb(152,0,46)', 'rgb(249,160,27)', 

                                                   getSeasonRecords(records, 'Miami Heat', 2004, 2014)]

(SAS.name, SAS.color1, SAS.color2, SAS.records) = ['SAS', 'rgb(196,206,212)', 'rgb(0,0,0)', 

                                                   getSeasonRecords(records, 'San Antonio Spurs', 1997, 2017)]

(LAL.name, LAL.color1, LAL.color2, LAL.records) = ['LAL', 'rgb(253,185,39)', 'rgb(85,37,131)', 

                                                   getSeasonRecords(records, 'Los Angeles Lakers', 1997, 2012)]

(CHI.name, CHI.color1, CHI.color2, CHI.records) = ['CHI', 'rgb(206,17,65)', 'rgb(0,0,0)', 

                                                   getSeasonRecords(records, 'Chicago Bulls', 1985, 2000)]

(LAL2.name, LAL2.color1, LAL2.color2, LAL2.records) = ['LAL2', 'rgb(253,185,39)', 'rgb(85,37,131)', 

                                                       getSeasonRecords(records, 'Los Angeles Lakers', 1980, 1990)]

(BOS.name, BOS.color1, BOS.color2, BOS.records) = ['BOS', 'rgb(0,130,72)', 'rgb(186,150,83)', 

                                                   getSeasonRecords(records, 'Boston Celtics', 1980, 1988)]



#Greatest Teams of All Time

GTOAT = [GSW, MIA, SAS, LAL, CHI, LAL2, BOS]



import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode(connected=True)

import random



graphData = []



#Returns a random rgb color string

def randomColor():

    (r,g,b) = [str(random.randint(1,255)), str(random.randint(1,255)), str(random.randint(1,255))]

    color = 'rgb(' + r + ','+ g + ',' + b + ')'

    return color



#Helps update the data when a new stat option is picked from the dropdown

def updateData(category):

    yValues = []

    for team in GTOAT:

        yValues.append(team.records['W/L%'])

        for player in list(team['Player'].unique()):

            yValues.append(team[team['Player']==player][category])

    return yValues

    

#Hides and shows data when a new team is picked from the dropdown

def updateVisibility(selected):

    visibilityValues = []

    for team in GTOAT:

        if team.name == selected:

            #show record bars

            visibilityValues.append(True)

            #show player data

            for player in list(team['Player'].unique()):

                visibilityValues.append(True)

        else:

            visibilityValues.append(False)

            for player in list(team['Player'].unique()):

                visibilityValues.append(False)

    return visibilityValues

    

for team in GTOAT:

    graphData.append(go.Bar(

        x=team.records['Year'],

        y=team.records['W/L%'],

        name='Record', 

        text=team.records['Playoffs'],

        textposition = 'auto',

        textfont=dict(

            color='rgba(75, 85, 102,0.7)'

        ),

        marker=dict(

            color='rgba(158,202,225,0.3)',

            line=dict(

                color='rgba(8,48,107,0.3)',

                width=1.5),

        ),

        yaxis='y2',

        visible=(team.name=='GSW')

    ))

    for player in list(team['Player'].unique()):

        graphData.append(go.Scatter(

            x=team[team['Player']==player]['Year'],

            y=team[team['Player']==player]['PPG'],

            mode='lines+markers',

            line=dict(

                color=randomColor(),

                width=5

            ),

            marker = dict(

                size = 15,

                color = team.color1,

                line = dict(

                    color = team.color2,

                    width = 2

                ),

            ),

            name=player,

            text=player,

            visible=(team.name=='GSW')

        ))



updatemenus = list([

    dict(active=0,

         buttons=list([   

            dict(label = 'PPG',

                 method = 'update',

                 args = [{'y': updateData('PPG')},

                         {'title': 'Points Per Game'}]),

            dict(label = 'APG',

                 method = 'update',

                 args = [{'y': updateData('APG')},

                         {'title': 'Assists Per Game'}]),

            dict(label = 'RPG',

                 method = 'update',

                 args = [{'y': updateData('RPG')},

                         {'title': 'Rebounds Per Game'}]),

            dict(label = 'SPG',

                 method = 'update',

                 args = [{'y': updateData('SPG')},

                         {'title': 'Steals Per Game'}]),

            dict(label = 'BPG',

                 method = 'update',

                 args = [{'y': updateData('BPG')},

                         {'title': 'Blocks Per Game'}]),

            dict(label = 'PER',

                 method = 'update',

                 args = [{'y': updateData('PER')},

                         {'title': 'Player Efficiency Rating'}]),

            dict(label = 'VORP',

                 method = 'update',

                 args = [{'y': updateData('VORP')},

                         {'title': 'Value Over Replacement Player'}]),

            dict(label = 'FG%',

                 method = 'update',

                 args = [{'y': updateData('FG%')},

                         {'title': 'Field Goal Percentage'}]),

            dict(label = 'FT%',

                 method = 'update',

                 args = [{'y': updateData('FT%')},

                         {'title': 'Free Throw Percentage'}]),

            dict(label = '3P%',

                 method = 'update',

                 args = [{'y': updateData('3P%')},

                         {'title': '3-Point Field Goal Percentage'}]),

            dict(label = 'TS%',

                 method = 'update',

                 args = [{'y': updateData('TS%')},

                         {'title': 'True Shooting Percentage'}]),

        ]),

        direction = 'down',

        pad = {'r': 10, 't': 10},

        showactive = True,

        x = 0.1,

        xanchor = 'left',

        y = 1.12,

        yanchor = 'top'

    ),

    dict(active=0,

         buttons=list([   

            dict(label = 'Golden State Warriors (2011-2017)',

                 method = 'update',

                 args = [{'visible': updateVisibility('GSW')}]),

            dict(label = 'Miami Heat (2004-2014)',

                 method = 'update',

                 args = [{'visible': updateVisibility('MIA')}]),

            dict(label = 'San Antonio Spurs (1997-2017)',

                 method = 'update',

                 args = [{'visible': updateVisibility('SAS')}]),

            dict(label = 'Los Angeles Lakers (1997-2012)',

                 method = 'update',

                 args = [{'visible': updateVisibility('LAL')}]),

            dict(label = 'Chicago Bulls (1985-2000)',

                 method = 'update',

                 args = [{'visible': updateVisibility('CHI')}]),

            dict(label = 'Los Angeles Lakers (1980-1990)',

                 method = 'update',

                 args = [{'visible': updateVisibility('LAL2')}]),

            dict(label = 'Boston Celtics (1980-1988)',

                 method = 'update',

                 args = [{'visible': updateVisibility('BOS')}]),

        ]),

        direction = 'down',

        pad = {'r': 10, 't': 10},

        showactive = True,

        x = 0.55,

        xanchor = 'left',

        y = 1.12,

        yanchor = 'top'

    )

])



layout = go.Layout(

    hovermode = 'closest',

    updatemenus = updatemenus,

    yaxis2=dict(

        title='Season Win %',

        titlefont=dict(

            color='rgba(8,48,107,0.5)',

            size=16

        ),

        tickfont=dict(

            color='rgba(8,48,107,0.5)'

        ),

        overlaying='y',

        side='right'

    ),

    annotations=go.Annotations([

        go.Annotation(

            x=0.5004254919715793,

            y=-0.16191064079952971,

            showarrow=False,

            text='Year',

            xref='paper',

            yref='paper',

            font=dict(

                size=16,

            ),

        ),

        go.Annotation(

            x=1.3,

            y=1.05,

            align="right",

            valign="top",

            text='Core Players & Records',

            showarrow=False,

            xref="paper",

            yref="paper",

            xanchor="middle",

            yanchor="top"

        )

    ]),

    legend=dict(

        x=1.1

    ),

    autosize=True,

    margin=go.Margin(

        b=100

    ),

    height=600,

    title='Points Per Game'

)



figure = dict(data = graphData, layout = layout)

iplot(figure)