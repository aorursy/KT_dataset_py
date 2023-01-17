import pandas as pd

import numpy as np



import plotly.plotly as py

from plotly.graph_objs import *

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



data = pd.read_csv('../input/shot_logs.csv')



playerIDList= list(data['player_id'].unique())

defenderIDList = list(data['CLOSEST_DEFENDER_PLAYER_ID'].unique())

attackerIDList = list(data['player_id'].unique())



playerData = pd.DataFrame(index=playerIDList, columns=['player', 'made', 'missed', 'fg_percentage', 

                                                       'made_against', 'missed_against', 'fg_percentage_against'])



for defenderID in defenderIDList:

    name= data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['CLOSEST_DEFENDER'].iloc[0]

    made = np.sum(data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['SHOT_RESULT']=='made')

    missed = np.sum(data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['SHOT_RESULT']=='missed')

    percentage = made/(made+missed)

    playerData.at[defenderID, 'player'] = name

    playerData.at[defenderID, 'made_against'] = made

    playerData.at[defenderID, 'missed_against'] = missed

    playerData.at[defenderID, 'fg_percentage_against'] = percentage



for attackerID in attackerIDList:

    made = np.sum(data[(data['player_id'] == attackerID)]['SHOT_RESULT']=='made')

    missed = np.sum(data[(data['player_id'] == attackerID)]['SHOT_RESULT']=='missed')

    percentage = made/(made+missed)

    playerData.at[attackerID, 'made'] = made

    playerData.at[attackerID, 'missed'] = missed

    playerData.at[attackerID, 'fg_percentage'] = percentage

    

newPlayerData = playerData.sort_values('fg_percentage_against')

newPlayerData2 = newPlayerData.drop(newPlayerData[newPlayerData.missed_against < 200].index)
newPlayerData2
ESPNRankTop30 = ['James, LeBron', 'Paul, Chris', 'Davis, Anthony', 'Westbrook, Russell', 'Griffin, Blake', 'Curry, Stephen',

                'Love, Kevin', 'Durant, Kevin', 'Harden, James', 'Howard, Dwight', 'Anthony, Carmelo', 'Noah, Joakim',

                'Aldridge, LaMarcus', 'Gasol, Marc', 'Parker, Tony', 'Lillard, Damian', 'Nowitzki, Dirk', 'Wall, John',

                'Cousins, DeMarcus', 'Bosh, Chris', 'Duncan, Tim', 'Jefferson, Al', 'Irving, Kyrie', 'Leonard, Kawhi', 

                'Ibaka, Serge', 'Horford, Al', 'Dragic, Goran', 'Rose, Derrick', 'Lowry, Kyle', 'Drummond, Andre']

newPlayerData3 = newPlayerData[newPlayerData['player'].isin(ESPNRankTop30)]

newPlayerData3 = newPlayerData3.sort_values('player')

newPlayerData3['ranking'] = 0

for i in range(len(ESPNRankTop30)):

    newPlayerData3.loc[newPlayerData3['player'] == ESPNRankTop30[i], 'ranking'] = str(i+1)
newPlayerData3
line = Scatter(

            x= [0,1],

            y= [0,1],

            marker = dict(

                size=1,

                color='rgba(200, 200, 200, .5)'

            ),

            name = "Line of Neutrality"

        )



trace1 = Scatter(

            x=newPlayerData2['fg_percentage'],

            y=newPlayerData2['fg_percentage_against'],

            mode = 'markers',

            marker = dict(

                size = 10,

                color = 'rgba(132, 123, 255, .9)',

                line = dict(

                    width = 2,

                )

            ), 

            name='League',

            text= newPlayerData2['player']

        )



trace2 = Scatter(

            x=newPlayerData3['fg_percentage'],

            y=newPlayerData3['fg_percentage_against'],

            mode = 'markers',

            marker = dict(

                size = 10,

                color = 'rgba(255, 123, 132, .9)',

                line = dict(

                    width = 2,

                ),

            ),

            name='#NBARank Top 30',

            text = newPlayerData3['player'] + ' (#' + newPlayerData3['ranking'] + ')'

        )



data = [line, trace1, trace2]



layout = Layout(

    hovermode = 'closest',

    annotations=Annotations([

        Annotation(

            x=0.5004254919715793,

            y=-0.16191064079952971,

            showarrow=False,

            text='Made Field Goal %',

            xref='paper',

            yref='paper'

        ),

        Annotation(

            x=-0.05944728761514841,

            y=0.4714285714285711,

            showarrow=False,

            text='Allowed Field Goal %',

            textangle=-90,

            xref='paper',

            yref='paper'

        )

    ]),

    autosize=True,

    margin=Margin(

        b=100

    ),

    title='Made Vs. Allowed FG%',

    xaxis=XAxis(

        autorange=False,

        range=[0.35, 0.72],

        type='linear'

    ),

    yaxis=YAxis(

        autorange=False,

        range=[0.35, 0.55],

        type='linear'

    )

)



graph = Figure(data=data, layout=layout)

iplot(graph)
data = pd.read_csv('../input/shot_logs.csv')

defenderIDList = list(data['CLOSEST_DEFENDER_PLAYER_ID'].unique())

attackerIDList = list(data['player_id'].unique())



playerData = pd.DataFrame(index=playerIDList, columns=['player', 'made', 'missed', 'fg_percentage', 

                                                       'fg_distance'])



for attackerID in attackerIDList:

    name= data[(data['player_id'] == attackerID)]['player_name'].iloc[0]

    spacePos = name.find(' ')

    firstname = name[0].upper() + name[1:spacePos]

    lastname = name[spacePos+1].upper() + name[spacePos+2:]

    name = firstname + ' ' + lastname

    made = np.sum(data[(data['player_id'] == attackerID)]['SHOT_RESULT']=='made')

    missed = np.sum(data[(data['player_id'] == attackerID)]['SHOT_RESULT']=='missed')

    percentage = made/(made+missed)

    averageDist = np.mean(data[(data['player_id'] == attackerID)]['SHOT_DIST'])

    playerData.at[attackerID, 'player'] = name

    playerData.at[attackerID, 'made'] = made

    playerData.at[attackerID, 'missed'] = missed

    playerData.at[attackerID, 'fg_percentage'] = percentage

    playerData.at[attackerID, 'fg_distance'] = averageDist

    

newPlayerData = playerData.sort_values('fg_distance', ascending=False)

newPlayerData2 = newPlayerData.drop(newPlayerData[newPlayerData.made < 200].index)

newPlayerData2
import plotly



import plotly.plotly as py

from plotly.graph_objs import *

from ipywidgets import widgets 

from IPython.display import display, clear_output, Image

from plotly.graph_objs import *

from plotly.widgets import GraphWidget



ESPNRankTop30 = ['Lebron James', 'Chris Paul', 'Anthony Davis', 'Russell Westbrook', 'Blake Griffin', 'Stephen Curry',

                'Kevin Love', 'Kevin Durant', 'James Harden', 'Dwight Howard', 'Carmelo Anthony', 'Joakim Noah',

                'Lamarcus Aldridge', 'Marc Gasol', 'Tony Parker', 'Damian Lillard', 'Dirk Nowtizski', 'John Wall',

                'Demarcus Cousins', 'Chris Bosh', 'Tim Duncan', 'Al Jefferson', 'Kyrie Irving', 'Kawhi Leonard', 

                'Serge Ibaka', 'Al Horford', 'Goran Dragic', 'Derrick Rose', 'Kyle Lowry', 'Andre Drummond']



trace1 = Scatter(

            x=newPlayerData2['fg_distance'],

            y=newPlayerData2['fg_percentage'],

            mode = 'markers',

            marker = dict(

                size = newPlayerData2['made']/20,

                color = 'rgba(132, 123, 255, .9)',

                line = dict(

                    width = 2,

                ),

            ),

            name='League',

            text = newPlayerData2['player']

        )



newPlayerData3 = newPlayerData2[newPlayerData2.player.isin(ESPNRankTop30)]

trace2 = Scatter(

            x=newPlayerData3['fg_distance'],

            y=newPlayerData3['fg_percentage'],

            mode = 'markers',

            marker = dict(

                size = newPlayerData3['made']/20,

                color = 'rgba(255, 123, 132, .9)',

                line = dict(

                    width = 2,

                ),

            ),

            name='#NBARank Top 30',

            text = newPlayerData3['player']

        )



data = [trace1, trace2]



layout = Layout(

    hovermode = 'closest',

    annotations=Annotations([

        Annotation(

            x=0.5004254919715793,

            y=-0.16191064079952971,

            showarrow=False,

            text='Average Shot Distance (Feet)',

            xref='paper',

            yref='paper'

        ),

        Annotation(

            x=-0.06944728761514841,

            y=0.4714285714285711,

            showarrow=False,

            text='Field Goal %',

            textangle=-90,

            xref='paper',

            yref='paper'

        )

    ]),

    autosize=True,

    margin=Margin(

        b=100

    ),

    title='Comparing Players\' FG% and Average Shot Distance (Minimum 200 Made Shots)'

)



graph = Figure(data=data, layout=layout)

iplot(graph)
data = pd.read_csv('../input/shot_logs.csv')

attackerIDList = list(data['player_id'].unique())

playerIDList = []

for ID in attackerIDList:

    for period in range(1,4):

        playerIDList.append(ID+period/10)



playerData = pd.DataFrame(index=playerIDList, columns=['player', 'period', 'made', 'missed', 'fg_percentage'])



for attackerID in attackerIDList:

    name= data[(data['player_id'] == attackerID)]['player_name'].iloc[0]

    spacePos = name.find(' ')

    firstname = name[0].upper() + name[1:spacePos]

    lastname = name[spacePos+1].upper() + name[spacePos+2:]

    name = firstname + ' ' + lastname

    for period in range(1,5):

        made = np.sum(np.logical_and(data[(data['player_id'] == attackerID)]['SHOT_RESULT']=='made',

                                     data[(data['player_id'] == attackerID)]['PERIOD']==period))

        missed = np.sum(np.logical_and(data[(data['player_id'] == attackerID)]['SHOT_RESULT']=='missed',

                                       data[(data['player_id'] == attackerID)]['PERIOD']==period))

        percentage = made/(made+missed)

        playerData.at[attackerID+period/10, 'player'] = name

        playerData.at[attackerID+period/10, 'period'] = period

        playerData.at[attackerID+period/10, 'made'] = made

        playerData.at[attackerID+period/10, 'missed'] = missed

        playerData.at[attackerID+period/10, 'fg_percentage'] = percentage

    

newPlayerData = playerData.sort_values('player', ascending=True)

inelligibleNames = newPlayerData[newPlayerData.made < 50]['player']

inelligibleNames = inelligibleNames.unique()

newPlayerData2 = newPlayerData[~newPlayerData.player.isin(inelligibleNames)]

newPlayerData2
from ipywidgets import widgets 

from IPython.display import display, clear_output, Image

from plotly.graph_objs import *

from plotly.widgets import GraphWidget



ESPNRankTop30 = ['Lebron James', 'Chris Paul', 'Anthony Davis', 'Russell Westbrook', 'Blake Griffin', 'Stephen Curry',

                'Kevin Love', 'Kevin Durant', 'James Harden', 'Dwight Howard', 'Carmelo Anthony', 'Joakim Noah',

                'Lamarcus Aldridge', 'Marc Gasol', 'Tony Parker', 'Damian Lillard', 'Dirk Nowtizski', 'John Wall',

                'Demarcus Cousins', 'Chris Bosh', 'Tim Duncan', 'Al Jefferson', 'Kyrie Irving', 'Kawhi Leonard', 

                'Serge Ibaka', 'Al Horford', 'Goran Dragic', 'Derrick Rose', 'Kyle Lowry', 'Andre Drummond']



trace1 = Scatter(

            x=newPlayerData2['period'],

            y=newPlayerData2['fg_percentage'],

            mode = 'markers',

            marker = dict(

                size = newPlayerData2['made']/5,

                color = 'rgba(132, 123, 255, .9)',

                line = dict(

                    width = 2,

                ),

            ),

            name='League',

            text = newPlayerData2['player']

        )



newPlayerData3 = newPlayerData2[newPlayerData2.player.isin(ESPNRankTop30)]

trace2 = Scatter(

            x=newPlayerData3['period'],

            y=newPlayerData3['fg_percentage'],

            mode = 'markers',

            marker = dict(

                size = newPlayerData3['made']/5,

                color = 'rgba(255, 123, 132, .9)',

                line = dict(

                    width = 2,

                ),

            ),

            name='#NBARank Top 30',

            text = newPlayerData3['player']

        )



data = [trace1,trace2]



layout = Layout(

    height = 700,

    hovermode = 'closest',

    annotations=Annotations([

        Annotation(

            x=0.5004254919715793,

            y=-0.16191064079952971,

            showarrow=False,

            text='Quarter',

            xref='paper',

            yref='paper'

        ),

        Annotation(

            x=-0.06944728761514841,

            y=0.4714285714285711,

            showarrow=False,

            text='Field Goal %',

            textangle=-90,

            xref='paper',

            yref='paper'

        )

    ]),

    autosize=True,

    margin=Margin(

        b=100

    ),

    title='Comparing Players\' FG% per Quarter (Minimum 50 Shots per Quarter)'

)



graph = Figure(data=data, layout=layout)

iplot(graph)
ESPNRankTop30 = ['Lebron James', 'Chris Paul', 'Anthony Davis', 'Russell Westbrook', 'Blake Griffin', 'Stephen Curry',

                'Kevin Love', 'Kevin Durant', 'James Harden', 'Dwight Howard', 'Carmelo Anthony', 'Joakim Noah',

                'Lamarcus Aldridge', 'Marc Gasol', 'Tony Parker', 'Damian Lillard', 'Dirk Nowtizski', 'John Wall',

                'Demarcus Cousins', 'Chris Bosh', 'Tim Duncan', 'Al Jefferson', 'Kyrie Irving', 'Kawhi Leonard', 

                'Serge Ibaka', 'Al Horford', 'Goran Dragic', 'Derrick Rose', 'Kyle Lowry', 'Andre Drummond']

newPlayerData3 = newPlayerData[newPlayerData.player.isin(ESPNRankTop30)]



newPlayerData3 = newPlayerData3.sort_values('player')

newPlayerData3['ranking'] = 0

for i in range(len(ESPNRankTop30)):

    newPlayerData3.loc[newPlayerData3['player'] == ESPNRankTop30[i], 'ranking'] = i+1

    

newPlayerData3 = newPlayerData3.sort_values('period', ascending = True)

newPlayerData3
from plotly.graph_objs import *

from ipywidgets import widgets 

from IPython.display import display, clear_output, Image

from plotly.graph_objs import *

from plotly.widgets import GraphWidget



data = []

newPlayerData3 = newPlayerData3.sort_values(['ranking','period'], ascending = [True, True])

for player in list(newPlayerData3['player'].unique()):

    data.append(Scatter(

        x=newPlayerData3[newPlayerData3['player']==player]['period'],

        y=newPlayerData3[newPlayerData3['player']==player]['fg_percentage'],

        mode = 'lines+markers',

        line = dict(

            color = 'rgba(100, 100, 100, .7)',

            width = 1

#             dash = 'dot'

        ),

        marker = dict(

            size = newPlayerData3['made']/5,

            color = 'rgba(132, 123, 255, .9)',

            line = dict(

                color = 'rgb(250,250,250)',

                width = 2

            ),

        ),

        name = player + ' #' + str(newPlayerData3[newPlayerData3['player']==player]['ranking'].iloc[0]),

        text = (player + ' #' + newPlayerData3[newPlayerData3['player']==player]['ranking'].astype(str) + '<br>' +

                'Made: ' + newPlayerData3[newPlayerData3['player']==player]['made'].astype(str) + '<br>' +

                'Missed: ' + newPlayerData3[newPlayerData3['player']==player]['missed'].astype(str))

    ))



layout = Layout(

    hovermode = 'closest',

    annotations=Annotations([

        Annotation(

            x=0.5004254919715793,

            y=-0.16191064079952971,

            showarrow=False,

            text='Quarter',

            xref='paper',

            yref='paper'

        ),

        Annotation(

            x=-0.06944728761514841,

            y=0.4714285714285711,

            showarrow=False,

            text='Field Goal %',

            textangle=-90,

            xref='paper',

            yref='paper'

        )

    ]),

    autosize=True,

    margin=Margin(

        b=100

    ),

    title='Comparing Players\' FG% per Quarter (ESPNRank Top 30)'

)



graph = Figure(data=data, layout=layout)

iplot(graph)
data = pd.read_csv('../input/shot_logs.csv')

ESPNRankTop30 = ['lebron james', 'chris paul', 'anthony davis', 'russell westbrook', 'blake griffin', 'stephen curry',

                'kevin love', 'kevin durant', 'james harden', 'dwight howard', 'carmelo anthony', 'joakim noah',

                'lamarcus aldridge', 'marc gasol', 'tony parker', 'damian lillard', 'dirk nowtizski', 'john wall',

                'demarcus cousins', 'chris bosh', 'tim duncan', 'al jefferson', 'kyrie irving', 'kawhi leonard', 

                'serge ibaka', 'al horford', 'goran dragic', 'derrick rose', 'kyle lowry', 'andre drummond']

data = data[data.player_name.isin(ESPNRankTop30)]

defenderIDList = list(data['CLOSEST_DEFENDER_PLAYER_ID'].unique())



playerData = pd.DataFrame(index=defenderIDList, columns=['player', 'made_against', 'missed_against', 'fg_percentage_against'])



for defenderID in defenderIDList:

    name= data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['CLOSEST_DEFENDER'].iloc[0]

    made = np.sum(data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['SHOT_RESULT']=='made')

    missed = np.sum(data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['SHOT_RESULT']=='missed')

    percentage = made/(made+missed)

    playerData.at[defenderID, 'player'] = name

    playerData.at[defenderID, 'made_against'] = made

    playerData.at[defenderID, 'missed_against'] = missed

    playerData.at[defenderID, 'fg_percentage_against'] = percentage



playerData2 = playerData.drop(playerData[np.logical_and(playerData.missed_against < 20, playerData.made_against < 20)].index)
playerData2 = playerData2.sort_values('fg_percentage_against', ascending = True)

playerData2.head(10)
playerData2 = playerData2.sort_values('fg_percentage_against', ascending = False)

playerData2.head(10)
data = pd.read_csv('../input/shot_logs.csv')

stars = ['lebron james', 'chris paul', 'anthony davis', 'russell westbrook', 'blake griffin', 'stephen curry',

                'kevin love', 'kevin durant', 'james harden', 'dwight howard', 'carmelo anthony', 'joakim noah',

                'lamarcus aldridge', 'marc gasol', 'tony parker', 'damian lillard', 'dirk nowtizski', 'john wall',

                'demarcus cousins', 'chris bosh', 'tim duncan', 'al jefferson', 'kyrie irving', 'kawhi leonard', 

                'serge ibaka', 'al horford', 'goran dragic', 'derrick rose', 'kyle lowry', 'andre drummond']



kryptoniteData = pd.DataFrame(index=stars, columns=['kryptonite', 'made_against', 'missed_against', 

                                                            'fg_percentage_against'])



for star in stars:

    data = pd.read_csv('../input/shot_logs.csv')

    data = data[data.player_name.isin([star])]

    defenderIDList = list(data['CLOSEST_DEFENDER_PLAYER_ID'].unique())



    playerData = pd.DataFrame(index=defenderIDList, columns=['player', 'made_against', 'missed_against', 'fg_percentage_against'])

    try:

        for defenderID in defenderIDList:

            name= data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['CLOSEST_DEFENDER'].iloc[0]

            made = np.sum(data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['SHOT_RESULT']=='made')

            missed = np.sum(data[(data['CLOSEST_DEFENDER_PLAYER_ID'] == defenderID)]['SHOT_RESULT']=='missed')

            percentage = made/(made+missed)

            playerData.at[defenderID, 'player'] = name

            playerData.at[defenderID, 'made_against'] = made

            playerData.at[defenderID, 'missed_against'] = missed

            playerData.at[defenderID, 'fg_percentage_against'] = percentage



        playerData2 = playerData.drop(playerData[(playerData.missed_against + playerData.made_against) < 8].index)

        playerData2 = playerData2.sort_values('fg_percentage_against', ascending = True)

        kryptoniteData.at[star, 'kryptonite']  = playerData2['player'].iloc[0]

        kryptoniteData.at[star, 'made_against']  = playerData2['made_against'].iloc[0]

        kryptoniteData.at[star, 'missed_against']  = playerData2['missed_against'].iloc[0]

        kryptoniteData.at[star, 'fg_percentage_against']  = playerData2['fg_percentage_against'].iloc[0]

    except:

        kryptoniteData.at[star, 'kryptonite']  = 'N/A'

        kryptoniteData.at[star, 'made_against']  = 'N/A'

        kryptoniteData.at[star, 'missed_against']  = 'N/A'

        kryptoniteData.at[star, 'fg_percentage_against']  = 'N/A'

        

kryptoniteData
data = pd.read_csv('../input/shot_logs.csv')

stars = ['lebron james', 'chris paul', 'anthony davis', 'russell westbrook', 'blake griffin', 'stephen curry',

                'kevin love', 'kevin durant', 'james harden', 'dwight howard', 'carmelo anthony', 'joakim noah',

                'lamarcus aldridge', 'marc gasol', 'tony parker', 'damian lillard', 'dirk nowtizski', 'john wall',

                'demarcus cousins', 'chris bosh', 'tim duncan', 'al jefferson', 'kyrie irving', 'kawhi leonard', 

                'serge ibaka', 'al horford', 'goran dragic', 'derrick rose', 'kyle lowry', 'andre drummond']



playerData = pd.DataFrame(index=list(range(len(stars)+1)), columns=['player', 'made_1ft', 'missed_1ft', 'fg_percentage_1ft',

                                                        'made_2ft', 'missed_2ft', 'fg_percentage_2ft',

                                                        'made_3ft', 'missed_3ft', 'fg_percentage_3ft',

                                                        'made_4ft', 'missed_4ft', 'fg_percentage_4ft',

                                                        'made_5ft', 'missed_5ft', 'fg_percentage_5ft',

                                                        'made_over_5ft', 'missed_over_5ft', 'fg_percentage_over_5ft'])



for defenderDist in range(1,6):

    made = np.sum(data[np.logical_and(data['CLOSE_DEF_DIST'] <= defenderDist, 

                                      data['CLOSE_DEF_DIST'] > (defenderDist-1))]['SHOT_RESULT']=='made')

    missed = np.sum(data[np.logical_and(data['CLOSE_DEF_DIST'] < defenderDist, 

                                            data['CLOSE_DEF_DIST'] > (defenderDist-1))]['SHOT_RESULT']=='missed')

    percentage = made/(made+missed)

    playerData.loc[len(stars), 'player'] = 'League Average'

    playerData.loc[len(stars), ('made_' + str(defenderDist) + 'ft')] = made

    playerData.loc[len(stars), ('missed_' + str(defenderDist) + 'ft')] = missed

    playerData.loc[len(stars), ('fg_percentage_' + str(defenderDist) + 'ft')] = percentage

    

made = np.sum(data[data['CLOSE_DEF_DIST'] > 5]['SHOT_RESULT']=='made')

missed = np.sum(data[data['CLOSE_DEF_DIST'] > 5]['SHOT_RESULT']=='missed')

percentage = made/(made+missed)

playerData.loc[len(stars), 'made_over_5ft'] = made

playerData.loc[len(stars), 'missed_over_5ft'] = missed

playerData.loc[len(stars), 'fg_percentage_over_5ft'] = percentage    

    

for star in stars:

    stardata = data[data.player_name == star]

    playerData.loc[stars.index(star), 'player'] = star

    for defenderDist in range(1,6):

        try:

            made = np.sum(stardata[np.logical_and(stardata['CLOSE_DEF_DIST'] <= defenderDist, 

                                                  stardata['CLOSE_DEF_DIST'] > (defenderDist-1))]['SHOT_RESULT']=='made')

            missed = np.sum(stardata[np.logical_and(stardata['CLOSE_DEF_DIST'] < defenderDist, 

                                                  stardata['CLOSE_DEF_DIST'] > (defenderDist-1))]['SHOT_RESULT']=='missed')

            percentage = made/(made+missed)

            playerData.loc[stars.index(star), ('made_' + str(defenderDist) + 'ft')] = made

            playerData.loc[stars.index(star), ('missed_' + str(defenderDist) + 'ft')] = missed

            playerData.loc[stars.index(star), ('fg_percentage_' + str(defenderDist) + 'ft')] = percentage

        except:

            playerData.loc[stars.index(star), ('made_' + str(defenderDist) + 'ft')] = 'N/A'

            playerData.loc[stars.index(star), ('missed_' + str(defenderDist) + 'ft')] = 'N/A'

            playerData.loc[stars.index(star), ('fg_percentage_' + str(defenderDist) + 'ft')] = 'N/A'



    try:

        made = np.sum(stardata[stardata['CLOSE_DEF_DIST'] > 5]['SHOT_RESULT']=='made')

        missed = np.sum(stardata[stardata['CLOSE_DEF_DIST'] > 5]['SHOT_RESULT']=='missed')

        percentage = made/(made+missed)

        playerData.loc[stars.index(star), 'made_over_5ft'] = made

        playerData.loc[stars.index(star), 'missed_over_5ft'] = missed

        playerData.loc[stars.index(star), 'fg_percentage_over_5ft'] = percentage    

    except:

        playerData.loc[stars.index(star), 'made_over_5ft'] = 'N/A'

        playerData.loc[stars.index(star), 'missed_over_5ft'] = 'N/A'

        playerData.loc[stars.index(star), 'fg_percentage_over_5ft'] = 'N/A'



playerData
trace1 = go.Bar(

    x=list(playerData['player']),

    y=list(playerData['fg_percentage_1ft']),

    name='Defender 0-1ft away',

    text = 'Made: ' + playerData['made_1ft'].astype(str) + '<br>' + 'Missed: ' + playerData['missed_1ft'].astype(str)

)

trace2 = go.Bar(

    x=list(playerData['player']),

    y=list(playerData['fg_percentage_2ft']),

    name='Defender 1-2ft away',

    text = 'Made: ' + playerData['made_2ft'].astype(str) + '<br>' + 'Missed: ' + playerData['missed_2ft'].astype(str)

)

trace3 = go.Bar(

    x=list(playerData['player']),

    y=list(playerData['fg_percentage_3ft']),

    name='Defender 2-3ft away',

    text = 'Made: ' + playerData['made_3ft'].astype(str) + '<br>' + 'Missed: ' + playerData['missed_3ft'].astype(str)

)

trace4 = go.Bar(

    x=list(playerData['player']),

    y=list(playerData['fg_percentage_4ft']),

    name='Defender 3-4ft away',

    text = 'Made: ' + playerData['made_4ft'].astype(str) + '<br>' + 'Missed: ' + playerData['missed_4ft'].astype(str)

)

trace5 = go.Bar(

    x=list(playerData['player']),

    y=list(playerData['fg_percentage_5ft']),

    name='Defender 4-5ft away',

    text = 'Made: ' + playerData['made_5ft'].astype(str) + '<br>' + 'Missed: ' + playerData['missed_5ft'].astype(str)

)

trace6 = go.Bar(

    x=list(playerData['player']),

    y=list(playerData['fg_percentage_over_5ft']),

    name='Defender >5ft away',

    text = 'Made: ' + playerData['made_over_5ft'].astype(str) + '<br>' + 'Missed: ' + playerData['missed_over_5ft'].astype(str)

)



data = [trace1,trace2,trace3,trace4,trace5,trace6]

layout = go.Layout(

    barmode='group',

    title = "FG% Depending on Distance from Defender",

    annotations=Annotations([

        Annotation(

            x=-0.05944728761514841,

            y=0.4714285714285711,

            showarrow=False,

            text='Field Goal %',

            textangle=-90,

            xref='paper',

            yref='paper'

        )

    ]),

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)