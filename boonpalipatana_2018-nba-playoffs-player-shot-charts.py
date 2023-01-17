import pandas as pd
import numpy as np
import matplotlib
all_shots_df = pd.read_csv('../input/playoff_shots.csv', sep=',')
all_shots_df
shots_df = all_shots_df[(all_shots_df.PLAYER_NAME == 'Stephen Curry') | # Golden State Warriors
                        (all_shots_df.PLAYER_NAME == 'Kevin Durant') |
                        (all_shots_df.PLAYER_NAME == 'Klay Thompson') |
                        (all_shots_df.PLAYER_NAME == 'Draymond Green') |
                        (all_shots_df.PLAYER_NAME == 'LeBron James') | # Cleveland Cavaliers
                        (all_shots_df.PLAYER_NAME == 'Kevin Love') |
                        (all_shots_df.PLAYER_NAME == 'Jeff Green') |
                        (all_shots_df.PLAYER_NAME == 'George Hill')]
shots_df
shots_df.columns
shots_df.PLAYER_NAME.unique()
court_shapes = []

#Outer Lines
outer_lines_shape = dict(
    type='rect',
    xref='x',
    yref='y',
    x0='-250',
    y0='-47.5',
    x1='250',
    y1='422.5',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)
 
court_shapes.append(outer_lines_shape)

#Hoop Shape
hoop_shape = dict(
    type='circle',
    xref='x',
    yref='y',
    x0='7.5',
    y0='7.5',
    x1='-7.5',
    y1='-7.5',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)
 
court_shapes.append(hoop_shape)

#Basket Backboard
backboard_shape = dict(
    type='rect',
    xref='x',
    yref='y',
    x0='-30',
    y0='-7.5',
    x1='30',
    y1='-6.5',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    ),
    fillcolor='rgba(10, 10, 10, 1)'
)
 
court_shapes.append(backboard_shape)

#Outer Box of Three-Second Area
outer_three_sec_shape = dict(
    type='rect',
    xref='x',
    yref='y',
    x0='-80',
    y0='-47.5',
    x1='80',
    y1='143.5',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)
 
court_shapes.append(outer_three_sec_shape)

#Inner Box of Three-Second Area
inner_three_sec_shape = dict(
    type='rect',
    xref='x',
    yref='y',
    x0='-60',
    y0='-47.5',
    x1='60',
    y1='143.5',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)
 
court_shapes.append(inner_three_sec_shape)

#Three Point Line (Left)
left_line_shape = dict(
    type='line',
    xref='x',
    yref='y',
    x0='-220',
    y0='-47.5',
    x1='-220',
    y1='92.5',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)
 
court_shapes.append(left_line_shape)

#Three Point Line (Right)
right_line_shape = dict(
    type='line',
    xref='x',
    yref='y',
    x0='220',
    y0='-47.5',
    x1='220',
    y1='92.5',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)
 
court_shapes.append(right_line_shape)

#Three Point Line Arc
three_point_arc_shape = dict(
    type='path',
    xref='x',
    yref='y',
    path='M -220 92.5 C -70 300, 70 300, 220 92.5',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)
 
court_shapes.append(three_point_arc_shape)

#Center Circle
center_circle_shape = dict(
    type='circle',
    xref='x',
    yref='y',
    x0='60',
    y0='482.5',
    x1='-60',
    y1='362.5',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)
 
court_shapes.append(center_circle_shape)

#Restraining Circle
res_circle_shape = dict(
    type='circle',
    xref='x',
    yref='y',
    x0='20',
    y0='442.5',
    x1='-20',
    y1='402.5',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)
 
court_shapes.append(res_circle_shape)

#Free Throw Circle
free_throw_circle_shape = dict(
    type='circle',
    xref='x',
    yref='y',
    x0='60',
    y0='200',
    x1='-60',
    y1='80',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)
 
court_shapes.append(free_throw_circle_shape)

#Restricted Area
res_area_shape = dict(
    type='circle',
    xref='x',
    yref='y',
    x0='40',
    y0='40',
    x1='-40',
    y1='-40',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1,
        dash='dot'
    )
)
 
court_shapes.append(res_area_shape)
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)

def updateVisibility(selectedPlayer):
    visibilityValues = []
    for player in list(shots_df.PLAYER_NAME.unique()):
        if player == selectedPlayer:
            visibilityValues.append(True)
            visibilityValues.append(True)
        else:
            visibilityValues.append(False)
            visibilityValues.append(False)
    return visibilityValues

data = []
buttons_data = []
for player in list(shots_df.PLAYER_NAME.unique()):
    shot_trace_made = go.Scatter(
        x = shots_df[(shots_df['EVENT_TYPE'] == 'Made Shot') & (shots_df['PLAYER_NAME'] == player)]['LOC_X'],
        y = shots_df[(shots_df['EVENT_TYPE'] == 'Made Shot') & (shots_df['PLAYER_NAME'] == player)]['LOC_Y'],
        mode = 'markers',
        marker = dict(
            size = 6,
            color = 'rgba(63, 191, 63, 0.9)',
        ), 
        name = 'Made',
        text = shots_df[(shots_df['EVENT_TYPE'] == 'Made Shot') & (shots_df['PLAYER_NAME'] == player)]['SHOT_ZONE_BASIC'],
        textposition = 'auto',
        textfont = dict(
            color = 'rgba(75, 85, 102,0.7)'
        ),
        visible = (player =='LeBron James')
    )

    shot_trace_missed = go.Scatter(
        x = shots_df[(shots_df['EVENT_TYPE'] == 'Missed Shot') & (shots_df['PLAYER_NAME'] == player)]['LOC_X'],
        y = shots_df[(shots_df['EVENT_TYPE'] == 'Missed Shot') & (shots_df['PLAYER_NAME'] == player)]['LOC_Y'],
        mode = 'markers',
        marker = dict(
            size = 6,
            color = 'rgba(241, 18, 18, 0.9)',
        ), 
        name = 'Missed',
        text = shots_df[(shots_df['EVENT_TYPE'] == 'Missed Shot') & (shots_df['PLAYER_NAME'] == player)]['SHOT_ZONE_BASIC'],
        textposition = 'auto',
        textfont = dict(
            color = 'rgba(75, 85, 102,0.7)'
        ),
        visible = (player =='LeBron James')
    )

    data.append(shot_trace_made)
    data.append(shot_trace_missed)
    
    buttons_data.append(
        dict(
            label = player,
            method = 'update',
            args = [{'visible': updateVisibility(player)}]
        )
    )
    

updatemenus = list([
    dict(active=0,
         buttons = buttons_data,
         direction = 'down',
         pad = {'r': 10, 't': 10},
         showactive = True,
         x = 0.21,
         xanchor = 'left',
         y = 1.19,
         yanchor = 'top',
         font = dict (
             size = 14
         )
    )
])

layout = go.Layout(
    title='________________ Shot Chart',
    titlefont=dict(
        size=14
    ),
    hovermode = 'closest',
    updatemenus = updatemenus,
    showlegend = True,
    height = 600,
    width = 600, 
    shapes = court_shapes,
    xaxis = dict(
        showticklabels = False
    ),
    yaxis = dict(
        showticklabels = False
    )
)
 
fig = go.Figure(data=data, layout=layout)
iplot(fig)
all_shots_df.SHOT_ZONE_AREA.unique()
all_shots_df.SHOT_ZONE_BASIC.unique()
zone_shapes = []

#Three Point Line Arc Top Left
zone1 = dict(
    type='path',
    xref='x',
    yref='y',
    path='M -250,92 L -220,92 C -187,140 -161,174 -80,228  L -95,350 L -250,350 Z',
    fillcolor= 'rgba(93, 140, 255, 0.5)',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)
 
zone_shapes.append(zone1)

#Three Point Line Arc Top Right
zone2 = dict(
    type='path',
    xref='x',
    yref='y',
    path='M 250,92 L 220,92 C 187,140 161,174 80,228  L 95,350 L 250,350 Z',
    fillcolor= 'rgba(93, 255, 18, 0.5)',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)
 
zone_shapes.append(zone2)

#Three Point Line Arc Top Center
zone3 = dict(
    type='path',
    xref='x',
    yref='y',
    path='M -80,228 C -50,240 0,266 80,228 L 95,350 L -95,350 Z',
    fillcolor= 'rgba(253, 153, 18, 0.5)',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)

zone_shapes.append(zone3)

#Three Point Line Left Corner
zone4 = dict(
    type='path',
    xref='x',
    yref='y',
    path='M -250,-47.5 L -220,-47.5 L -220,92 L -250,92 Z',
    fillcolor= 'rgba(253, 153, 18, 0.5)',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)

zone_shapes.append(zone4)

#Three Point Line Right Corner
zone5 = dict(
    type='path',
    xref='x',
    yref='y',
    path='M 250,-47.5 L 220,-47.5 L 220,92 L 250,92 Z',
    fillcolor= 'rgba(253, 153, 18, 0.5)',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)

zone_shapes.append(zone5)

#Under the basket
zone6 = dict(
    type='path',
    xref='x',
    yref='y',
    path='M -40,-47.5 L -40,0 C -40,53 40,53 40,0 L 40,-47.5 Z',
    fillcolor= 'rgba(253, 153, 18, 0.5)',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)

zone_shapes.append(zone6)

#Paint
zone7 = dict(    
    type='path',
    xref='x',
    yref='y',
    path='M -60,-47.5 L -40,-47.5 L -40,0 C -40,53 40,53 40,0 L 40,-47.5 L 60,-47.5 L 60,143.5 L -60,143.5 Z',
    fillcolor= 'rgba(253, 20, 18, 0.5)',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)

zone_shapes.append(zone7)

#Midrange Center
zone8 = dict(    
    type='path',
    xref='x',
    yref='y',
    path='M -60,143.5 L -80,228 C -50,240 0,266 80,228 L 60,143.5 Z',
    fillcolor= 'rgba(54, 239, 233, 0.5)',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)

zone_shapes.append(zone8)

#Midrange Center Left
zone8 = dict(    
    type='path',
    xref='x',
    yref='y',
    path='M -220,92 C -187,140 -161,174 -80,228 L -60,143.5 L -60,92 Z',
    fillcolor= 'rgba(54, 123, 43, 0.5)',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)

zone_shapes.append(zone8)

#Midrange Center Right
zone9 = dict(    
    type='path',
    xref='x',
    yref='y',
    path='M 220,92 C 187,140 161,174 80,228 L 60,143.5 L 60,92 Z',
    fillcolor= 'rgba(54, 123, 43, 0.5)',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)

zone_shapes.append(zone9)

#Midrange Left
zone10 = dict(    
    type='path',
    xref='x',
    yref='y',
    path='M -220,-47.5 L -220,92 L -60,92 L -60,-47.5 Z',
    fillcolor= 'rgba(134, 23, 43, 0.5)',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)

zone_shapes.append(zone10)

#Midrange Right
zone11 = dict(    
    type='path',
    xref='x',
    yref='y',
    path='M 220,-47.5 L 220,92 L 60,92 L 60,-47.5 Z',
    fillcolor= 'rgba(134, 23, 43, 0.5)',
    line=dict(
        color='rgba(10, 10, 10, 1)',
        width=1
    )
)

zone_shapes.append(zone11)


shot_zones_text = go.Scatter(
    x=[-175, 0, 175, -120, 0, 120, -210, -135, 0, 0, 135, 210],
    y=[250, 300, 250, 140, 200, 140, -25, 40, 90, -10, 40, -25],
    mode='text',
    name='Zones',
    text=['<b>Zone '+str(n)+'</b>' for n in range(1,13)],
    hovertext = ['left_center_3', 'center_3', 'right_center_3', 'left_center_mid',
                 'center_mid', 'right_center_mid', 'left_corner_3', 'left_mid', 
                 'paint', 'restricted_area', 'right_mid', 'right_corner_3'],
    textposition='middle',        
    textfont = dict(
        color = 'rgba(10, 10, 10, 1)',
        size = 14
    )
)


layout = go.Layout(
    title='Shot Chart Zones',
    titlefont=dict(
        size=20
    ),
    hovermode = 'closest',
    showlegend = True,
    height = 600,
    width = 600, 
    shapes = court_shapes+zone_shapes,
    xaxis = dict(
        showticklabels = False,
        range = [-250, 250]
    ),
    yaxis = dict(
        showticklabels = False,
        range = [-47.5, 452.5]
    )
)
 
fig = go.Figure(data=[shot_zones_text], layout=layout)
iplot(fig)
league_averages = {}

#Center Three Pointer
made = sum(all_shots_df[(all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])
attempted = len(all_shots_df[(all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                             & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])
league_averages['center_3'] = made/attempted

#Left Center Three Pointer
made = sum(all_shots_df[(all_shots_df.SHOT_ZONE_AREA == 'Left Side Center(LC)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])
attempted = len(all_shots_df[(all_shots_df.SHOT_ZONE_AREA == 'Left Side Center(LC)') 
                             & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])
league_averages['left_center_3'] = made/attempted

#Right Center Three Pointer
made = sum(all_shots_df[(all_shots_df.SHOT_ZONE_AREA == 'Right Side Center(RC)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])
attempted = len(all_shots_df[(all_shots_df.SHOT_ZONE_AREA == 'Right Side Center(RC)') 
                             & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])
league_averages['right_center_3'] = made/attempted

#Left Corner Three Pointer
made = sum(all_shots_df[(all_shots_df.SHOT_ZONE_BASIC == 'Left Corner 3')]['SHOT_MADE_FLAG'])
attempted = len(all_shots_df[(all_shots_df.SHOT_ZONE_BASIC == 'Left Corner 3')]['SHOT_MADE_FLAG'])
league_averages['left_corner_3'] = made/attempted

#Right Corner Three Pointer
made = sum(all_shots_df[(all_shots_df.SHOT_ZONE_BASIC == 'Right Corner 3')]['SHOT_MADE_FLAG'])
attempted = len(all_shots_df[(all_shots_df.SHOT_ZONE_BASIC == 'Right Corner 3')]['SHOT_MADE_FLAG'])
league_averages['right_corner_3'] = made/attempted

#Center Mid Range
made = sum(all_shots_df[(all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')]['SHOT_MADE_FLAG'])
attempted = len(all_shots_df[(all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                             & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')]['SHOT_MADE_FLAG'])
league_averages['center_mid'] = made/attempted

#Left Center Mid Range
made = sum(all_shots_df[((all_shots_df.SHOT_ZONE_AREA == 'Left Side Center(LC)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                        ((all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
attempted = len(all_shots_df[((all_shots_df.SHOT_ZONE_AREA == 'Left Side Center(LC)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                        ((all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
league_averages['left_center_mid'] = made/attempted

#Right Center Mid Range
made = sum(all_shots_df[((all_shots_df.SHOT_ZONE_AREA == 'Right Side Center(RC)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                        ((all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
attempted = len(all_shots_df[((all_shots_df.SHOT_ZONE_AREA == 'Right Side Center(RC)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                        ((all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
league_averages['right_center_mid'] = made/attempted

#Left Mid Range
made = sum(all_shots_df[((all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                        ((all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
attempted = len(all_shots_df[((all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                        ((all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
league_averages['left_mid'] = made/attempted

#Right Mid Range
made = sum(all_shots_df[((all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                        ((all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
attempted = len(all_shots_df[((all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                        ((all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
league_averages['right_mid'] = made/attempted


#Paint (Not Restricted Area)
made = sum(all_shots_df[(all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)')]['SHOT_MADE_FLAG'])
attempted = len(all_shots_df[(all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                        & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)')]['SHOT_MADE_FLAG'])
league_averages['paint'] = made/attempted

#Restricted Area
made = sum(all_shots_df[(all_shots_df.SHOT_ZONE_BASIC == 'Restricted Area')]['SHOT_MADE_FLAG'])
attempted = len(all_shots_df[(all_shots_df.SHOT_ZONE_BASIC == 'Restricted Area')]['SHOT_MADE_FLAG'])
league_averages['restricted_area'] = made/attempted

print('League Average FG% by Zone')
for zone, avg_made in league_averages.items():
    print('{}: {}%'.format(zone, round(avg_made*100, 2)))
def getZoneAverages(player):
    player_zone_averages = {}

    #Center Three Pointer
    made = sum(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                            & (all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                            & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])
    attempted = len(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                 & (all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                                 & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])
    if (attempted == 0): player_zone_averages['center_3'] = None
    else: player_zone_averages['center_3'] = made/attempted

    #Left Center Three Pointer
    made = sum(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                            & (all_shots_df.SHOT_ZONE_AREA == 'Left Side Center(LC)') 
                            & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])
    attempted = len(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                 & (all_shots_df.SHOT_ZONE_AREA == 'Left Side Center(LC)') 
                                 & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])
    if (attempted == 0): player_zone_averages['left_center_3'] = None
    else: player_zone_averages['left_center_3'] = made/attempted

    #Right Center Three Pointer
    made = sum(all_shots_df[(all_shots_df.PLAYER_NAME == player)
                            & (all_shots_df.SHOT_ZONE_AREA == 'Right Side Center(RC)') 
                            & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])
    attempted = len(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                 & (all_shots_df.SHOT_ZONE_AREA == 'Right Side Center(RC)') 
                                 & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])
    if (attempted == 0): player_zone_averages['right_center_3'] = None
    else: player_zone_averages['right_center_3'] = made/attempted

    #Left Corner Three Pointer
    made = sum(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                            & (all_shots_df.SHOT_ZONE_BASIC == 'Left Corner 3')]['SHOT_MADE_FLAG'])
    attempted = len(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                 & (all_shots_df.SHOT_ZONE_BASIC == 'Left Corner 3')]['SHOT_MADE_FLAG'])
    if (attempted == 0): player_zone_averages['left_corner_3'] = None
    else: player_zone_averages['left_corner_3'] = made/attempted

    #Right Corner Three Pointer
    made = sum(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                            & (all_shots_df.SHOT_ZONE_BASIC == 'Right Corner 3')]['SHOT_MADE_FLAG'])
    attempted = len(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                 & (all_shots_df.SHOT_ZONE_BASIC == 'Right Corner 3')]['SHOT_MADE_FLAG'])
    if (attempted == 0): player_zone_averages['right_corner_3'] = None
    else: player_zone_averages['right_corner_3'] = made/attempted

    #Center Mid Range
    made = sum(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                            & (all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                            & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')]['SHOT_MADE_FLAG'])
    attempted = len(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                 & (all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                                 & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')]['SHOT_MADE_FLAG'])
    if (attempted == 0): player_zone_averages['center_mid'] = None
    else: player_zone_averages['center_mid'] = made/attempted

    #Left Center Mid Range
    made = sum(all_shots_df[((all_shots_df.PLAYER_NAME == player) 
                             & (all_shots_df.SHOT_ZONE_AREA == 'Left Side Center(LC)') 
                             & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                            ((all_shots_df.PLAYER_NAME == player) 
                             & (all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                             & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
    attempted = len(all_shots_df[((all_shots_df.PLAYER_NAME == player) 
                                  & (all_shots_df.SHOT_ZONE_AREA == 'Left Side Center(LC)') 
                                  & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                                 ((all_shots_df.PLAYER_NAME == player) 
                                  & (all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                                  & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
    if (attempted == 0): player_zone_averages['left_center_mid'] = None
    else: player_zone_averages['left_center_mid'] = made/attempted

    #Right Center Mid Range
    made = sum(all_shots_df[((all_shots_df.PLAYER_NAME == player) 
                             & (all_shots_df.SHOT_ZONE_AREA == 'Right Side Center(RC)') 
                             & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                            ((all_shots_df.PLAYER_NAME == player) 
                             & (all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                             & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
    attempted = len(all_shots_df[((all_shots_df.PLAYER_NAME == player) 
                                  & (all_shots_df.SHOT_ZONE_AREA == 'Right Side Center(RC)') 
                                  & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                                 ((all_shots_df.PLAYER_NAME == player) 
                                  & (all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                                  & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
    if (attempted == 0): player_zone_averages['right_center_mid'] = None
    else: player_zone_averages['right_center_mid'] = made/attempted

    #Left Mid Range
    made = sum(all_shots_df[((all_shots_df.PLAYER_NAME == player) 
                             & (all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                             & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                            ((all_shots_df.PLAYER_NAME == player) 
                             & (all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                             & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
    attempted = len(all_shots_df[((all_shots_df.PLAYER_NAME == player) 
                                  & (all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                                  & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                                 ((all_shots_df.PLAYER_NAME == player) 
                                  & (all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                                  & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
    if (attempted == 0): player_zone_averages['left_mid'] = None
    else: player_zone_averages['left_mid'] = made/attempted

    #Right Mid Range
    made = sum(all_shots_df[((all_shots_df.PLAYER_NAME == player) 
                             & (all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                             & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                            ((all_shots_df.PLAYER_NAME == player) 
                             & (all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                             & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
    attempted = len(all_shots_df[((all_shots_df.PLAYER_NAME == player) 
                                  & (all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                                  & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                                 ((all_shots_df.PLAYER_NAME == player) & 
                                  (all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                                  & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
    if (attempted == 0): player_zone_averages['right_mid'] = None
    else: player_zone_averages['right_mid'] = made/attempted


    #Paint (Not Restricted Area)
    made = sum(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                            & (all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                            & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)')]['SHOT_MADE_FLAG'])
    attempted = len(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                 & (all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                                 & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)')]['SHOT_MADE_FLAG'])
    if (attempted == 0): player_zone_averages['paint'] = None
    else: player_zone_averages['paint'] = made/attempted

    #Restricted Area
    made = sum(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                            & (all_shots_df.SHOT_ZONE_BASIC == 'Restricted Area')]['SHOT_MADE_FLAG'])
    attempted = len(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                 & (all_shots_df.SHOT_ZONE_BASIC == 'Restricted Area')]['SHOT_MADE_FLAG'])
    if (attempted == 0): player_zone_averages['restricted_area'] = None
    else: player_zone_averages['restricted_area'] = made/attempted
    
    return player_zone_averages
def getZoneColor(zone_name, pct):
    if (pct is None):
        return (184,184,184,0.5)
    else:
        cmap = matplotlib.cm.get_cmap('seismic')
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:,-1] = np.concatenate(((np.linspace(0.8, 0.2, int(cmap.N/2)), np.linspace(0.2, 0.8, int(cmap.N/2)))))
        cmap = matplotlib.colors.ListedColormap(my_cmap)
        rgba = cmap(0.5 + ((pct - league_averages[zone_name])/league_averages[zone_name]/2))            
        return rgba
def getZoneText(player, zone_name):
    attempted = 0
    #Center Three Pointer
    if (zone_name == "center_3"):
        made = sum(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                & (all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                                & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])
        attempted = len(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                     & (all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                                     & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])
        
    #Left Center Three Pointer
    elif (zone_name == "left_center_3"):
        made = sum(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                & (all_shots_df.SHOT_ZONE_AREA == 'Left Side Center(LC)') 
                                & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])
        attempted = len(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                     & (all_shots_df.SHOT_ZONE_AREA == 'Left Side Center(LC)') 
                                     & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])

    #Right Center Three Pointer
    elif (zone_name == "right_center_3"):
        made = sum(all_shots_df[(all_shots_df.PLAYER_NAME == player)
                                & (all_shots_df.SHOT_ZONE_AREA == 'Right Side Center(RC)') 
                                & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])
        attempted = len(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                     & (all_shots_df.SHOT_ZONE_AREA == 'Right Side Center(RC)') 
                                     & (all_shots_df.SHOT_ZONE_BASIC == 'Above the Break 3')]['SHOT_MADE_FLAG'])

    #Left Corner Three Pointer
    elif (zone_name == "left_corner_3"):
        made = sum(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                & (all_shots_df.SHOT_ZONE_BASIC == 'Left Corner 3')]['SHOT_MADE_FLAG'])
        attempted = len(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                     & (all_shots_df.SHOT_ZONE_BASIC == 'Left Corner 3')]['SHOT_MADE_FLAG'])


    #Right Corner Three Pointer
    elif (zone_name == "right_corner_3"):    
        made = sum(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                & (all_shots_df.SHOT_ZONE_BASIC == 'Right Corner 3')]['SHOT_MADE_FLAG'])
        attempted = len(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                     & (all_shots_df.SHOT_ZONE_BASIC == 'Right Corner 3')]['SHOT_MADE_FLAG'])

    #Center Mid Range
    elif (zone_name == "center_mid"):  
        made = sum(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                & (all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                                & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')]['SHOT_MADE_FLAG'])
        attempted = len(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                     & (all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                                     & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')]['SHOT_MADE_FLAG'])

    #Left Center Mid Range
    elif (zone_name == "left_center_mid"):  
        made = sum(all_shots_df[((all_shots_df.PLAYER_NAME == player) 
                                 & (all_shots_df.SHOT_ZONE_AREA == 'Left Side Center(LC)') 
                                 & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                                ((all_shots_df.PLAYER_NAME == player) 
                                 & (all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                                 & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
        attempted = len(all_shots_df[((all_shots_df.PLAYER_NAME == player) 
                                      & (all_shots_df.SHOT_ZONE_AREA == 'Left Side Center(LC)') 
                                      & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                                     ((all_shots_df.PLAYER_NAME == player) 
                                      & (all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                                      & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])

    #Right Center Mid Range
    elif (zone_name == "right_center_mid"):  
        made = sum(all_shots_df[((all_shots_df.PLAYER_NAME == player) 
                                 & (all_shots_df.SHOT_ZONE_AREA == 'Right Side Center(RC)') 
                                 & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                                ((all_shots_df.PLAYER_NAME == player) 
                                 & (all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                                 & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
        attempted = len(all_shots_df[((all_shots_df.PLAYER_NAME == player) 
                                      & (all_shots_df.SHOT_ZONE_AREA == 'Right Side Center(RC)') 
                                      & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                                     ((all_shots_df.PLAYER_NAME == player) 
                                      & (all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                                      & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])

    #Left Mid Range
    elif (zone_name == "left_mid"):  
        made = sum(all_shots_df[((all_shots_df.PLAYER_NAME == player) 
                                 & (all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                                 & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                                ((all_shots_df.PLAYER_NAME == player) 
                                 & (all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                                 & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
        attempted = len(all_shots_df[((all_shots_df.PLAYER_NAME == player) 
                                      & (all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                                      & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                                     ((all_shots_df.PLAYER_NAME == player) 
                                      & (all_shots_df.SHOT_ZONE_AREA == 'Left Side(L)') 
                                      & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])

    #Right Mid Range
    elif (zone_name == "right_mid"):  
        made = sum(all_shots_df[((all_shots_df.PLAYER_NAME == player) 
                                 & (all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                                 & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                                ((all_shots_df.PLAYER_NAME == player) 
                                 & (all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                                 & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])
        attempted = len(all_shots_df[((all_shots_df.PLAYER_NAME == player) 
                                      & (all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                                      & (all_shots_df.SHOT_ZONE_BASIC == 'Mid-Range')) |
                                     ((all_shots_df.PLAYER_NAME == player) & 
                                      (all_shots_df.SHOT_ZONE_AREA == 'Right Side(R)') 
                                      & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)'))]['SHOT_MADE_FLAG'])

    #Paint (Not Restricted Area)
    elif (zone_name == "paint"):  
        made = sum(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                & (all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                                & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)')]['SHOT_MADE_FLAG'])
        attempted = len(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                     & (all_shots_df.SHOT_ZONE_AREA == 'Center(C)') 
                                     & (all_shots_df.SHOT_ZONE_BASIC == 'In The Paint (Non-RA)')]['SHOT_MADE_FLAG'])

    #Restricted Area
    elif (zone_name == "restricted_area"):  
        made = sum(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                & (all_shots_df.SHOT_ZONE_BASIC == 'Restricted Area')]['SHOT_MADE_FLAG'])
        attempted = len(all_shots_df[(all_shots_df.PLAYER_NAME == player) 
                                     & (all_shots_df.SHOT_ZONE_BASIC == 'Restricted Area')]['SHOT_MADE_FLAG'])
    
    if (attempted == 0): return ''
    else: return "<b>{}-{}<br>{}</b>".format(made, attempted, round(made/attempted, 3))  
def getColoredZones(player_zone_averages):
    zone_shapes = []

    #Three Point Line Arc Top Left
    zone1 = dict(
        type='path',
        xref='x',
        yref='y',
        path='M -250,92 L -220,92 C -187,140 -161,174 -80,228  L -95,350 L -250,350 Z',
        fillcolor= "rgba{}".format(str(getZoneColor('left_center_3', player_zone_averages['left_center_3']))),
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )

    zone_shapes.append(zone1)

    #Three Point Line Arc Top Right
    zone2 = dict(
        type='path',
        xref='x',
        yref='y',
        path='M 250,92 L 220,92 C 187,140 161,174 80,228  L 95,350 L 250,350 Z',
        fillcolor= 'rgba{}'.format(str(getZoneColor('right_center_3', player_zone_averages['right_center_3']))),
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )

    zone_shapes.append(zone2)

    #Three Point Line Arc Top Center
    zone3 = dict(
        type='path',
        xref='x',
        yref='y',
        path='M -80,228 C -50,240 0,266 80,228 L 95,350 L -95,350 Z',
        fillcolor= 'rgba{}'.format(str(getZoneColor('center_3', player_zone_averages['center_3']))),
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )

    zone_shapes.append(zone3)

    #Three Point Line Left Corner
    zone4 = dict(
        type='path',
        xref='x',
        yref='y',
        path='M -250,-47.5 L -220,-47.5 L -220,92 L -250,92 Z',
        fillcolor= 'rgba{}'.format(str(getZoneColor('left_corner_3', player_zone_averages['left_corner_3']))),
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )

    zone_shapes.append(zone4)

    #Three Point Line Right Corner
    zone5 = dict(
        type='path',
        xref='x',
        yref='y',
        path='M 250,-47.5 L 220,-47.5 L 220,92 L 250,92 Z',
        fillcolor= 'rgba{}'.format(str(getZoneColor('right_corner_3', player_zone_averages['right_corner_3']))),
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )

    zone_shapes.append(zone5)

    #Under the basket
    zone6 = dict(
        type='path',
        xref='x',
        yref='y',
        path='M -40,-47.5 L -40,0 C -40,53 40,53 40,0 L 40,-47.5 Z',
        fillcolor= 'rgba{}'.format(str(getZoneColor('restricted_area', player_zone_averages['restricted_area']))),
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )

    zone_shapes.append(zone6)

    #Paint
    zone7 = dict(    
        type='path',
        xref='x',
        yref='y',
        path='M -60,-47.5 L -40,-47.5 L -40,0 C -40,53 40,53 40,0 L 40,-47.5 L 60,-47.5 L 60,143.5 L -60,143.5 Z',
        fillcolor= 'rgba{}'.format(str(getZoneColor('paint', player_zone_averages['paint']))),
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )

    zone_shapes.append(zone7)

    #Midrange Center
    zone8 = dict(    
        type='path',
        xref='x',
        yref='y',
        path='M -60,143.5 L -80,228 C -50,240 0,266 80,228 L 60,143.5 Z',
        fillcolor= 'rgba{}'.format(str(getZoneColor('center_mid', player_zone_averages['center_mid']))),
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )

    zone_shapes.append(zone8)

    #Midrange Center Left
    zone8 = dict(    
        type='path',
        xref='x',
        yref='y',
        path='M -220,92 C -187,140 -161,174 -80,228 L -60,143.5 L -60,92 Z',
        fillcolor= 'rgba{}'.format(str(getZoneColor('left_center_mid', player_zone_averages['left_center_mid']))),
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )

    zone_shapes.append(zone8)

    #Midrange Center Right
    zone9 = dict(    
        type='path',
        xref='x',
        yref='y',
        path='M 220,92 C 187,140 161,174 80,228 L 60,143.5 L 60,92 Z',
        fillcolor= 'rgba{}'.format(str(getZoneColor('right_center_mid', player_zone_averages['right_center_mid']))),
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )

    zone_shapes.append(zone9)

    #Midrange Left
    zone10 = dict(    
        type='path',
        xref='x',
        yref='y',
        path='M -220,-47.5 L -220,92 L -60,92 L -60,-47.5 Z',
        fillcolor= 'rgba{}'.format(str(getZoneColor('left_mid', player_zone_averages['left_mid']))),
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )

    zone_shapes.append(zone10)

    #Midrange Right
    zone11 = dict(    
        type='path',
        xref='x',
        yref='y',
        path='M 220,-47.5 L 220,92 L 60,92 L 60,-47.5 Z',
        fillcolor= 'rgba{}'.format(str(getZoneColor('right_mid', player_zone_averages['right_mid']))),
        line=dict(
            color='rgba(10, 10, 10, 1)',
            width=1
        )
    )

    zone_shapes.append(zone11)
    return zone_shapes
def updateVisibility(selectedPlayer):
    visibilityValues = []
    for player in list(shots_df.PLAYER_NAME.unique()):
        if player == selectedPlayer:
            visibilityValues.append(True) #shot_trace_made
            visibilityValues.append(True) #shot_trace_missed
            visibilityValues.append(True) #shot_zone_text
        else:
            visibilityValues.append(False)
            visibilityValues.append(False)
            visibilityValues.append(False)
    return visibilityValues

zone_shapes = getColoredZones(getZoneAverages('LeBron James'))
data = []
buttons_data = []
for player in list(shots_df.PLAYER_NAME.unique()):
    shot_trace_made = go.Scatter(
        x = shots_df[(shots_df['EVENT_TYPE'] == 'Made Shot') & (shots_df['PLAYER_NAME'] == player)]['LOC_X'],
        y = shots_df[(shots_df['EVENT_TYPE'] == 'Made Shot') & (shots_df['PLAYER_NAME'] == player)]['LOC_Y'],
        mode = 'markers',
        marker = dict(
            size = 6,
            color = 'rgba(63, 191, 63, 0.9)',
        ), 
        name = 'Made',
        text = shots_df[(shots_df['EVENT_TYPE'] == 'Made Shot') & (shots_df['PLAYER_NAME'] == player)]['SHOT_ZONE_AREA'],
        textposition = 'auto',
        textfont = dict(
            color = 'rgba(75, 85, 102,0.7)'
        ),
        visible = (player =='LeBron James')
    )

    shot_trace_missed = go.Scatter(
        x = shots_df[(shots_df['EVENT_TYPE'] == 'Missed Shot') & (shots_df['PLAYER_NAME'] == player)]['LOC_X'],
        y = shots_df[(shots_df['EVENT_TYPE'] == 'Missed Shot') & (shots_df['PLAYER_NAME'] == player)]['LOC_Y'],
        mode = 'markers',
        marker = dict(
            size = 6,
            color = 'rgba(241, 18, 18, 0.9)',
        ), 
        name = 'Missed',
        text = shots_df[(shots_df['EVENT_TYPE'] == 'Missed Shot') & (shots_df['PLAYER_NAME'] == player)]['SHOT_ZONE_AREA'],
        textposition = 'auto',
        textfont = dict(
            color = 'rgba(75, 85, 102,0.7)'
        ),
        visible = (player =='LeBron James')
    )

    shot_zones_text = go.Scatter(
        x=[-175, 0, 175, -120, 0, 120, -190, -135, 0, 0, 135, 190],
        y=[250, 300, 250, 140, 200, 140, -25, 40, 90, -10, 40, -25],
        mode='text',
        name='FG %',
        text=[getZoneText(player, 'left_center_3'), 
              getZoneText(player, 'center_3'), 
              getZoneText(player, 'right_center_3'), 
              getZoneText(player, 'left_center_mid'), 
              getZoneText(player, 'center_mid'), 
              getZoneText(player, 'right_center_mid'), 
              getZoneText(player, 'left_corner_3'), 
              getZoneText(player, 'left_mid'), 
              getZoneText(player, 'paint'), 
              getZoneText(player, 'restricted_area'), 
              getZoneText(player, 'right_mid'), 
              getZoneText(player, 'right_corner_3')],
        textposition='middle',        
        textfont = dict(
            color = 'rgba(10, 10, 10, 1)',
            size = 14
        ),
        visible = (player == 'LeBron James')
    )
    
    data.append(shot_trace_made)
    data.append(shot_trace_missed)
    data.append(shot_zones_text)
    
    buttons_data.append(
        dict(
            label = player,
            method = 'update',
            args = [{'visible': updateVisibility(player)}, 
                    {'shapes': court_shapes + getColoredZones(getZoneAverages(player))}]
        )
    )
    

updatemenus = list([
    dict(active=0,
         buttons = buttons_data,
         direction = 'down',
         pad = {'r': 10, 't': 10},
         showactive = True,
         x = 0.21,
         xanchor = 'left',
         y = 1.19,
         yanchor = 'top',
         font = dict (
             size = 14
         )
    )
])

layout = go.Layout(
    title='________________ Shot Chart',
    titlefont=dict(
        size=14
    ),
    hovermode = 'closest',
    updatemenus = updatemenus,
    showlegend = True,
    height = 600,
    width = 600, 
    shapes = court_shapes+zone_shapes,
    xaxis = dict(
        showticklabels = False,
        range = [-250, 250]
    ),
    yaxis = dict(
        showticklabels = False,
        range = [-47.5, 452.5]
    )
)
 
fig = go.Figure(data=data, layout=layout)
iplot(fig)
from collections import Counter
shot_counts = Counter(all_shots_df.PLAYER_NAME)
shot_counts.most_common(20)
players20 = [i[0] for i in shot_counts.most_common(20)]
def updateVisibility(selectedPlayer):
    visibilityValues = []
    for player in list(players20):
        if player == selectedPlayer:
            visibilityValues.append(True) #shot_trace_made
            visibilityValues.append(True) #shot_trace_missed
            visibilityValues.append(True) #shot_zone_text
        else:
            visibilityValues.append(False)
            visibilityValues.append(False)
            visibilityValues.append(False)
    return visibilityValues

zone_shapes = getColoredZones(getZoneAverages('LeBron James'))
data = []
buttons_data = []
for player in list(players20):
    shot_trace_made = go.Scatter(
        x = all_shots_df[(all_shots_df['EVENT_TYPE'] == 'Made Shot') & (all_shots_df['PLAYER_NAME'] == player)]['LOC_X'],
        y = all_shots_df[(all_shots_df['EVENT_TYPE'] == 'Made Shot') & (all_shots_df['PLAYER_NAME'] == player)]['LOC_Y'],
        mode = 'markers',
        marker = dict(
            size = 6,
            color = 'rgba(63, 191, 63, 0.9)',
        ), 
        name = 'Made',
        text = all_shots_df[(all_shots_df['EVENT_TYPE'] == 'Made Shot') & (all_shots_df['PLAYER_NAME'] == player)]['SHOT_ZONE_AREA'],
        textposition = 'auto',
        textfont = dict(
            color = 'rgba(75, 85, 102,0.7)'
        ),
        visible = (player =='LeBron James')
    )

    shot_trace_missed = go.Scatter(
        x = all_shots_df[(all_shots_df['EVENT_TYPE'] == 'Missed Shot') & (all_shots_df['PLAYER_NAME'] == player)]['LOC_X'],
        y = all_shots_df[(all_shots_df['EVENT_TYPE'] == 'Missed Shot') & (all_shots_df['PLAYER_NAME'] == player)]['LOC_Y'],
        mode = 'markers',
        marker = dict(
            size = 6,
            color = 'rgba(241, 18, 18, 0.9)',
        ), 
        name = 'Missed',
        text = all_shots_df[(all_shots_df['EVENT_TYPE'] == 'Missed Shot') & (all_shots_df['PLAYER_NAME'] == player)]['SHOT_ZONE_AREA'],
        textposition = 'auto',
        textfont = dict(
            color = 'rgba(75, 85, 102,0.7)'
        ),
        visible = (player =='LeBron James')
    )

    shot_zones_text = go.Scatter(
        x=[-175, 0, 175, -120, 0, 120, -190, -135, 0, 0, 135, 190],
        y=[250, 300, 250, 140, 200, 140, -25, 40, 90, -10, 40, -25],
        mode='text',
        name='FG %',
        text=[getZoneText(player, 'left_center_3'), 
              getZoneText(player, 'center_3'), 
              getZoneText(player, 'right_center_3'), 
              getZoneText(player, 'left_center_mid'), 
              getZoneText(player, 'center_mid'), 
              getZoneText(player, 'right_center_mid'), 
              getZoneText(player, 'left_corner_3'), 
              getZoneText(player, 'left_mid'), 
              getZoneText(player, 'paint'), 
              getZoneText(player, 'restricted_area'), 
              getZoneText(player, 'right_mid'), 
              getZoneText(player, 'right_corner_3')],
        textposition='middle',        
        textfont = dict(
            color = 'rgba(10, 10, 10, 1)',
            size = 14
        ),
        visible = (player == 'LeBron James')
    )
    
    data.append(shot_trace_made)
    data.append(shot_trace_missed)
    data.append(shot_zones_text)
    
    buttons_data.append(
        dict(
            label = player,
            method = 'update',
            args = [{'visible': updateVisibility(player)}, 
                    {'shapes': court_shapes + getColoredZones(getZoneAverages(player))}]
        )
    )
    
updatemenus = list([
    dict(active=0,
         buttons = buttons_data,
         direction = 'down',
         pad = {'r': 10, 't': 10},
         showactive = True,
         x = 0.21,
         xanchor = 'left',
         y = 1.19,
         yanchor = 'top',
         font = dict (
             size = 14
         )
    )
])

layout = go.Layout(
    title='_________________   Shot Chart',
    titlefont=dict(
        size=14
    ),
    hovermode = 'closest',
    updatemenus = updatemenus,
    showlegend = True,
    height = 600,
    width = 600, 
    shapes = court_shapes+zone_shapes,
    xaxis = dict(
        showticklabels = False,
        range = [-250, 250]
    ),
    yaxis = dict(
        showticklabels = False,
        range = [-47.5, 452.5]
    )
)
 
fig = go.Figure(data=data, layout=layout)
iplot(fig)