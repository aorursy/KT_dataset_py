import plotly.express as px
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

pd.options.mode.chained_assignment = None 

# !pip install plotly --upgrade
## Loading Football Data dataset
df = pd.read_csv('../input/football-data/Football_Data.csv', delimiter=',', index_col = 0)
## Loading tabla with every transfer
transfers = pd.read_csv('../input/football-data/Transfer_data.csv', index_col = 0)
## Loading data for every game
stat_per_game = pd.read_csv('../input/football-data/stat_per_game.csv', index_col=0)
df.sample(5)
df.describe()
# So we create a scatter plot with Transfers expends on yaxis and transfers income in xaxis..

# Try zooming into the graph (clicking and dragging on the graph)

fig = px.scatter(df, y='Transfer_E', x='Transfer_I',size='wins',hover_name='Team',
                 hover_data=['League', 'Year'], color='position')

fig.update_layout(
    title="Transfers Income and Expend",
    xaxis_title="Tran. Income",
    yaxis_title="Tran. Expend",
    font=dict(
        family="Courier New, monospace",
        size=16,
        color="#7f7f7f"
    )
)

fig.show()
Monaco_transf = transfers[(transfers.Team_from == 'Monaco') & (transfers.Season == 2018)]
Monaco_transf
ManCity_transf = transfers[(transfers.Team_to == 'Manchester City') & (transfers.Season == 2017)]
ManCity_transf
import math
# Copy df
diff_xpts = df.copy()
customdata= list(diff_xpts['Year'])

# Include subplots titles, we want a figure with two rows and one column
fig = make_subplots(rows=2, cols=1, subplot_titles=('Relation between real points and expected points',
                                                    'Relation between position and expected points diff'))

fig.append_trace(
    go.Scatter( y= df['xpts'], x=df['pts'], mode='markers',text=df['Team'],marker_line_width=1,
               marker=dict(size=df['wins'],
                color=df['position'], showscale=True
                ),customdata=customdata,
               hovertemplate=  "<b>%{text}</b><br><br>" +
        "Real points: %{x}<br>" +
        "Expected Points: %{y:.1f}<br>"+
              'Year: %{customdata}'), row=1, col=1
)

fig.append_trace(
    go.Scatter(x= diff_xpts['pts'], y=diff_xpts['xpts_diff'], mode='markers',text=df['Team'],marker_line_width=1.5,
              marker=dict(color=diff_xpts['position'],
                         size= 10, opacity=0.8),
              customdata=customdata,
               hovertemplate=  "<b>%{text}</b><br><br>" +
        "Points: %{x}<br>" +
        "Exp. points diff: %{y:.1f}<br>"+
              'Year: %{customdata}')
            , row=2, col=1)


# Add diagonal line for both plots

fig.add_shape(
            type="line",
            x0=10,
            y0=10,
            x1=100,
            y1=100,
            line=dict(
                color="Red",
                width=4,
                dash="dot",
            ),
row=1,col=1)

fig.add_shape(
            type="line",
            x0=10,
            y0=0,
            x1=120,
            y1=0,
            line=dict(
                color="Red",
                width=4,
                dash="dot",
            ),
row=2,col=1)

# Update xaxis properties
fig.update_xaxes(title_text="Real Points", row=1, col=1)
fig.update_xaxes(title_text="Real Points", row=2, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="Expected Points", row=1, col=1)
fig.update_yaxes(title_text="Expected points diff", row=2, col=1)

# Select the size 
fig.update_layout(height=800, width=800,
                  title_text="Position and expected points",
                  showlegend= False,
                 hoverlabel=dict(
                 bgcolor="white", 
                 font_size=13, 
                 font_family="Rockwell"),
                 )


fig.show()
# Filter by first and second position
first_place = df[(df.position == 1) | (df.position == 2)]


fig = px.scatter(first_place, y='xpts', x='pts',size='wins',hover_name='Team',
                 hover_data=['League', 'Year'], color='League')

fig.add_shape(
        # Line Diagonal
            type="line",
            x0=55,
            y0=55,
            x1=100,
            y1=100,
            line=dict(
                color="Red",
                width=4,
                dash="dot",
            )
)



fig.show()
## White color means the team didn't play that season.

## Filter the dataframe by La Liga
la_liga = df[df.League == 'La_liga']

dfr = pd.DataFrame(la_liga.pivot('Team', 'Year', 'position'))
dfr['Mean'] = (dfr[2014]+dfr[2015]+dfr[2016]+dfr[2017]+dfr[2018])/5
dfr = dfr.sort_values('Mean')
dfr = dfr.drop(['Mean'],axis=1)

plt.figure(figsize = (16,8))
ax = sns.heatmap(dfr, cmap='coolwarm', annot=True, linewidths=0.2, linecolor='white')
# Filter only for 2018/2019 season
la_liga_stats = la_liga[la_liga.Year == 2018]

# Append a new column that shows total shots on target conceded
la_liga_stats['RS_OnTarget'] = la_liga_stats['H_RS_OnTarget'] + la_liga_stats['A_RS_OnTarget']

#Set to bar plots so they 
SOT = go.Bar(x = la_liga_stats['Team'], y = la_liga_stats['S_OnTarget'], name = 'Shots on Target')
RSOT = go.Bar(x = la_liga_stats['Team'], y = la_liga_stats['RS_OnTarget'], name = 'Rivals Shots on Target')

data = [SOT, RSOT]

layout = go.Layout(
    barmode='group',
    title="Comparing shots on target for La Liga Teams in 2018/2019",
    xaxis={'title': 'Teams'},
    yaxis={'title': "Shots on target",
    }
)


fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

fig = make_subplots(rows=1, cols=2, subplot_titles=('Relation between deep passes',
                                                    'Relation between Passes allowed per defensive action'))


fig.append_trace(
    go.Scatter( y= la_liga_stats['deep_allowed'], x=la_liga_stats['deep'], mode='markers',text=la_liga_stats['Team'],marker_line_width=1,
               marker=dict(size=10),
                   hovertemplate=  "<b>%{text}</b><br><br>" +
                                   "Deep: %{x}<br>" +
                                   "Deep allowed: %{y:.1f}<br>"), row=1, col=1
)

fig.append_trace(
    go.Scatter(y= la_liga_stats['oppda_coef'], x=la_liga_stats['ppda_coef'], mode='markers',text=la_liga_stats['Team'],marker_line_width=1.5,
              marker=dict(size= 10),
              hovertemplate=  "<b>%{text}</b><br><br>" +
                              "ppda_coef: %{x}<br>" +
                              "oppda_coef: %{y:.1f}<br>"), row=1, col=2)

# Update xaxis properties
fig.update_xaxes(title_text="Deep passes", row=1, col=1)
fig.update_xaxes(title_text="ppda_coef", row=1, col=2)

# Update yaxis properties
fig.update_yaxes(title_text="Deep passes allowed", row=1, col=1)
fig.update_yaxes(title_text="Oponent ppda_coef", row=1, col=2)


fig.update_layout(height=500, width=1000,
                  title_text="Deep passes and ball pressure coef",
                  showlegend= False)

fig.show()
# Create a new variable that indicates how many expected goals have a team for every shot on target.

la_liga_stats['Shot_Eff'] = la_liga_stats['S_OnTarget'] / la_liga_stats['xG']

fig = make_subplots(rows=1, cols=2, subplot_titles=('Expected Goals',
                                                    'Shot Effectiveness'))


fig.append_trace(
    go.Bar(x = la_liga_stats.sort_values('xG',ascending=False)['Team'],
           y = la_liga_stats.sort_values('xG',ascending=False)['xG']), row=1, col=1)

fig.append_trace(
   go.Bar(x = la_liga_stats.sort_values('Shot_Eff')['Team'],
          y = la_liga_stats.sort_values('Shot_Eff')['Shot_Eff']), row=1, col=2)



# Update xaxis properties
fig.update_xaxes(title_text="Team", row=1, col=1)
fig.update_xaxes(title_text="Team", row=1, col=2)

# Update yaxis properties
fig.update_yaxes(title_text="Goals Expected", row=1, col=1)
fig.update_yaxes(title_text="Shot Eff", row=1, col=2)


fig.update_layout(height=500, width=1000,
                  
                  showlegend= False)

fig.show()
# Correct the final positions becouse both had the same points..

la_liga_stats.loc[la_liga_stats.Team == 'Sevilla', 'position'] = 6
la_liga_stats.loc[la_liga_stats.Team == 'Getafe', 'position'] = 5

fig = go.Figure(data=[go.Bar(x = la_liga_stats.sort_values('position')['Team'], 
                y = la_liga_stats.sort_values('position')['pts'],
                text=la_liga_stats.sort_values('position')['Team'],
                            )])

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', size=14))

# Change colorbars characteristics
fig.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=1, opacity=0.6)

fig.update_layout(

    title="La Liga final positions 2018/2019",
    xaxis={'title': 'Teams'},
    yaxis={'title': "Points",
    }
)


fig.show()
Sev = stat_per_game[stat_per_game.team == 'Sevilla']
Sev = Sev[Sev.year == 2018]
Sev['Matchday'] = range(1,39)
Sev['color'] = Sev['Matchday'].apply(lambda x: 1 if x == 26 else 0)
fig = px.bar(Sev, y='xpts_diff', x='Matchday',text='Matchday',hover_name='Matchday',
             hover_data=['pts', 'xpts', 'xpts_diff'], color='color',color_continuous_scale=[(0, "green"), (1, "red")])

fig.add_shape(
        # Line Diagonal
            type="line",
            x0=0,
            y0=0,
            x1=40,
            y1=0,
            line=dict(
                color="Red",
                width=2,))

fig.update_xaxes(ticksuffix="º")
            
fig.update_layout(coloraxis_showscale=False,
                xaxis_title='Matchday',
                yaxis_title="Expected points diff",
                font=dict(
                family="'Rockwell'",
                size=13,
                color="#7f7f7f"))

fig.show()
# Copy the df to a new one
newpoints= la_liga_stats.copy()
# Change Sevilla's points to 62
newpoints.loc[newpoints.Team == 'Sevilla','pts'] = 62

# Set colors so we can change only Sevilla's bar color
colors = ['blue',] * newpoints.shape[0]
colors[3] = 'red'

fig = go.Figure(data=[go.Bar(x = newpoints.sort_values('pts',ascending=False)['Team'], y = newpoints.sort_values('pts',ascending=False)['pts'],
                                 text=newpoints.sort_values('pts',ascending=False)['Team'],
                                 hovertext=newpoints.sort_values('pts',ascending=False)['pts'],
                                 textposition='auto',
                         )
                     ]
               )

# Change colorbars characteristics
fig.update_traces(marker_color=colors, marker_line_color='rgb(8,48,107)',
                  marker_line_width=1, opacity=0.6)

fig.update_xaxes(tickangle=50, tickfont=dict(family='Rockwell', size=14))



fig.update_xaxes(title_text="Team")
fig.update_yaxes(title_text="Points")


fig = go.Figure(fig, layout=layout)
py.iplot(fig)
print( "Winning Huesca's match would get Sevilla a money compensation difference of:  "+
        str(round(12.76 - 8.12 + 15, 2)) + " Millions")
# So first we look to how would had Sevilla finishes all Seasons

# Create a df for each different plot
newpoints1 = la_liga[la_liga.Year == 2014].sort_values('xpts', ascending=False)
newpoints2 = la_liga[la_liga.Year == 2015].sort_values('xpts', ascending=False)
newpoints3 = la_liga[la_liga.Year == 2016].sort_values('xpts', ascending=False)
newpoints4 = la_liga[la_liga.Year == 2017].sort_values('xpts', ascending=False)
newpoints5 = la_liga[la_liga.Year == 2018].sort_values('xpts', ascending=False)
# Same for each barplot colors
color1 = ['blue'] * newpoints.shape[0]
color1[list(newpoints1.Team.unique()).index('Sevilla')] = 'red'
color2 = ['blue'] * newpoints.shape[0]
color2[list(newpoints2.Team.unique()).index('Sevilla')] = 'red'
color3 = ['blue'] * newpoints.shape[0]
color3[list(newpoints3.Team.unique()).index('Sevilla')] = 'red'
color4 = ['blue'] * newpoints.shape[0]
color4[list(newpoints4.Team.unique()).index('Sevilla')] = 'red'
color5 = ['blue'] * newpoints.shape[0]
color5[list(newpoints5.Team.unique()).index('Sevilla')] = 'red'

# Initialize figure
fig = go.Figure()

# Add Traces

fig.add_trace(
    go.Bar(x = newpoints1['Team'], y = newpoints1['xpts'],
                                 text=newpoints1['Team'],
                                 hovertext=newpoints1['xpts'],
                                 textposition='auto',
                                 visible=False,
                                 # Change bar colors
                                 marker_color=color1, marker_line_color='rgb(8,48,107)',
                                 marker_line_width=1, opacity=0.6
                         ))

fig.add_trace(
    go.Bar(x = newpoints2['Team'], y = newpoints2['xpts'],
                                 text=newpoints2['Team'],
                                 hovertext=newpoints2['xpts'],
                                 textposition='auto',
                                 visible=False,
                                 marker_color=color2, marker_line_color='rgb(8,48,107)',
                                 marker_line_width=1, opacity=0.6
                         ))

fig.add_trace(
    go.Bar(x = newpoints3['Team'], y = newpoints3['xpts'],
                                 text=newpoints3['Team'],
                                 hovertext=newpoints3['xpts'],
                                 textposition='auto',
                                 visible=False,
                                 marker_color=color3, marker_line_color='rgb(8,48,107)',
                                 marker_line_width=1, opacity=0.6
           
                         ))

fig.add_trace(
    go.Bar(x = newpoints4['Team'], y = newpoints4['xpts'],
                                 text=newpoints4['Team'],
                                 hovertext=newpoints4['xpts'],
                                 textposition='auto',
                                 visible=False,
                                 marker_color=color4, marker_line_color='rgb(8,48,107)',
                                 marker_line_width=1, opacity=0.6
                         ))

fig.add_trace(
    go.Bar(x = newpoints5['Team'], y = newpoints5['xpts'],
                                 text=newpoints5['Team'],
                                 hovertext=newpoints5['xpts'],
                                 textposition='auto',
                                # We make this trace Visible=True so it's the one that appear automatically
           marker_color=color5, marker_line_color='rgb(8,48,107)',
                  marker_line_width=1, opacity=0.6
                         ))


# Take care of buttons
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            active=4,
            x=0.9,
            y=1.2,
            buttons=list([
                dict(label="2014",
                     method="update",
                     args=[{"visible": [True, False, False, False, False]},
                           {"title": "Season 2014/2015"}]),
                dict(label="2015",
                     method="update",
                     args=[{"visible": [False, True, False, False, False]},
                           {"title": "Season 2015/2016"}]),
                dict(label="2016",
                     method="update",
                     args=[{"visible": [False, False, True, False, False]},
                           {"title": "Season 2016/2017"}]),
                dict(label="2017",
                     method="update",
                     args=[{"visible": [False, False, False, True, False]},
                           {"title": "Season 2017/2018"}]),
                 dict(label="2018",
                     method="update",
                     args=[{"visible": [False, False, False, False, True]},
                           {"title": "Season 2018/2019"}]),
                
            ]),
        )
    ])

print("Sevilla final positions:")
for year in range(2014,2019):
    ran = la_liga[la_liga.Year == year]
    ran['ranking'] = ran['xpts'].rank(ascending=False)
    print("Season {}/{}: ".format(year,year+1) + ' ' +  str(int(ran[ran.Team == 'Sevilla']['ranking'].values[0])) + 'º')
    
# Check the website indicated above to look for the table with all the money obtained by each team

print("Sevilla tv compensation by final positions:\n")
print('-'*70)
print("\n"
"Season 2014/2015 ----> 5.46 Millions for tv\n"
"Season 2015/2016 ----> 5.46 Millions + 12 Millions for Champions \n"
"Season 2016/2017 ----> 5.46 Millions\n"
"Season 2017/2018 ----> 7.29 Millions + 13 Millions for Champions \n"
"Season 2018/2019 ----> 15.08 Millions + 15 Millions for Champions \n")
print('-'*70)     
print("Total = {} Millions".format(str(5.46*3+7.29+15.08+ 12+ 13+15)))
print('Real earnings for this seasons: 15.8 Millions for tv + 6 Millions for Europa League \n'
      'Expected earnings for this seasons: 78.15 Millions \n'
      'Difference: %.1f Millions'%(78.15-15.8-6))