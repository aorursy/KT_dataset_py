import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#plt.style.use('ggplot')



%matplotlib inline



import plotly

import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

import plotly.figure_factory as ff

from plotly import subplots

from plotly.subplots import make_subplots

init_notebook_mode(connected=True)



pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def get_data(path):

    df = pd.read_csv(path)

    col_names = ['Year','Player','Pos','HT','WT','Team','SelectionType','NBADraft','Nationality']

    df.columns = col_names



    return df



def change_nationality(df):

    

    df['Nationality'] = np.where(df['Nationality'] != 'United States','Rest of World',df['Nationality'])

    

    return df



def change_team(df):

    df['Team'] = np.where(df['Team'] == 'New Jersey Nets','Brooklyn Nets',df['Team'])

    df['Team'] = np.where(df['Team'] == 'New Orleans Hornets','New Orleans Pelicans',df['Team'])

    df['Team'] = np.where(df['Team'] == 'Charlotte Bobcats','Charlotte Hornets',df['Team'])

    df['Team'] = np.where(df['Team'] == 'Seattle SuperSonics','Oklahoma City Thunder',df['Team'])

    

    return df
df = get_data('/kaggle/input/nba-all-star-game-20002016/NBA All Stars 2000-2016 - Sheet1.csv')
df
df['Round'] = df['NBADraft'].str.split(' ').apply(lambda x: x[2])

df['Pick'] = df['NBADraft'].str.split(' ').apply(lambda x: x[-1])

df['Round'] = np.where(df['Round'] == 'Draft,','Undrafted',df['Round'])

df['DraftYear'] = df['NBADraft'].str.split(' ').apply(lambda x: x[0]).astype(np.int64)

df.drop('NBADraft',axis = 1, inplace = True)



real_position = {'SF':'F','F-C':'F','PG':'G','SG':'G',

                 'G-F':'G','G':'G','F':'F','C':'C','GF':'F',

                 'FC':'F','PF':'F'}

df['Pos'] = df['Pos'].map(real_position)



df = change_nationality(df)

df = change_team(df)



# correct one row

df['Conference'] = np.where(df.SelectionType.str.startswith("Eastern"),"East","West")

df['Conference'] = np.where(df.Team.str.startswith("New Orleans"),"West",df.Conference)
df.head(10)
df.isnull().sum()
df.info()
teams_count = pd.DataFrame(df.groupby('Team').size()).reset_index()

teams_count.columns = ['Team','Count']



fig = go.Figure()

fig.add_trace(go.Bar(x = teams_count.Team,

                     y = teams_count.Count,

                     text = teams_count.Count,

                     marker=dict(

                      color=teams_count.Count,

                      colorscale='geyser',

                  ),))



fig.update_traces(texttemplate='%{text:s}', textposition='outside')

fig.update_layout(

    title={'text': 'Number of partecipants per Team',

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    xaxis_tickfont_size=12,

    xaxis=dict(tickangle=45),

    yaxis=dict(

        titlefont_size=16,

        tickfont_size=14,

    ),

    showlegend = False,

    barmode='stack',

    bargap=0.1,

    bargroupgap=0.01

)

fig.update_xaxes(type='category',categoryorder='category ascending')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()



title={'text': 'Number of partecipants per Team',

             'y':0.95, 'x':0.5,

            'xanchor': 'center', 'yanchor': 'top'}
df_TY = pd.DataFrame(df.groupby(['Year','Team']).size()).reset_index()

df_TY.columns = ['Year','Team','Count']

map_TC = df[['Team','Conference']].drop_duplicates().set_index('Team').to_dict()['Conference']

df_TY['Conference'] = df_TY.Team.map(map_TC)



df_TY_east = df_TY.loc[df_TY['Conference'] == 'East']

df_TY_west = df_TY.loc[df_TY['Conference'] == 'West']





fig = go.Figure()



East = go.Scatter(x = df_TY_east.Year,

                  y = df_TY_east.Team,

                  marker=dict(

                      size=(df_TY_east.Count+3)**2,

                      color=df_TY_east.Count,

                      colorscale='geyser',

                      showscale=True,

                  ),

                  name = 'East',

                  mode = 'markers',

                  text = df_TY_east.Count)



West = go.Scatter(x = df_TY_west.Year,

                  y = df_TY_west.Team,

                  marker=dict(

                      size=(df_TY_west.Count+3)**2,

                      color=df_TY_west.Count,

                      colorscale='geyser',

                      showscale=True,

                  ),

                  name = 'West',

                  mode = 'markers',

                  text = df_TY_west.Count)



data = [East,West]

updatemenus = list([

    dict(active = 2,

        buttons=list([   

            dict(label = 'East',

                 method = 'restyle',

                 args = ['visible', [True,'legendonly']]

                ),

            dict(label = 'West',

                 method = 'restyle',

                 args = ['visible', ['legendonly',True]]

                ),

            dict(label = 'All',

                 method = 'restyle',

                 args = ['visible', [True,True]])

        ]),

         direction="down",

         pad={"r": 10, "t": 10},

         showactive=True,

         x=-0.2,

         xanchor="left",

         y=1.1,

         yanchor="top"

    )

])



layout = dict(title={'text': 'Number of Partecipant per Year & Team',

                     'y':0.95, 'x':0.5,

                     'xanchor': 'center',

                     'yanchor': 'top'},

              showlegend=False,

              height = 750,

              updatemenus=updatemenus,

              xaxis=dict(

                  categoryorder='category ascending',

              ),

              yaxis=dict(

                  titlefont_size=12,

                  tickfont_size=12,

              ),

              annotations=[

        go.layout.Annotation(text="Select Conference", x=-0.2, xref="paper", y=1.13, yref="paper",

                             align="left", showarrow=False),

              ]

             )

        

fig = dict(data=data, layout=layout)

iplot(fig)
plotly.colors.diverging.Geyser
df_Pos = pd.DataFrame(df.groupby('Pos').size()).reset_index()

df_Pos.columns = ['Pos','Count']

df_Pos_Year = pd.DataFrame(df.groupby(['Year','Pos']).size()).reset_index()

df_Pos_Year.columns = ['Year','Pos','Count']





def color_selection(pos):

    if pos == 'G':

        color = 'rgb(112, 164, 148)'

    elif pos == 'F':

        color = 'rgb(222, 138, 90)'

    else:

        color = 'rgb(246, 237, 189)'

    return color



fig = make_subplots(

    rows=1, cols=2,

    specs=[[{'type':'domain'}, {}]],

    column_widths=[0.3, 0.7],

    subplot_titles=('Positions Frequencies',

                    'Positions Distributions'))



fig.add_trace(

    go.Pie(

        labels = df_Pos.Pos,

        values = df_Pos.Count,

        name = 'Positions',

        text = df_Pos.Count,

        hoverinfo='label+percent+name',

        textinfo= 'percent+label+text',

        textposition = 'inside',

        marker = dict(

            colors = plotly.colors.diverging.Geyser,

        ),

        legendgroup='pie'

    ),

    row = 1,

    col = 1

)



for pos in df_Pos_Year.Pos.unique():



    fig.add_trace(

        go.Bar(

            x=df_Pos_Year.loc[df_Pos_Year['Pos'] == pos].Year,

            y=df_Pos_Year.loc[df_Pos_Year['Pos'] == pos].Count,

            name=pos,

            text = df_Pos_Year.loc[df_Pos_Year['Pos'] == pos].Count,

            texttemplate='%{text:f}',

            textposition='inside',

            legendgroup='bar',

            marker=dict(

                      color=color_selection(pos),

                      colorscale='geyser',

                     ),

        ),

        row = 1,

        col = 2,

    )





fig.update_traces(texttemplate='%{text:s}', textposition='outside')

fig.update_layout(

    title={'text': "Positions at the All-Star Game",

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    

    xaxis_tickfont_size=12,

    yaxis2=dict(

        titlefont_size=16,

        tickfont_size=14,

        range = [0,df_Pos_Year.Count.max()*1.2]

    ),

    height = 500,

    showlegend = True,

    barmode='group',

    bargap=0.1,

    bargroupgap=0.01 

)

fig.update_xaxes(type='category',categoryorder='category ascending')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
df_ST = pd.DataFrame(df.groupby(['SelectionType']).size()).reset_index()

df_ST.columns = ['SelectionType','Count']



fig = go.Figure()

fig.add_trace(go.Bar(x = df_ST.SelectionType,

                     y = df_ST.Count,

                     text = df_ST.Count,

                     marker=dict(

                      color=df_ST.Count,

                      colorscale='geyser',

                     ),

                    )

             )



fig.update_traces(texttemplate='%{text:s}', textposition='inside')

fig.update_layout(

    title={'text': 'Number of partecipants per Team',

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    xaxis_tickfont_size=12,

    xaxis=dict(tickangle=45),

    yaxis=dict(

        titlefont_size=16,

        tickfont_size=14,

        range = [0,df_ST.Count.max()+5]

    ),

    height = 500,

    showlegend = False,

    barmode='stack',

    bargap=0.1,

    bargroupgap=0.01 

)

fig.update_xaxes(type='category',categoryorder='category ascending')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()



title={'text': 'Barplot of Selection Type',

             'y':0.95, 'x':0.5,

            'xanchor': 'center', 'yanchor': 'top'}
df.head()
df.Round.value_counts()
df['Round'] = df.Round.astype('str')





df_R_east = pd.DataFrame(df.drop_duplicates('Player').loc[df['Conference'] == 'East'].groupby(['Round']).size()).reset_index()

df_R_east.columns = ['Round','Count']



df_R_west = pd.DataFrame(df.drop_duplicates('Player').loc[df['Conference'] == 'West'].groupby(['Round']).size()).reset_index()

df_R_west.columns = ['Round','Count']





fig = make_subplots(

    rows=1, cols=2,

    specs=[[{}, {}]],

    subplot_titles=('Eastern Conference',

                    'Western Conference'))

fig.add_trace(

    go.Bar(

        x = df_R_east.Round,

        y = df_R_east.Count,

        text = df_R_east.Count,

        marker=dict(color=df_R_east.Count,

                    colorscale='geyser',

                  ),

        name = 'East',

    ),

    row = 1,

    col = 1,

)



fig.add_trace(

    go.Bar(

        x = df_R_west.Round,

        y = df_R_west.Count,text = df_R_west.Count,

        marker=dict(color=df_R_west.Count,

                    colorscale='geyser',

                   ),

        name = 'West',

    ),

    row = 1,

    col = 2,

)



fig.update_traces(texttemplate='%{text:s}', textposition='outside')

fig.update_layout(

    title={'text': "Barplot of Draft's Round",

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    

    xaxis_tickfont_size=12,

    yaxis1=dict(

        titlefont_size=16,

        tickfont_size=14,

        range = [0,df_R_east.Count.max()+20]

    ),

    yaxis2=dict(

        titlefont_size=16,

        tickfont_size=14,

        range = [0,df_R_west.Count.max()+20]

    ),

    height = 500,

    showlegend = False,

    barmode='stack',

    bargap=0.1,

    bargroupgap=0.01 

)

fig.update_xaxes(type='category',categoryorder='category ascending')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
df.drop_duplicates('Player').loc[df['Round'] == 'Undrafted']
df.loc[df['Round'] == 'Undrafted'].groupby('Player').size()
df['Round'] = np.where(df['Round'] == '3','2',df['Round'])

df = df.loc[df['Round'] != 'Undrafted']

df.Round = df.Round.astype(np.int64)

df.Pick = df.Pick.astype(np.int64)



df['RealPick'] = df[['Round', 'Pick']].prod(axis=1)



condlist = [(df.RealPick.values<=5),

            (df.RealPick.values>5) & (df.RealPick.values<=10),

            (df.RealPick.values>10) & (df.RealPick.values<=20),

            (df.RealPick.values>20) & (df.RealPick.values<=30),

            (df.RealPick.values>30) & (df.RealPick.values<=45),

            (df.RealPick.values>45)]

choicelist = ['1-5','6-10','11-20','21-30','31-45','45-60']

df['PickGroup'] = np.select(condlist, choicelist)
df['Round'] = df.Round.astype('str')





df_P_east = pd.DataFrame(df.drop_duplicates('Player').loc[df['Conference'] == 'East'].groupby(['PickGroup']).size()).reset_index()

df_P_east.columns = ['PickGroup','Count']



df_P_west = pd.DataFrame(df.drop_duplicates('Player').loc[df['Conference'] == 'West'].groupby(['PickGroup']).size()).reset_index()

df_P_west.columns = ['PickGroup','Count']





df_P10_east = pd.DataFrame(df.drop_duplicates('Player').loc[(df['Conference'] == 'East') & (df['RealPick'] <= 10)].groupby(['RealPick']).size()).reset_index()

df_P10_east.columns = ['Pick10','Count']



df_P10_west = pd.DataFrame(df.drop_duplicates('Player').loc[(df['Conference'] == 'West') & (df['RealPick'] <= 10)].groupby(['RealPick']).size()).reset_index()

df_P10_west.columns = ['Pick10','Count']





fig = make_subplots(

    rows=2, cols=2,

    specs=[[{'type':'pie'}, {}],

           [{'type':'pie'},{}]],

    subplot_titles=('Eastern Conference',

                    'Eastern Conference - Top 10',

                    'Western Conference',

                    'Western Conference - Top 10'))



fig.add_trace(

    go.Pie(

        labels = df_P_east.PickGroup,

        values = df_P_east.Count,

        name = 'East',

        text = df_P_east.Count,

        hoverinfo='label+percent+name',

        textinfo= 'percent+label',

        textposition = 'outside',

        marker = dict(

            colors = plotly.colors.diverging.Geyser,

        )

    ),

    row = 1,

    col = 1

)





fig.add_trace(

    go.Pie(

        labels = df_P_west.PickGroup,

        values = df_P_west.Count,

        name = 'West',

        text = df_P_west.Count,

        hoverinfo='label+percent+name',

        textinfo= 'percent+label',

        textposition = 'outside',

        marker = dict(

            colors = plotly.colors.diverging.Geyser,

        )

    ),

    row = 2,

    col = 1

)



fig.add_trace(

    go.Bar(

        x = df_P10_east.Pick10,

        y = df_P10_east.Count,

        text = df_P10_east.Count,

        marker=dict(color=df_P10_east.Count,

                    colorscale='geyser',

                  ),

        texttemplate='%{text:s}',

        textposition='outside',

        name = 'East top 10',

    ),

    row = 1,

    col = 2,

)





fig.add_trace(

    go.Bar(

        x = df_P10_west.Pick10,

        y = df_P10_west.Count,

        text = df_P10_west.Count,

        marker=dict(color=df_P10_west.Count,

                    colorscale='geyser',

                  ),

        texttemplate='%{text:s}',

        textposition='outside',

        name = 'West top 10'

    ),

    row = 2,

    col = 2,

)



fig.update_layout(

    title={'text': "Barplot of Draft's Pick Position",

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    

    xaxis_tickfont_size=12,

    yaxis3=dict(

        titlefont_size=16,

        tickfont_size=14,

        range = [0,df_P10_west.Count.max()*1.3]

    ),

    yaxis4=dict(

        titlefont_size=16,

        tickfont_size=14,

        range = [0,df_P10_west.Count.max()*1.3]

    ),

    height = 1000,

    showlegend = False,

    barmode='stack',

    bargap=0.1,

    bargroupgap=0.01 

)

fig.update_xaxes(type='category',

                 categoryorder='array',

                 categoryarray = ['1-5','6-10','11-20','21-30','31-45','45-60'])

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
df.sort_values(by = 'Year', inplace = True)

df_s = df.drop_duplicates('Player',keep='first')

df_s['GapASG'] = np.subtract(df_s['Year'],df_s['DraftYear'])
df_s.head(10)
import numpy as np



gap_east = df_s.loc[df_s['Conference'] == "East"].GapASG

gap_west = df_s.loc[df_s['Conference'] == "West"].GapASG



hist_data = [gap_east, gap_west]



group_labels = ['East', 'West']



# Create distplot with custom bin_size

fig = ff.create_distplot(

    hist_data,

    group_labels,

    bin_size=1,

    show_rug=False,

    colors = ['rgb(202, 86, 44)', 'rgb(112, 164, 148)'],

)



fig.update_layout(

    title={'text': "Distribution plot by Conference",

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    

    height = 700,

    showlegend = True,

)





fig.show()
df_s.loc[df_s['GapASG'] >= 10].sort_values(by = 'GapASG')
df_s.loc[df_s['GapASG'] < 3].sort_values(by = 'GapASG')
df_1 = pd.DataFrame(df_s.loc[df_s['GapASG'] < 3].groupby('DraftYear').size()).reset_index()

df_1.columns = ['Year','Count']

df_2 = pd.DataFrame(df_s.loc[df_s['GapASG'] < 3].groupby('Nationality').size()).reset_index()

df_2.columns = ['Nationality','Count']

df_2 = pd.DataFrame(df_s.loc[df_s['GapASG'] < 3].groupby('Nationality').size()).reset_index()

df_2.columns = ['Nationality','Count']

df_3 = pd.DataFrame(df_s.loc[df_s['GapASG'] < 3].groupby('Pick').size()).reset_index()

df_3.columns = ['Pick','Count']





fig2 = go.Figure()



fig2.add_trace(

    go.Bar(

        x = df_1.Year,

        y = df_1.Count,

        text = df_1.Count,

        marker=dict(color=df_1.Count,

                    colorscale='geyser',

                  ),

        texttemplate='%{text:s}',

        textposition='outside',

        name = 'East top 10',

    ),

)



fig2.update_layout(

    title={'text': "First Years Selection",

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    

    xaxis_tickfont_size=12,

    yaxis=dict(

        titlefont_size=16,

        tickfont_size=14,

        range = [0,df_1.Count.max()*1.3],

    ),

    height = 700,

    showlegend = False,

    barmode='stack',

    bargap=0.1,

    bargroupgap=0.01,

    uniformtext_minsize=8,

    uniformtext_mode='hide'

)

fig2.update_xaxes(type='category',

                 categoryorder='array',

                 categoryarray = ['1-5','6-10','11-20','21-30','31-45','45-60'])

fig2.update_layout()

fig2.show()





fig = make_subplots(

    rows=1, cols=2,

    specs=[[{"type":"domain"},{}]],

    subplot_titles=('',

                    '',

                    ))





fig.add_trace(

    go.Pie(

        labels = df_s.loc[df_s['GapASG'] < 3].groupby('Nationality').size().index,

        values = df_s.loc[df_s['GapASG'] < 3].groupby('Nationality').size().values,

        name = 'Nationality',

        text = df_s.loc[df_s['GapASG'] < 3].groupby('Nationality').size().values,

        hoverinfo='label+percent+name',

        textinfo= 'percent+label',

        textposition = 'outside',

        marker = dict(

            colors = plotly.colors.diverging.Geyser,

        ),

    ),

    row = 1,

    col = 1

)





fig.add_trace(

    go.Bar(

        x = df_3.Pick,

        y = df_3.Count,

        text = df_3.Count,

        marker=dict(color=df_3.Count,

                    colorscale='geyser',

                  ),

        texttemplate='%{text:s}',

        textposition='inside',

        name = 'Draft Pick',

    ),

    row = 1,

    col = 2,

)





fig.update_layout(

    title={'text': "Insights Player First to Participate",

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    xaxis_tickfont_size=12,

    height = 800,

    showlegend = False,

    barmode='stack',

    bargap=0.1,

    bargroupgap=0.01 

)

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
players_first  = df_s.loc[df_s['GapASG'] < 3].Player



fig = go.Figure()

fig.add_trace(

    go.Bar(

        x=df.loc[df['Player'].isin(players_first)].Player,

        y=df.loc[df['Player'].isin(players_first)].Round,

        name='Partecipation',

        marker=dict(

            color = df.loc[df['Player'].isin(players_first)].Year,

            colorscale = 'geyser'

        ),

        text = df.loc[df['Player'].isin(players_first)].Year.astype('category'),

        texttemplate='%{text:f}',

        textposition='inside',

    )

)

        

fig.update_layout(

    title={'text': "Appearances at All-Star Game",

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15,

    bargroupgap=0.1

)

fig.show()
df_p = pd.DataFrame(df.groupby('Player').size()).reset_index()

df_p.columns = ['Player','Count']



df_p.loc[df_p['Count']>5]



fig = go.Figure()

fig.add_trace(

    go.Bar(

        x=df_p.Player,

        y=df_p.Count,

        name='Player',

        marker=dict(

            color = df_p.Count,

            colorscale = 'geyser'

        ),

        text = df_p.Count,

        texttemplate='%{text:s}',

        textposition='inside',

    )

)

        

fig.update_layout(

    title={'text': "Appearances at All-Star Game Overall",

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    xaxis_tickfont_size=14,

    xaxis=dict(tickangle=45),

    yaxis=dict(

        title='',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='stack',

    bargap=0.15,

    bargroupgap=0.1

)

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()







fig2 = go.Figure()

fig2.add_trace(

    go.Bar(

        x=df_p.loc[df_p['Count']>5].Player,

        y=df_p.loc[df_p['Count']>5].Count,

        name='Player',

        marker=dict(

            color = df_p.loc[df_p['Count']>5].Count,

            colorscale = 'geyser'

        ),

        text = df_p.loc[df_p['Count']>5].Count,

        texttemplate='%{text:s}',

        textposition='inside',

    )

)

        

fig2.update_layout(

    title={'text': "Appearances at All-Star Game - Best Players",

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    xaxis_tickfont_size=14,

    xaxis=dict(tickangle=45),

    yaxis=dict(

        title='',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='stack',

    bargap=0.15,

    bargroupgap=0.1

)

fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig2.show()
df_Nat = (df.groupby(['Year','Nationality']).size()).reset_index()

df_Nat.columns = ['Year','Nationality','Count']



def color_selection(nat):

    if nat == 'United States':

        color = 'rgb(202, 86, 44)'

    else:

        color = 'rgb(112, 164, 148)'



        

    return color





fig = go.Figure()



for nat in df_Nat.Nationality.unique():

    fig.add_trace(

        go.Bar(

            x=df_Nat.loc[df_Nat['Nationality'] == nat].Year,

            y=df_Nat.loc[df_Nat['Nationality'] == nat].Count,

            name=nat,

            marker=dict(

                color = color_selection(nat),

                colorscale = 'geyser'

            ),

            text = df_Nat.loc[df_Nat['Nationality'] == nat].Count,

            texttemplate='%{text:f}',

            textposition='inside',

        )

    )

        

fig.update_layout(

    title={'text': "Nationality at All-Star Game",

           'y':0.95,

           'x':0.5,

           'xanchor': 'center',

           'yanchor': 'top'},

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='',

        titlefont_size=16,

        tickfont_size=14,

    ),

    barmode='group',

    bargap=0.05,

    bargroupgap=0.01

)

fig.show()