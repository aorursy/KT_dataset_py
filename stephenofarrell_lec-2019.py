import requests

import csv

import lxml.html as lh

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import chart_studio.plotly as py

import plotly.express as px

import plotly.graph_objs as go

import warnings

import matplotlib.pyplot as plt

import matplotlib.image as mpimg





warnings.filterwarnings('ignore')

pd.set_option('display.max_rows',None)



fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 15

fig_size[1] = 12

plt.rcParams["figure.figsize"] = fig_size



from IPython.display import Image

from IPython.display import HTML

from collections import Counter



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



teams = {'G2 Esports':'black',

         'Fnatic':'orange',

         'Splyce':'red',

         'FC Schalke 04':'cyan',

         'Origen':'blue',

         'Team Vitality':'yellow',

         'Misfits Gaming':'red',

         'SK Gaming':'black',

         'Rogue':'blue',

         'Excel Esports':'navy'}

switcher = {

    "G2 Esports": "G2",

    "Team Vitality": "VIT",

    "Rogue": "RGE",

    "Fnatic": "FNC",

    "FC Schalke 04": "S04",

    "SK Gaming": "SK",

    "Excel Esports": "XL",

    "Misfits Gaming": "MSF",

    "Origen": "OG",

    "Splyce": "SPY",

}

def swap(x):

    temp = x['Team1']

    x['Team1'] = x['Team2']

    x['Team2'] = temp

    x['Result'] = (False if (x['Result']) else True)

    return(x)



def swaps(x):

    df.loc[x] = swap(df.loc[x])

df = pd.read_csv("/kaggle/input/league-of-legends-european-championship-2019/lec2019_datasets/lec_matchdata.csv")

df2 = pd.read_csv("/kaggle/input/league-of-legends-european-championship-2019/lec2019_datasets/lec_playerdata.csv")

df3 = pd.read_csv("/kaggle/input/league-of-legends-european-championship-2019/lec2019_datasets/lec_championdata.csv")
sums = df[['Sel','Choice']].groupby('Sel').sum()

totals = df['Sel'].value_counts().sort_index()

winrate_totals=[sums.iloc[i]/totals[i] for i in range(10)]

bardata = pd.concat(winrate_totals,axis = 1,keys = [s.name for s in winrate_totals])



px.bar(bardata, x=bardata.columns, y=[bardata.values[0][i] for i in range(10)])



fig = go.Figure(data=[

    go.Bar(name='Blue Side', 

           x=bardata.columns, 

           y=[bardata.values[0][i] for i in range(10)],

           marker_color = "#1d0efd",

           textposition="auto",

           text=["%s%%" % round(100*bardata.values[0][i],1) for i in range(10)]),



    

    go.Bar(name='Red Side', 

           x=bardata.columns, 

           y=[1-bardata.values[0][i] for i in range(10)],

           marker_color = "#fd0e35")

])

fig.update_yaxes(showticklabels=False)

fig.update_layout(

    barmode='stack',

    title = "Side Selection for Each Team: Regular Season",

    font=dict(

        family="Calibri, monospace",

        size=16,

        color="#7f7f7f")

                 )

fig.show()
def scores(team):

    for i in (df[df['Team2'] == team].index):

        swaps(i)

    team_results = df[df['Team1'] == team].append(df[df['Team2'] == team]).sort_values('UTC')

    score = team_results.groupby('Team1').cumsum().set_index(team_results['UTC'])

    return(score)
casters=df[['Team1','Team2','PBP','Color']]

casters['Combo'] = casters['PBP'] + " + " + casters['Color']

caster_combos = casters.groupby('Combo').count()

test = casters.sort_values('Combo')['Combo'].drop_duplicates()

parents = ["",""]

for i in test:

    if("," in i):

        parents.append("Tri Cast")

    else:

        parents.append("Duo Cast")

labels_src = casters.sort_values("Combo")['Combo'].drop_duplicates()

labels = ["Tri Cast","Duo Cast"]

for label in labels_src:

    labels.append(label)



values = [0,0]

for value in caster_combos['Team1']:

    values.append(value)

    

fig = go.Figure(go.Treemap(labels=labels,

                           parents = parents,

                           values = values))

fig.update_layout(title="Caster Combo Distribution: Regular Season")

fig.show()
def casters_treemap(caster_type):

    pbp = pd.DataFrame([i for i in casters[caster_type].drop_duplicates() if ',' not in i],columns=[caster_type])

    pbp = pbp.rename(index=pbp[caster_type])

    pbp = pbp.drop(caster_type,axis=1)





    for team in teams:

        pbp[team] = 0

    for i in range(len(casters)):

        ind = casters.iloc[i]

        pbp_index = ind[caster_type].replace(" ","").split(",")

        for caster in pbp_index:    

            pbp.loc[caster,ind['Team1']] += 1

            pbp.loc[caster,ind['Team2']] += 1



    biglist = [len(pbp)*['o'],[10*[i] for i in list(pbp.index)]]

    parents = [val for sublist in biglist for val in sublist]

    parents = [val for sublist in parents for val in sublist]

    for i in range(len(pbp)):

        parents[i] = ""



    values = pbp.sum(axis=1)

    for i in list(pbp.index):

        values = values.append(pbp.loc[i])



    fig = go.Figure()



    fig = fig.add_trace(go.Treemap(

        labels = pbp.index.append(len(pbp)*[pbp.columns]),

        parents = parents,

        values = values,

        branchvalues= 'total'

    ))

    fig.update_layout(title='%s Team Distribution (Regular Season)' % caster_type)

    fig.show()

    

for i in ['PBP','Color']:

    casters_treemap(i)
df = df.drop([90,181]) #dropping tiebreakers
mvps = df.groupby('MVP').count().sort_values('Team1',ascending = False)



fig = go.Figure()

fig.add_trace(go.Bar(y=mvps.index[range(5)][::-1], 

                     x=mvps['Team1'].head()[::-1],

                     orientation='h',

                    marker_color = "purple"))

fig.update_layout(title_text = 'Total MVP awards')

fig.show()
df = df.drop("Unnamed: 0",axis = 1)

test = pd.DataFrame(columns = teams,index= df['UTC'])

for x in teams:

    test[x] = scores(x)['Result']

test = test.sort_values('UTC').drop_duplicates()
fig = go.Figure(layout=go.Layout(

        title=go.layout.Title(text="Cumulative Wins: Regular Season Games")

    ))

def plotcumwins(x):

    fig.add_trace(go.Scatter(x=list(range(1,len(test[x])+1)),

                               y=test[x],

                               name = x,

                               line = dict(color=teams[x])))

for i in teams:

    plotcumwins(i)

fig.add_shape(go.layout.Shape(type="line",x0=18,y0=0,x1=18,y1=30,line=dict(

                color="RoyalBlue",

                width=1

            )))

fig.update_xaxes(title_text='Gameday',range=[0,36])

fig.update_yaxes(title_text='Total Number of Wins')

fig
tops = df2[df2['Pos'] == "TOP"]

jgls = df2[df2['Pos'] == "Jungle"]

mids = df2[df2['Pos'] == "Middle"]

adcs = df2[df2['Pos'] == "ADC"]

sups = df2[df2['Pos'] == "Support"]


img=mpimg.imread('/kaggle/input/lecnationalities/nationalities.png')

imgplot = plt.imshow(img)

years = [0,2,1,2,1,4,1,2,0,1,2,3,4,0,6,2,2,0,4,4,1,5,0,1,5,8,3,4,3,3,3,6,3,0,0,1,6,4,0,1,3,3,2,4,7,0,0,3,8,0,1,5,4,2,4,3,2,0,0,2,4,0,0,3,0,0,0,4,1,0,4,5,0]

x = Counter(years)



fig = go.Figure(go.Pie(labels=["%s yr" % i for i in sorted(x.keys())],values=[x[i] for i in sorted(x.keys())],sort=False))

fig.update_layout(title_text="Years since Major Region Debut")

fig.show()
x = df2[['Player','Split']].groupby('Split').count().iloc[[2,4]]

fig = go.Figure(go.Bar(x = x.index, 

                       y = list(x['Player']),

                      marker_color = "#6baed6"))



fig.update_layout(title_text = "Number of Players per Split")

fig.update_yaxes(title_text="Number of Players")



fig.show()
fig = px.scatter(x=adcs['Gold%'],

                 y=adcs['DMG%'],

                 color=adcs['Split'],

                 hover_name=adcs['Player'])

fig.update_xaxes(title_text="Gold%",range=[35,15])

fig.update_yaxes(title_text="Damage%")

fig.update_layout(title_text="ADC Gold Efficiency")

fig.show()
fig = px.scatter(y=mids['GD10'],

                 x = mids['XPD10'],

                hover_name = mids['Player'],

                color = mids['Split'])



fig.update_xaxes(title_text="XP Difference at 10")

fig.update_yaxes(title_text="Gold Difference at 10")

fig.update_layout(title_text=" Midlaner Laning Dominance")

fig.show()
fig = px.scatter(x=sups['WPM'],

                 y=sups['WCPM'],

                 color=sups['Split'],

                 hover_name=sups['Player'])

fig.update_xaxes(title_text="Wards Placed Per Minute")

fig.update_yaxes(title_text="Wards Cleared Per Minute")

fig.update_layout(title_text="Support Vision Control")

fig.show()
time_splits = df3.columns[[15,18,21,24,27,30]]

times = pd.DataFrame(index = time_splits, columns = ["Spring","Summer"])

grouped_df3 = df3.groupby('Split').sum()

for i in time_splits:

    times['Spring'][i] = (grouped_df3[i][0] / 10) / 107

    times['Summer'][i] = (grouped_df3[i][1] / 10) / 114



times.columns

time_splits = [i.replace("games","minutes") for i in time_splits] 
fig = go.Figure(data=[

    go.Bar(name='Spring', x=time_splits, y=times['Spring']),

    go.Bar(name='Summer', x=time_splits, y=times['Summer'])

])

fig.update_layout(title_text = "Game Length: Spring v Summer Split")

fig.update_yaxes(title_text = "Percentage of Games")

fig.show()
champs = df3[['Champion 1','∑ 2','W 3','L 4']].groupby('Champion 1').sum().sort_values("∑ 2",ascending = False)

x = df3[['W 3','Split']].groupby('Split').count()



fig = go.Figure(go.Bar(x = x.index, 

                       y = list(x['W 3']),

                      marker_color = "#f768a1"))



fig.update_layout(title_text = "Number of Unique Champions per Split")

fig.update_yaxes(title_text="Number of Champions")



fig.show()
fig = go.Figure(data=[

    go.Bar(name='Wins', 

           x=champs.index[range(10)], 

           y=champs['W 3'],

           marker_color = "#1c9099",

           textposition="auto"),



    

    go.Bar(name='Losses', 

           x=champs.index[range(10)], 

           y=champs['L 4'],

           marker_color = "#a6bddb",

           textposition = "auto",

            text = champs['∑ 2']),

          

])

fig.update_yaxes(title_text = "Number of Games")

fig.update_layout(

    barmode='stack',

    title = "Most Picked Champions: Spring & Summer Split",

    font=dict(

        family="Calibri, monospace",

        size=16,

        color="#7f7f7f")

                 )

fig.show()
champs['WR'] = champs['W 3'] / champs['∑ 2']

champs = champs[champs['∑ 2'] > 10].sort_values('WR',ascending = False)



fig = go.Figure(data=[

    go.Bar(name='Winrate', 

           x=champs.index[range(10)], 

           y=champs['WR'],

           marker_color = "#fec44f",

           textposition="auto",

          text=["%s%%" % round(100*champs['WR'].iloc[i],1) for i in range(10)]),

])

fig.update_yaxes(title_text = "Win Percentage")

fig.update_layout(

    barmode='stack',

    title = "Highest Winrate Champions: Spring & Summer Split (Min 10 Games)",

    font=dict(

        family="Calibri, monospace",

        size=16,

        color="#7f7f7f")

                 )

fig.show()
champs = champs[champs['∑ 2'] > 10].sort_values('WR')

fig = go.Figure(data=[

    go.Bar(name='Winrate', 

           x=champs.index[range(10)], 

           y=champs['WR'],

           marker_color = "#78c679",

           textposition="auto",

          text=["%s%%" % round(100*champs['WR'].iloc[i],1) for i in range(10)]),

])

fig.update_yaxes(title_text = "Win Percentage")

fig.update_layout(

    barmode='stack',

    title = "Lowest Winrate Champions: Spring & Summer Split (Min 10 Games)",

    font=dict(

        family="Calibri, monospace",

        size=16,

        color="#7f7f7f")

                 )

fig.show()
fig = go.Figure(go.Bar(x=["Fnatic","Promisq"],y=[0,2],marker_color="grey"))

fig.update_layout(title_text="Number of LEC Medals")

fig.show()