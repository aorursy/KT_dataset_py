import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import plotly as py

import plotly.graph_objs as go

import seaborn as sns

sns.set(style = 'dark')

import random

from collections import Counter as counter

from IPython.display import HTML

import os

import warnings



warnings.filterwarnings("ignore")

py.offline.init_notebook_mode(connected = True)
HTML('''

<script>

  function code_toggle() {

    if (code_shown){

      $('div.input').hide('500');

      $('#toggleButton').val('Show Code')

    } else {

      $('div.input').show('500');

      $('#toggleButton').val('Hide Code')

    }

    code_shown = !code_shown

  }



  $( document ).ready(function(){

    code_shown=false;

    $('div.input').hide()

  });

</script>

<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
fifa_df = pd.read_csv('../input/united/united_players.csv')
pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)
useful_features = ['Name',

                   'Age',

                   'Photo', 

                   'Nationality', 

                   'Flag',

                   'Overall',

                   'Potential', 

                   'Club', 

                   'Club Logo', 

                   'Value',

                   'Wage',

                   'Preferred Foot',

                   'International Reputation',

                   'Weak Foot',

                   'Skill Moves',

                   'Work Rate',

                   'Body Type',

                   'Position',

                   'Joined', 

                   'Contract Valid Until',

                   'Height',

                   'Weight',

                   'Crossing', 

                   'Finishing',

                   'HeadingAccuracy',

                   'ShortPassing', 

                   'Volleys', 

                   'Dribbling',

                   'Curve',

                   'FKAccuracy',

                   'LongPassing',

                   'BallControl',

                   'Acceleration',

                   'SprintSpeed',

                   'Agility',

                   'Reactions', 

                   'Balance',

                   'ShotPower', 

                   'Jumping',

                   'Stamina', 

                   'Strength',

                   'LongShots',

                   'Aggression',

                   'Interceptions',

                   'Positioning', 

                   'Vision', 

                   'Penalties',

                   'Composure',

                   'Marking',

                   'StandingTackle', 

                   'SlidingTackle',

                   'GKDiving',

                   'GKHandling',

                   'GKKicking',

                   'GKPositioning',

                   'GKReflexes']
df = pd.DataFrame(fifa_df , columns = useful_features)
class fast_plot():

    def __init__(self):

        c = ['r' , 'g' , 'b' , 'y' , 'orange' , 'grey' , 'lightcoral' , 'crimson' , 

            'springgreen' , 'teal' , 'c' , 'm' , 'gold' , 'skyblue' , 'darkolivegreen',

            'tomato']

        self.color = c

        

    

    def regplot_one_vs_many(self , x  , y  , data , rows , cols):

        color_used = []

        

        n = 0

        for feature in y:

            

            for i in range(1000):

                colour = random.choice(self.color)

                if colour not in color_used:

                    color_used.append(colour)

                    break

                    

            n += 1 

            plt.subplot(rows , cols , n)

            plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

            sns.regplot(x  = x , y = feature , data = data , 

                        color = colour)

    

    def bar_plot(self , x = None, y = None, data = None ,x_tick_rotation = None ,xlabel = None , ylabel = None , title = ''):

        ax = sns.barplot(x = x , y = y , data = data , palette = 'rocket')

        rects = ax.patches

        for rect , label in zip(rects , data[y]):

            height = rect.get_height()

            ax.text(rect.get_x() + rect.get_width() / 2, height + (height * 0.1 /100), round(label,1),

                    ha='center', va='bottom')

        if not xlabel == None and not ylabel == None:

            plt.xlabel(xlabel)

            plt.ylabel(ylabel)

        if not x_tick_rotation == None:

            plt.xticks(rotation = x_tick_rotation)

        plt.title(title)

        

        

plots = (fast_plot())
plt.figure(1 , figsize = (15 , 6))

df['Age'].plot(kind = 'hist' , bins = 50)

plt.title('Histogram of Age of players')

plt.show()
df.sort_values(by = 'Age' , ascending = False)[['Name','Club','Nationality'

                                               ,'Overall', 'Age' ]].head(5)
df.sort_values(by = 'Age' , ascending = True)[['Name','Age','Club','Nationality'

                                               ,'Overall' ]].head(5)
vals = ['Stamina' , 'Strength' , 'Acceleration','SprintSpeed' , 'Agility' , 'Jumping' ,

       'Vision','Reactions']

plt.figure(1 , figsize = (15 , 10))

plots.regplot_one_vs_many(x = 'Age' , y = vals , data = df , rows = 2 , cols = 4 )

plt.show()
plt.figure(1 , figsize = (15 , 6))

sns.regplot(df['Age'] , df['Overall'])

plt.title('Scatter Plot of Age vs Overall rating')

plt.show()
plt.figure(1 , figsize = (15 , 7))

countries = []

c = counter(df['Nationality']).most_common()[:11]

for n in range(11):

    countries.append(c[n][0])



sns.countplot(x  = 'Nationality' ,

              data = df[df['Nationality'].isin(countries)] ,

              order  = df[df['Nationality'].isin(countries)]['Nationality'].value_counts().index , 

             palette = 'rocket') 

plt.xticks(rotation = 90)

plt.title('Maximum number footballers belong to which country' )

plt.show()

plt.figure(1 , figsize = (15 , 7))

df['Overall'].plot(kind = 'hist' , bins = 50 )

plt.title('Histogram of Overall players ratings out of 100')

plt.show()
df_best_players = pd.DataFrame.copy(df.sort_values(by = 'Overall' , 

                                                   ascending = False ).head(20))



plt.figure(1 , figsize = (15 , 5))

plots.bar_plot(x ='Name' , y = 'Overall' , data = df_best_players , 

              x_tick_rotation = 50 , xlabel = 'Name of Players' , 

              ylabel = 'Overall Rating out of 100' ,

               title = 'Top 20 players according to Overall Rating out of 100')

plt.ylim(80 , 95)

plt.show()
player_features = ['Crossing', 'Finishing', 'HeadingAccuracy',

       'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',

       'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',

       'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',

       'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',

       'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',

       'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

       'GKKicking', 'GKPositioning', 'GKReflexes']
df[df["Club"] == "Manchester United"][['Name' , 'Position' , 'Overall' , 'Age']]
best_dict = {}

for nation in df['Nationality'].unique():

    overall_rating = df['Overall'][df['Nationality'] == nation].sum()

    best_dict[nation] = overall_rating

df_bnp = pd.DataFrame.from_dict(best_dict , orient = 'index' , 

                                                 columns = ['overall'])

df_bnp['nation'] = df_bnp.index

df_bnp = df_bnp.sort_values(by = 'overall' , ascending =  False)



plt.figure(1 , figsize = (15 , 6))

plots.bar_plot(x = 'nation' , y = 'overall' , data = df_bnp.head(10) , 

              x_tick_rotation = 50 , xlabel = 'Countries' , 

              ylabel = 'Sum of Overall Rating of players w.r.t Nationality',

              title = 'Countries with best Players (sum of overall ratings of players per club)')

plt.show()
plt.figure(1 , figsize = (15 , 6))

df['Potential'].plot(kind = 'hist' , bins = 50)

plt.title('Histogram of Potential of players (out of 100)')

plt.show()
def cleaning_value(x):

    if '€' in str(x) and 'M' in str(x):

        c = str(x).replace('€' , '')

        c = str(c).replace('M' , '')

        c = float(c) * 1000000

        

    else:

        c = str(x).replace('€' , '')

        c = str(c).replace('K' , '')

        c = float(c) * 1000

            

    return c



fn = lambda x : cleaning_value(x)



df['Value_num'] = df['Value'].apply(fn)
df.sort_values(by = 'Value_num' , ascending = False)[['Name' , 'Club' , 'Nationality' , 

                                                     'Overall' , 'Value' , 'Wage']].head(5)
df.sort_values(by = 'ShotPower' , ascending = False)[['Name' , 'Club' , 'Nationality' , 

                                                     'ShotPower' ]].head(5)
df.sort_values(by = 'LongPassing' , ascending = False)[['Name' , 'Club' , 'Nationality' , 

                                                     'LongPassing']].head(5)
df.sort_values(by = 'Vision' , ascending = False)[['Name' , 'Club' , 'Nationality' , 

                                                     'Vision' ]].head(5)
plt.figure(1 , figsize = (25 , 16))

n = 0

for feat in player_features:

    n += 1

    plt.subplot(6 , 6 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    df[feat].plot(kind = 'hist' , bins = 50)

    plt.ylabel('')

    plt.xlabel(feat)



plt.show()
vals = ['ShortPassing' , 'LongPassing' , 'Crossing']

plt.figure(1 , figsize = (15 , 9))

plots.regplot_one_vs_many(x = 'ShotPower' , y = vals , data = df , 

                         rows = 1 , cols = 3)

plt.show()
vals = ['Dribbling' , 'Acceleration' , 'SprintSpeed']

plt.figure(1 , figsize = (15 , 6))

plots.regplot_one_vs_many(x = 'BallControl' , y = vals , data = df , 

                         rows = 1 , cols = 3)

plt.show()
vals = ['HeadingAccuracy' , 'FKAccuracy' , 'Volleys' , 'Penalties' , 'LongShots']

plt.figure(1 , figsize = (15 , 9))

plots.regplot_one_vs_many(x = 'Finishing', y = vals , data = df , 

                         rows = 2 , cols = 3)

plt.show()
def scatter3D(x , y , z , txt , xlabel , ylabel , zlabel , title):

    camera = dict(

        up=dict(x=0, y=0, z=1),

        center=dict(x=0, y=0, z=0),

        eye=dict(x=2, y=2, z=0.1))

    trace0 = go.Scatter3d(

        x = x,

        y = y,

        z = z,

        mode = 'markers',

        text  = txt,

        marker = dict(

            size = 12,

            color = z,

            colorscale = 'Viridis',

            showscale = True,

            line = dict(

                color = 'rgba(217 , 217 , 217 , 0.14)',

                width = 0.5

            ),

            opacity = 0.8

        )

    )

    

    layout = go.Layout(

        title = title,

        scene = dict(

            camera = camera,

            xaxis = dict(title  = xlabel),

            yaxis = dict(title  = ylabel),

            zaxis = dict(title  = zlabel)

        )

    )

    data = [trace0]

    fig = go.Figure(data = data , layout = layout)

    py.offline.iplot(fig)
scatter3D(df['Acceleration'].where(df['Acceleration'] > 70) ,

         df['Composure'].where(df['Composure'] > 70),

         df['Finishing'].where(df['Finishing'] > 70), 

         df['Name'],

         'Acceleration' , 

         'Composure',

         'Finishing',

         'Best Finishers')
scatter3D(df['LongPassing'].where(df['LongPassing'] > 70) ,

         df['ShortPassing'].where(df['ShortPassing'] > 70) , 

         df['Vision'].where(df['Vision'] > 70) , 

         df['Name'],

         'Long passing' , 'Short passing' , 'Vision' , 

         'Best Passes & Assists')
scatter3D(df['Volleys'].where(df['Volleys'] > 70) ,

         df['Curve'].where(df['Curve'] > 70),

         df['ShotPower'].where(df['ShotPower'] > 70),

         df['Name'],

         'Volleys','Curve' , 'Shot Power',

         'Best at Volley Goals')
scatter3D(df['Finishing'].where(df['Finishing'] > 75) ,

         df['Vision'].where(df['Vision'] > 75),

         df['Penalties'].where(df['Penalties'] > 75),

         df['Name'],

         'Finishing','Vision' , 'Penalties',

         'Best Penalty Takers')
scatter3D(df['Dribbling'].where(df['Dribbling'] > 80) ,

         df['SprintSpeed'].where(df['SprintSpeed'] > 80),

         df['Finishing'].where(df['Finishing'] > 80),

         df['Name'],

         'Dribbling','Sprint Speed' , 'Finishing',

         'Best At Solo Goals')
scatter3D(df['Skill Moves'].where(df['Skill Moves'] >= 0.4) ,

         df['BallControl'].where(df['BallControl'] > 80),

         df['Reactions'].where(df['Reactions'] > 80),

         df['Name'],

         'Skill Moves', 'Ball Control' , 'Reactions',

         'Best Footwork')
scatter3D(df['HeadingAccuracy'].where(df['HeadingAccuracy'] > 70) ,

         df['Jumping'].where(df['Jumping'] > 80),

         df['Finishing'].where(df['Finishing'] > 70),

         df['Name'],

         'Heading Acc', 'Jumping' , 'Finishing',

         'Best Headers')
scatter3D(df['Vision'].where(df['Vision'] > 80) ,

         df['FKAccuracy'].where(df['FKAccuracy'] > 80),

         df['LongPassing'].where(df['LongPassing'] > 80),

         df['Name'],

         'Vision', 'Free kick Acc' , 'Long passing',

         'Freekick Specialists')
scatter3D(df['LongShots'].where(df['LongShots'] > 72) ,

         df['ShotPower'].where(df['ShotPower'] > 72),

         df['Finishing'].where(df['Finishing'] > 72),

         df['Name'],

         'LongShots','Shot Power' , 'Finishing',

         'Long Distance Scorers')
scatter3D(df['SlidingTackle'].where(df['SlidingTackle'] > 75) ,

         df['Strength'].where(df['Strength'] > 75),

         df['Aggression'].where(df['Aggression'] > 75),

         df['Name'],

         'Sliding Tackle','Strength' , 'Aggression',

         'Best Defenders')
scatter3D(df['Marking'].where(df['Marking'] > 80) ,

         df['Strength'].where(df['Strength'] > 80),

         df['StandingTackle'].where(df['StandingTackle'] > 80),

         df['Name'],

         'Marking','Strength' , 'Standing Tackle',

         'Best Defenders')
scatter3D(df['GKDiving'].where(df['GKDiving'] > 85) ,

         df['GKReflexes'].where(df['GKReflexes'] > 85),

         df['GKPositioning'].where(df['GKPositioning'] > 85),

         df['Name'],

         'Diving','Reflexes' , 'Positioning',

         'Best Goal Keepers')
plt.figure(1 , figsize = (15 , 6))

sns.countplot(x = 'Position' , data = df , palette = 'inferno_r' )

plt.title('Count Plot of Postions of player')

plt.show()
for i, val in df.groupby(df['Position'])[player_features].mean().iterrows():

    print('Position {}: {}, {}, {}'.format(i, *tuple(val.nlargest(3).index)))
idx = 1

plt.figure(figsize=(15,45))

for position_name, features in df.groupby(df['Position'])[player_features].mean().iterrows():

    top_features = dict(features.nlargest(5))

    

    # number of variable

    categories=top_features.keys()

    N = len(categories)



    # We are going to plot the first line of the data frame.

    # But we need to repeat the first value to close the circular graph:

    values = list(top_features.values())

    values += values[:1]



    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]

    angles += angles[:1]



    # Initialise the spider plot

    ax = plt.subplot(9, 3, idx, polar=True)



    # Draw one axe per variable + add labels labels yet

    plt.xticks(angles[:-1], categories, color='grey', size=8)



    # Draw ylabels

    ax.set_rlabel_position(0)

    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)

    plt.ylim(0,100)

    

    plt.subplots_adjust(hspace = 0.5)

    

    # Plot data

    ax.plot(angles, values, linewidth=1, linestyle='solid')



    # Fill area

    ax.fill(angles, values, 'b', alpha=0.1)

    

    plt.title(position_name, size=11, y=1.1)

    

    idx += 1 
df_postion  = pd.DataFrame()

for position_name, features in df.groupby(df['Position'])[player_features].mean().iterrows():

    top_features = dict(features.nlargest(5))

    df_postion[position_name] = tuple(top_features)

df_postion.head()