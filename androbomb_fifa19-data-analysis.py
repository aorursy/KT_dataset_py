# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

print('\n ')

print('Getting traing dataset...')

data = pd.read_csv('../input/fifa19/data.csv', index_col=0)

print('Traing data set obtained. \n')
data.head(3)
data.drop('Photo', inplace=True, axis=1)

data.drop('Flag', inplace=True, axis=1)

data.drop('Club Logo', inplace=True, axis=1)

data.fillna(0, inplace=True)

data.head(3)
data.columns
stats = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing',

       'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing',

       'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions',

       'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',

       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',

       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',

       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']



for st in stats: 

    print('Best 3 players in', st)

    print(data.sort_values(st, ascending = False)[['Name', st]].head(3))

    print('\n')

    
plt.figure(figsize=(15,8))

for st in stats:

    b = sns.distplot(data[st], label=st, kde=False)

    b.set_xlabel('Stats', fontsize=20)

    b.set_ylabel('Count', fontsize=20)

    

plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
stats_ballskills = ['BallControl', 'Dribbling']

stats_defence = ['Marking', 'SlidingTackle', 'StandingTackle']

stats_mental = ['Aggression', 'Reactions', 'Positioning', 'Interceptions', 'Vision', 'Composure']

stats_passing = ['Crossing', 'ShortPassing', 'LongPassing']

stats_physical = ['Acceleration', 'Stamina', 'Strength', 'Balance', 'SprintSpeed', 'Agility', 'Jumping']

stats_shooting = ['HeadingAccuracy', 'ShotPower', 'Finishing', 'LongShots', 'Curve', 'FKAccuracy', 'Penalties', 'Volleys']



stats_GK = ['GKDiving','GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']



plt.figure(figsize=(15,8))

for st in stats_ballskills:

    b = sns.distplot(data[st], label=st, kde=False)

    plt.title('Ball Skills', fontsize=25)

    b.set_xlabel('Stats', fontsize=20)

    b.set_ylabel('Count', fontsize=20)

    

plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()



plt.figure(figsize=(15,8))

for st in stats_defence:

    b = sns.distplot(data[st], label=st, kde=False)

    plt.title('Defence', fontsize=25)

    b.set_xlabel('Stats', fontsize=20)

    b.set_ylabel('Count', fontsize=20)

    

plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()



plt.figure(figsize=(15,8))

for st in stats_mental:

    b = sns.distplot(data[st], label=st, kde=False)

    plt.title('Mental', fontsize=25)

    b.set_xlabel('Stats', fontsize=20)

    b.set_ylabel('Count', fontsize=20)

    

plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()



plt.figure(figsize=(15,8))

for st in stats_passing:

    b = sns.distplot(data[st], label=st, kde=False)

    plt.title('Passing', fontsize=25)

    b.set_xlabel('Stats', fontsize=20)

    b.set_ylabel('Count', fontsize=20)

    

plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()



plt.figure(figsize=(15,8))

for st in stats_physical:

    b = sns.distplot(data[st], label=st, kde=False)

    plt.title('Physical', fontsize=25)

    b.set_xlabel('Stats', fontsize=20)

    b.set_ylabel('Count', fontsize=20)

    

plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()



plt.figure(figsize=(15,8))

for st in stats_shooting:

    b = sns.distplot(data[st], label=st, kde=False)

    plt.title('Shooting', fontsize=25)

    b.set_xlabel('Stats', fontsize=20)

    b.set_ylabel('Count', fontsize=20)

    

plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()



plt.figure(figsize=(15,8))

for st in stats_GK:

    c = sns.distplot(data[st], label=st, kde=False)

    plt.title('Goalkeeper', fontsize=25)

    c.set_xlabel('Stats', fontsize=20)

    c.set_ylabel('Count', fontsize=20)

    

plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
plt.figure(figsize=(15,8))

sns.distplot(data[data['Age']<23]['Overall'], label='Under 23', kde=False)

sns.distplot(data[(data['Age']>=23) & (data['Age']<28)]['Overall'], label='[23, 28)', kde=False)

sns.distplot(data[(data['Age']>=28) & (data['Age']<33)]['Overall'], label='[28, 33)', kde=False)

c = sns.distplot(data[data['Age']>=33]['Overall'], label='Over 33', kde=False)

plt.title('Overall by age (Histogram)', fontsize=25)

c.set_xlabel('Overall', fontsize=20)

c.set_ylabel('Count', fontsize=20)

plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()



plt.figure(figsize=(15,8))

sns.distplot(data[data['Age']<23]['Overall'], label='Under 23', kde=True, hist=False)

sns.distplot(data[(data['Age']>=23) & (data['Age']<28)]['Overall'], label='[23, 28)', kde=True,hist=False)

sns.distplot(data[(data['Age']>=28) & (data['Age']<33)]['Overall'], label='[28, 33)', kde=True,hist=False)

c = sns.distplot(data[data['Age']>=33]['Overall'], label='Over 33', kde=True,hist=False)

plt.title('Overall by age (KDE)', fontsize=25)

c.set_xlabel('Overall', fontsize=20)

c.set_ylabel('Percentage', fontsize=20)

plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
plt.figure(figsize=(15,8))

sns.distplot(data[data['Age']<23]['Potential'], label='Under 23', kde=False)

sns.distplot(data[(data['Age']>=23) & (data['Age']<28)]['Potential'], label='[23, 28)', kde=False)

sns.distplot(data[(data['Age']>=28) & (data['Age']<33)]['Potential'], label='[28, 33)', kde=False)

c = sns.distplot(data[data['Age']>=33]['Potential'], label='Over 33', kde=False)

plt.title('Potential by age (Histogram)', fontsize=25)

c.set_xlabel('Potential', fontsize=20)

c.set_ylabel('Count', fontsize=20)

plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()



plt.figure(figsize=(15,8))

sns.distplot(data[data['Age']<23]['Potential'], label='Under 23', kde=True, hist=False)

sns.distplot(data[(data['Age']>=23) & (data['Age']<28)]['Potential'], label='[23, 28)', kde=True,hist=False)

sns.distplot(data[(data['Age']>=28) & (data['Age']<33)]['Potential'], label='[28, 33)', kde=True,hist=False)

c = sns.distplot(data[data['Age']>=33]['Potential'], label='Over 33', kde=True,hist=False)

plt.title('Potential by age (KDE)', fontsize=25)

c.set_xlabel('Potential', fontsize=20)

c.set_ylabel('Percentage', fontsize=20)

plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
plt.figure(figsize=(15,8))

sns.distplot(data[data['Age']<23]['Potential'], label='Under 23', kde=True, hist=False)

sns.distplot(data[(data['Age']>=23) & (data['Age']<28)]['Potential'], label='[23, 28)', kde=True,hist=False)

sns.distplot(data[(data['Age']>=28) & (data['Age']<33)]['Overall'], label='[28, 33)', kde=True,hist=False)

c = sns.distplot(data[data['Age']>=33]['Overall'], label='Over 33', kde=True,hist=False)

plt.title('Evolution of Overall stats by age', fontsize=25)

c.set_xlabel('Potential/Overall', fontsize=20)

c.set_ylabel('Percentage', fontsize=20)

plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
print('Number of players Nationalities in the dataset:')

print(data['Nationality'].nunique())

print('\n')

print('List of players Nationalities in the dataset:')

nationality = data['Nationality'].unique()

print(nationality)
plt.figure(figsize=(40,20))

b = sns.countplot(x = 'Nationality', data=data, order = data['Nationality'].value_counts().index)

b.axes.set_title("Distribution of Nationalities",fontsize=30)

b.set_xlabel("Nationality",fontsize=25)

b.set_ylabel("# of Players",fontsize=25)

b.set_xticklabels(b.get_xticklabels(), rotation=90, fontsize=12)

plt.show()
df = data

for nat in nationality:

    if ((data[data['Nationality'] == nat].count()[0])<=100):

        df = df[~df.Nationality.str.contains(nat)]



plt.figure(figsize=(20,12))

b = sns.countplot(x = 'Nationality', data=df, order = df['Nationality'].value_counts().index)

b.axes.set_title("Distribution of Nationalities with more than 100 players",fontsize=30)

b.set_xlabel("Nationality",fontsize=25)

b.set_ylabel("# of Players",fontsize=25)

b.set_xticklabels(b.get_xticklabels(), rotation=90, fontsize=16)

plt.show()
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

from plotly import tools

import plotly.graph_objs as go
def show_stat(data, xx):

    x = data[data['Name'] == xx]

    # define the average stats

    ball_skills = np.mean(np.array([x['BallControl'].values[0], x['Dribbling'].values[0]]))

    defense = np.mean(np.array([x['Marking'].values[0], x['SlidingTackle'].values[0], x['StandingTackle'].values[0]]))

    mental = np.mean(np.array([x['Aggression'].values[0], x['Reactions'].values[0], x['Positioning'].values[0], x['Interceptions'].values[0],x['Vision'].values[0]]))

    passing =  np.mean(np.array([x['Crossing'].values[0], x['ShortPassing'].values[0], x['LongPassing'].values[0]]))

    physical = np.mean(np.array([x['Acceleration'].values[0], x['Stamina'].values[0], x['Strength'].values[0], x['Balance'].values[0],x['SprintSpeed'].values[0],x['Agility'].values[0],x['Jumping'].values[0]]))

    shooting = np.mean(np.array([x['HeadingAccuracy'].values[0], x['ShotPower'].values[0], x['Finishing'].values[0], x['LongShots'].values[0],x['Curve'].values[0],x['FKAccuracy'].values[0],x['Penalties'].values[0], x['Volleys'].values[0]]))

    

    goalkeeper = np.mean(np.array([x['GKPositioning'].values[0],x['GKKicking'].values[0],x['GKHandling'].values[0],x['GKReflexes'].values[0],x['GKDiving'].values[0]]))

    

    if goalkeeper<30 :

        data = [go.Scatterpolar(

        r = [

            mental,

            ball_skills,

            passing, 

            physical, 

            shooting, 

            defense, 

            #goalkeeper,

            mental #has to be the same as the beginning

        ],

        theta = [

            'Mental', 'Ball Skills',  'Passing', 'Physical', 'Shooting', 'Defense', 'Mental'

        ],fill = 'toself')]

    else : 

        data = [go.Scatterpolar(

        r = [

            mental,

            ball_skills,

            passing, 

            physical, 

            #shooting, 

            defense, 

            goalkeeper,

            mental #has to be the same as the beginning

        ],

        theta = [

            'Mental', 'Ball Skills',  'Passing', 'Physical',  'Defense', 'Goalkeeper', 'Mental'

        ],fill = 'toself')]

    



    layout = go.Layout(polar = dict(

        radialaxis = dict(

            visible = True,

            range = [0, 99]

        )

    ),showlegend = False,

                       title = "Stats of {}".format(x.Name.values[0]))

    

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename = "Single Player stats")
show_stat(data, 'V. van Dijk')
show_stat(data, 'T. Courtois')
# Creating a method to compare 2 player

def compare_players(data, xx,yy):

    x = data[data['Name'] == xx]

    y = data[data['Name'] == yy]

    

    # define the average stats

    # player x

    ball_skills = np.mean(np.array([x['BallControl'].values[0], x['Dribbling'].values[0]]))

    defense = np.mean(np.array([x['Marking'].values[0], x['SlidingTackle'].values[0], x['StandingTackle'].values[0]]))

    mental = np.mean(np.array([x['Aggression'].values[0], x['Reactions'].values[0], x['Positioning'].values[0], x['Interceptions'].values[0],x['Vision'].values[0]]))

    passing =  np.mean(np.array([x['Crossing'].values[0], x['ShortPassing'].values[0], x['LongPassing'].values[0]]))

    physical = np.mean(np.array([x['Acceleration'].values[0], x['Stamina'].values[0], x['Strength'].values[0], x['Balance'].values[0],x['SprintSpeed'].values[0],x['Agility'].values[0],x['Jumping'].values[0]]))

    shooting = np.mean(np.array([x['HeadingAccuracy'].values[0], x['ShotPower'].values[0], x['Finishing'].values[0], x['LongShots'].values[0],x['Curve'].values[0],x['FKAccuracy'].values[0],x['Penalties'].values[0], x['Volleys'].values[0]]))

    goalkeeper = np.mean(np.array([x['GKPositioning'].values[0],x['GKKicking'].values[0],x['GKHandling'].values[0],x['GKReflexes'].values[0],x['GKDiving'].values[0]]))

    

    #player y

    # define the average stats

    y_ball_skills = np.mean(np.array([y['BallControl'].values[0], y['Dribbling'].values[0]]))

    y_defense = np.mean(np.array([y['Marking'].values[0], y['SlidingTackle'].values[0], y['StandingTackle'].values[0]]))

    y_mental = np.mean(np.array([y['Aggression'].values[0], y['Reactions'].values[0], y['Positioning'].values[0], y['Interceptions'].values[0],y['Vision'].values[0]]))

    y_passing =  np.mean(np.array([y['Crossing'].values[0], y['ShortPassing'].values[0], y['LongPassing'].values[0]]))

    y_physical = np.mean(np.array([y['Acceleration'].values[0], y['Stamina'].values[0], y['Strength'].values[0], y['Balance'].values[0],y['SprintSpeed'].values[0], y['Agility'].values[0], y['Jumping'].values[0]]))

    y_shooting = np.mean(np.array([y['HeadingAccuracy'].values[0], y['ShotPower'].values[0], y['Finishing'].values[0], y['LongShots'].values[0], y['Curve'].values[0], y['FKAccuracy'].values[0], y['Penalties'].values[0], y['Volleys'].values[0]]))

    y_goalkeeper = np.mean(np.array([y['GKPositioning'].values[0],y['GKKicking'].values[0], y['GKHandling'].values[0], y['GKReflexes'].values[0], y['GKDiving'].values[0]]))

    

    if ((goalkeeper <30) & (y_goalkeeper <30)):

        trace0 = go.Scatterpolar(r = [

            mental,ball_skills,passing, physical, shooting, defense, #goalkeeper,

                                      mental #has to be the same as the beginning

                                     ],

                                 theta = ['Mental', 'Ball Skills',  'Passing', 'Physical', 'Shooting', 'Defense',  'Mental'

                                         ],

                                 fill = 'toself', 

                                 name = x.Name.values[0]

                                )



        trace1 = go.Scatterpolar(r = [

            y_mental, y_ball_skills, y_passing, y_physical, y_shooting, y_defense, #y_goalkeeper,

                                      y_mental #has to be the same as the beginning

                                     ],

                                 theta = ['Mental', 'Ball Skills',  'Passing', 'Physical', 'Shooting', 'Defense',  'Mental'

                                         ],

                                 fill = 'toself', 

                                 name = y.Name.values[0]

                                )

    elif ((goalkeeper >=30) & (y_goalkeeper >=30)): 

        trace0 = go.Scatterpolar(

            r = [

            mental,

            ball_skills,

            passing, 

            physical, 

            #shooting, 

            defense, 

            goalkeeper,

            mental #has to be the same as the beginning

        ],

        theta = [

            'Mental', 'Ball Skills',  'Passing', 'Physical',  'Defense', 'Goalkeeper', 'Mental'

        ],

        fill = 'toself', 

        name = x.Name.values[0]

    )



        trace1 = go.Scatterpolar(

        r = [

            y_mental,

            y_ball_skills,

            y_passing, 

            y_physical, 

            #y_shooting, 

            y_defense, 

            y_goalkeeper,

            y_mental #has to be the same as the beginning

        ],

        theta = [

            'Mental', 'Ball Skills',  'Passing', 'Physical',  'Defense', 'Goalkeeper', 'Mental'

        ],

        fill = 'toself',

        name = y.Name.values[0]

    )

    else : 

        trace0 = go.Scatterpolar(

        r = [

            mental,

            ball_skills,

            passing, 

            physical, 

            shooting, 

            defense, 

            goalkeeper,

            mental #has to be the same as the beginning

        ],

        theta = [

            'Mental', 'Ball Skills',  'Passing', 'Physical', 'Shooting', 'Defense', 'Goalkeeper', 'Mental'

        ],

        fill = 'toself', 

        name = x.Name.values[0]

    )



        trace1 = go.Scatterpolar(

        r = [

            y_mental,

            y_ball_skills,

            y_passing, 

            y_physical, 

            y_shooting, 

            y_defense, 

            y_goalkeeper,

            y_mental #has to be the same as the beginning

        ],

        theta = [

            'Mental', 'Ball Skills',  'Passing', 'Physical', 'Shooting', 'Defense', 'Goalkeeper', 'Mental'

        ],

        fill = 'toself',

        name = y.Name.values[0]

    )

            

    



    data = [trace0, trace1]



    layout = go.Layout(

      polar = dict(

        radialaxis = dict(

          visible = True,

          range = [0, 99]

        )

      ),

      showlegend = True,

      title = "{} vs {}".format(x.Name.values[0],y.Name.values[0])

    )

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename = "Two Player stats")
compare_players(data, 'Sergio Ramos', 'Cristiano Ronaldo')
compare_players(data, 'T. Courtois', 'B. Drągowski')
compare_players(data, 'T. Courtois', 'Cristiano Ronaldo')
data_ita = data[data['Nationality']=='Italy']

data_ita.head(5)
plt.figure(figsize=(15,8))

sns.distplot(data_ita[data_ita['Age']<23]['Potential'], label='Under 23', kde=True, hist=False)

sns.distplot(data_ita[(data_ita['Age']>=23) & (data_ita['Age']<28)]['Potential'], label='[23, 28)', kde=True,hist=False)

sns.distplot(data_ita[(data_ita['Age']>=28) & (data_ita['Age']<33)]['Overall'], label='[28, 33)', kde=True,hist=False)

c = sns.distplot(data_ita[data_ita['Age']>=33]['Overall'], label='Over 33', kde=True,hist=False)

plt.title('Evolution of Overall stats of Italian players by age', fontsize=25)

c.set_xlabel('Potential/Overall', fontsize=20)

c.set_ylabel('Percentage', fontsize=20)

plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
clubs_serieA = ['Juventus', 'Napoli', 'Atalanta', 'Inter', 'Milan', 'Roma',  'Torino', 

                'Lazio','Sampdoria', 'Sassuolo',   'Genoa', 'Chievo Verona',

                'Fiorentina', 'SPAL', 'Frosinone', 'Cagliari','Bologna', 'Parma', 'Udinese','Empoli']



clubs_serieB = ['Brescia', 'Benevento', 'Livorno', 'Foggia',  'Hellas Verona', 

              'Palermo',  'Lecce','US Salernitana 1919', 'Crotone', 

              'Spezia', 'Cittadella','Perugia', 'Carpi',  'Pescara',

              'Venezia FC', 'Cosenza', 'Reading', 'Ascoli','Padova']



# Serie A

plt.figure(figsize=(15,8))

for club in clubs_serieA:

    c = sns.distplot(data_ita[data_ita['Club']==club]['Overall'], label=club, kde=True,hist=False)



    

plt.title('Dustribution of Overall by Serie A clubs', fontsize=25)

c.set_xlabel('Overall', fontsize=20)

c.set_ylabel('Percentage', fontsize=20)    

plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()





# Serie B

plt.figure(figsize=(15,8))

for club in clubs_serieB:

    c = sns.distplot(data_ita[data_ita['Club']==club]['Overall'], label=club, kde=True,hist=False)



    

plt.title('Dustribution of Overall by Serie B clubs', fontsize=25)

c.set_xlabel('Overall', fontsize=20)

c.set_ylabel('Percentage', fontsize=20)    

plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
clubs_top4 = ['Juventus', 'Napoli', 'Atalanta', 'Inter', 

              'FC Barcelona', 'Atlético Madrid' , 'Real Madrid', 'Valencia CF',

             'Manchester City', 'Liverpool', 'Chelsea', 'Tottenham',

             'FC Bayern München', 'Borussia Dortmund', 'RB Leipzig', 'Bayer 04 Leverkusen',

             'Paris Saint-Germain', 'LOSC Lille', 'Olympique Lyonnais', 'AS Saint-Étienne']



plt.figure(figsize=(15,8))

for club in clubs_top4:

    c = sns.distplot(data[data['Club']==club]['Overall'], label=club, kde=True,hist=False)



    

plt.title('Dustribution of Overall by Top 4 Teams of Top 5 European National League', fontsize=25)

c.set_xlabel('Overall', fontsize=20)

c.set_ylabel('Percentage', fontsize=20)    

plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()