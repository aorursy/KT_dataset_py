import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

% matplotlib inline



import sqlite3

import datetime

from IPython.display import display



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
with sqlite3.connect('../input/database.sqlite') as con:

    Matches = pd.read_sql_query("SELECT * FROM Match", con)

    Teams = pd.read_sql_query("SELECT * FROM Team", con)

    Players_ratings = pd.read_sql_query("SELECT * FROM Player Pl, Player_Attributes At WHERE At.date = '2015-09-21 00:00:00' AND Pl.player_api_id = At.player_api_id", con)
# Player_api_id should be keep for coding, others id is useless so we just delete it.

# reference code: df = df.loc[:,~df.columns.duplicated()]

Players_ratings = Players_ratings.iloc[:,~Players_ratings.columns.duplicated()]

del Players_ratings['id']#, Players_ratings['player_fifa_api_id']

Players_ratings.head(3)
number = range(1,12)

P_h = 'home_player_'; P_a = 'away_player_'

X_h = 'home_player_X'; X_a = 'away_player_X'

Y_h = 'home_player_Y'; Y_a = 'away_player_Y'

team_h = 'home_team_api_id'; team_a = 'away_team_api_id'

date = 'date'

Infos = []

for i in number:

    p_h = P_h + str(i); p_a = P_a + str(i)

    x_h = X_h + str(i); x_a = X_a + str(i)

    y_h = Y_h + str(i); y_a = Y_a + str(i)



    df_h = [p_h, x_h, y_h, team_h, date]

    df_a = [p_a, x_a, y_a, team_a, date]

    M_h = Matches[df_h].as_matrix()

    M_a = Matches[df_a].as_matrix()

    Info = np.concatenate((M_h,M_a),axis = 0)

    Infos.append(Info)

# number of rows in the records of "player11":

# print(Info.shape)
Info_0 = np.array([[np.nan,np.nan,np.nan,np.nan,np.nan]])

for Info in Infos:

    Info_0 = np.concatenate((Info_0,Info), axis= 0)

player_Info = Info_0

print(player_Info.shape)



col = ['Player_id','X','Y','Team','Date']

Player_Info = pd.DataFrame(player_Info, columns = col)

Player_Info = Player_Info.dropna()

print(Player_Info.shape)

Player_Info = pd.DataFrame.drop_duplicates(Player_Info)

print(Player_Info.shape)



Player_Info = Player_Info.sort_values([col[0],col[4]])

Player_Info.head(3)
# Get the name of teams to validate data

col_team = ['team_api_id', 'team_long_name', 'team_short_name']

col_player = ['player_api_id', 'player_name']

Team_info = Teams[col_team]

Player_physic = Players_ratings[col_player]

test = Player_Info.merge(Team_info, left_on = 'Team', right_on= 'team_api_id', how = 'left')

#test = test.merge(Player_physic, left_on= 'Player_id', right_on= 'player_api_id', how = 'left')

# Get players name

test = test.merge(Player_physic, left_on = 'Player_id', right_on = 'player_api_id', how = 'left')

del test['player_api_id']

display(test.head())

print('Number (rows) of records:',str(test.shape[0]))

print("Number of players in table 'Players':", Players_ratings.shape[0])

print("Number of players in table 'records':", test['Player_id'].value_counts().shape[0])
#datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

test['Date'] = pd.to_datetime(test['Date'], format = '%Y-%m-%d %H:%M:%S')



#select recording date intervals according to what we need

mask = (test['Date'] > '2013-1-1') & (test['Date'] <= '2015-9-21')

test = test.loc[mask]

#drop players that we cannot get their names in the data update name list

test = test.dropna()

Recordings = pd.DataFrame.drop_duplicates(test)

#test.sort_values('Player_id','Date')

Recordings.head()#datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

test['Date'] = pd.to_datetime(test['Date'], format = '%Y-%m-%d %H:%M:%S')
plt.scatter(Recordings['X'], Recordings['Y'])

plt.show()
import scipy.stats as stats



#Player class, for each player id, record player's lineup records.

class Player(object):

    def __init__(self, id=np.nan):

        self.player_id = id

        self.player_name = ''

        self.X = []

        self.Y = []

        self.position = []

        self.team_id = []

        self.team_name = []

        self.num = 0 # number of lineup records

    

    def set_position(self):

        mode_y = stats.mode(self.Y).mode

        if (mode_y > 8):

            self.position = 'F' # forward player

        elif (mode_y >= 5):

            self.position = 'M' # middlefield player

        elif (mode_y > 1):

            self.position = 'D' # defender

        elif (mode_y == 1):

            self.position = 'GK' # goalkeeper

        else:

            self.position = None

            

    def set_info(self, pd_series):

        self.player_id = np.array(pd_series["Player_id"])

        self.player_name = np.array(pd_series["player_name"])

        

        self.X.append(np.array(pd_series["X"]))

        self.Y.append(np.array(pd_series["Y"]))

        self.team_id.append(np.array(pd_series["Team"]))

        self.team_name.append(np.array(pd_series["team_long_name"]))

        

        self.num += 1

    

    def set_records(self, pd_series):

        #col = ['Player_id', 'X', 'Y', 'Team']

        self.X.append(np.array(pd_series["X"]))

        self.Y.append(np.array(pd_series["Y"]))

        self.team_id.append(np.array(pd_series["Team"]))

        self.team_name.append(np.array(pd_series["team_long_name"]))

        

        self.num += 1

    

    def get_info(self):

        self.set_position()

        x = stats.mode(self.X).mode

        y = stats.mode(self.Y).mode

        t_id = self.team_id[-1] # get the most recent team each player plays for

        t_name = self.team_name[-1]

        result = np.vstack((self.player_id, self.player_name, x, y, t_id, t_name, self.position))

        result = result.flatten()

        return result
# In this cell, I applied a cursor logic.

Players = [] # a list of Player class objects.

Positions = [] # write-up result of each players' normal position and their 'mode team'



num_r, num_c = Recordings.shape

k = 0

for index, row in Recordings.iterrows():

    if len(Players) == 0:

        b = Player(row['Player_id'])

        b.set_info(row)

        Players.append(b)

    elif Players[-1].player_id == row['Player_id']:

        Players[-1].set_records(row)

    else:

        #we finished data collection of the last player, it is time to get result from class.

        c = Players[-1].get_info()

        Positions.append(c)

        #set a new object for this player

        b = Player(row['Player_id'])

        b.set_info(row)

        Players.append(b)

    k += 1

    if k == num_r:

        c = Players[-1].get_info()

        Positions.append(c)



#print(len(Players))

#print(len(Positions))

#two values should be equals
P = Positions[0]

P.reshape(1,-1)

print(P.shape)

for i in Positions[1:]:

    P = np.vstack((P,i))



print(P)

print(P.shape)
col = ['Player_id', 'Name', 'X_mod', 'Y_mod', 'Team_recent', 'Team_name', 'Position_mod']



Summary = pd.DataFrame(P, columns = col)

Summary[[col[2],col[3]]].apply(pd.to_numeric)

display(Summary.head(10))

print(Summary.shape)
a = Summary["Team_name"] == 'FC Barcelona'

Barca = Summary.loc[a]

Barca = Barca.sort_values(['Position_mod','Name'])



display(Barca)
col_ratings = Players_ratings.columns

print(list(col_ratings))
#change data type of dataframe

Players_ratings['player_api_id'] = Players_ratings['player_api_id'].astype('str')

Summary['Player_id'] = pd.to_numeric(Summary['Player_id'])

Summary['Player_id'] = Summary['Player_id'].astype(np.int64).astype('str')



Summary1 = Summary.merge(Players_ratings, how = 'left', left_on = 'Player_id', right_on = 'player_api_id')

del Summary1['player_api_id'], Summary1['player_name'], Summary1['player_fifa_api_id']

display(Summary1.head())
forward = Summary1["Position_mod"] == 'F'

Forward = Summary1.loc[forward]

pd.to_numeric(Forward['overall_rating'])

Forward = Forward.sort_values(by =['overall_rating'], ascending = False)

Forward.head(10)
mid = Summary1["Position_mod"] == 'M'

Midfield = Summary1.loc[mid]

pd.to_numeric(Midfield['overall_rating'])

Midfield = Midfield.sort_values(by =['overall_rating'], ascending = False)

Midfield.head(10)
goalie = Summary1["Position_mod"] == 'GK'

Goalie = Summary1.loc[goalie]

pd.to_numeric(Goalie['overall_rating'])

Goalie = Goalie.sort_values(by =['overall_rating'], ascending = False)

Goalie.head(10)
import numpy as np

import pylab as pl



class Radar(object):



    def __init__(self, fig, titles, labels, rect=None):

        if rect is None:

            rect = [0.05, 0.05, 0.95, 0.95]



        self.n = len(titles)

        self.angles = np.arange(90, 90+360, 360.0/self.n)

        self.axes = [fig.add_axes(rect, projection="polar", label="axes%d" % i) 

                         for i in range(self.n)]



        self.ax = self.axes[0]

        self.ax.set_thetagrids(self.angles, labels=titles, fontsize=14)



        for ax in self.axes[1:]:

            ax.patch.set_visible(False)

            ax.grid("off")

            ax.xaxis.set_visible(False)



        for ax, angle, label in zip(self.axes, self.angles, labels):

            ax.set_rgrids(range(1, 101), angle=angle, labels=label)

            ax.spines["polar"].set_visible(False)

            ax.set_ylim(0, 100)



    def plot(self, values, *args, **kw):

        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])

        values = np.r_[values, values[0]]

        self.ax.plot(angle, values, *args, **kw)
Attributes = ['crossing','finishing','heading_accuracy','short_passing','volleys','dribbling','curve','free_kick_accuracy','long_passing','ball_control','acceleration','sprint_speed','agility','reactions','balance','shot_power','jumping','stamina','strength','long_shots','aggression','interceptions','positioning','vision','penalties','marking','standing_tackle','sliding_tackle','gk_diving','gk_handling','gk_kicking','gk_positioning','gk_reflexes']



Labels = []

for i in range(len(Attributes)):

    Labels.append([])

Goalie1 = Goalie.iloc[:10]



fig = pl.figure(figsize=(6, 6))



titles = Attributes

labels = Labels

radar = Radar(fig, titles, labels)



for index, player in Goalie1.iterrows():

    radar.plot(player[Attributes], "-", lw=2, alpha=0.4, label=player['Name'])

    radar.ax.legend()



plt.show()
Midfield1 = Midfield.iloc[:10]



fig = pl.figure(figsize=(6, 6))



titles = Attributes

labels = Labels

radar = Radar(fig, titles, labels)



for index, player in Midfield1.iterrows():

    radar.plot(player[Attributes], "-", lw=2, alpha=0.4, label=player['Name'])

    radar.ax.legend()



plt.show()
Forward1 = Forward.iloc[:10]



fig = pl.figure(figsize=(6, 6))



titles = Attributes

labels = Labels

radar = Radar(fig, titles, labels)



for index, player in Forward1.iterrows():

    radar.plot(player[Attributes], "-", lw=2, alpha=0.4, label=player['Name'])

    radar.ax.legend()



plt.show()