import os

from kaggle.competitions import nflrush

import numpy as np

import pandas as pd

from sklearn import preprocessing

import matplotlib.pyplot as plt

import matplotlib.patches as patches

from sklearn.model_selection import KFold

import lightgbm as lgb

import gc

import tqdm



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
env = nflrush.make_env()

train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)



def preprocessing(df):

    Poss = list(df.PossessionTeam.unique())

    HT = list(df.HomeTeamAbbr.unique())

    VT = list(df.VisitorTeamAbbr.unique())

    ['HST', 'BLT', 'ARZ', 'CLV'] # poss

    ['BAL', 'HOU', 'ARI', 'CLE'] # home/away 

    abbrev = {'HOU':'HST','BAL':'BLT','ARI':'ARZ','CLE':'CLV'}



    df = df.replace({"HomeTeamAbbr": abbrev})

    df = df.replace({"VisitorTeamAbbr": abbrev})

    

    df['Team'] = np.where(df['Team'].copy()=='away',df['VisitorTeamAbbr'],df['HomeTeamAbbr'])

    df['Offense'] = np.where(df['Team']==df['PossessionTeam'],1,0)

    return df
def flip_left(df):

    

    # standardize yard line

    df['Opp'] = np.where(df['PossessionTeam']==df['FieldPosition'],0,1)

    df['OppYardLine'] =  50-df['YardLine'] + 50 

    df = df.rename(columns={'YardLine':'WrongYardLine'})

    df['YardLine'] = np.where(df['Opp']==1, df['OppYardLine'], df['WrongYardLine'])

    

    # offensive players should always be moving from 0 to 100

    df['AltLeftX'] = 120 - df['X'] - 10

    df['AltLeftY'] = 160/3 - df['Y']

    df['AltRightX'] = df['X'] - 10

    

    df = df.rename(columns={'X':'WrongX','Y':'WrongY'})

    

    df['ToLeft'] = np.where(df['PlayDirection']=='left',1,0)



    df['X'] = np.where(df['ToLeft']==1, df['AltLeftX'], df['AltRightX'])

    df['Y'] = np.where(df['ToLeft']==1, df['AltLeftY'], df['WrongY'])



    # change orientation if flipping field

    df['Dir'] = np.radians(df['Dir'].copy())

    df['Or'] = np.radians(df['Orientation'].copy())

    

    

#  taking this step from Michael Lopez's R kernel. I don't understand why it's necessary, but let's be safe

#   mutate(Dir_std_1 = ifelse(ToLeft & Dir < 90, Dir + 360, Dir), 

#          Dir_std_1 = ifelse(!ToLeft & Dir > 270, Dir - 360, Dir_std_1))



    

    

    df['Dir1'] = np.where(((df.ToLeft==1)&(df['Dir'] < (np.pi/2))), df['Dir']+(2*np.pi), df['Dir'])

    df['Dir1'] = np.where(((df.ToLeft==0)&(df['Dir'] > ((3*np.pi)/2))), df['Dir']-(2*np.pi), df['Dir1'])

    

    df['Or1'] = np.where(((df.ToLeft==1)&(df['Or'] < (np.pi/2))), df['Or']+(2*np.pi), df['Or'])

    df['Or1'] = np.where(((df.ToLeft==0)&(df['Or'] > ((3*np.pi)/2))), df['Or']-(2*np.pi), df['Or1'])

    

    # should be (3/2)*pi, pi/2

    print(df.loc[df.ToLeft==1].Dir1.median())

    print(df.loc[df.ToLeft==0].Dir1.median())

    

    # this part I understand

#     mutate(Dir_std_2 = ifelse(ToLeft, Dir_std_1 - 180, Dir_std_1))

    df['Dir'] = np.where((df['ToLeft']==1),df['Dir1']-np.pi,df['Dir1'])

    df['Or'] = np.where((df['ToLeft']==1),df['Or1']-np.pi,df['Or1'])

    

    # should be pi/2,pi/2

    print(df.loc[df.ToLeft==1].Dir.median())

    print(df.loc[df.ToLeft==0].Dir.median())

    

    # this part I don't understand either

    # i guess original direction is relative to the y axis? Not relative to team direction?

    df['Dir'] = (np.pi/2)-df['Dir'].copy()

    df['Or'] = (np.pi/2)-df['Or'].copy()

    

    # i think pad orientation is off by 90 degrees

    df['Or'] = df['Or'].copy() - (np.pi/2)

    

    df = df.drop(columns=['Dir1'])

    df = df.drop(columns=['Or1','Orientation','AltLeftX','AltLeftY','AltRightX','WrongX','WrongY'])

    

    return df 

train = preprocessing(train)

train = flip_left(train)

def fe(df):

    

    df['IsBc'] = np.where(df['NflIdRusher']==df['NflId'],1,0)

    df = df.sort_values(by=['PlayId', 'Team', 'IsBc']).reset_index()

    

    # future point

    df['U'] = df['S'] * 1.75 * np.cos(df['Dir'])+ df['X']

    df['W'] = df['S'] * 1.75 * np.sin(df['Dir']) + df['Y']

    

    # orientation vector

    df['Uo'] = df['S'] * 1.25 * np.cos(df['Or'])+ df['X']

    df['Wo'] = df['S'] * 1.25 * np.sin(df['Or']) + df['Y']

    

    

    

    return df
train = fe(train)
#https://www.kaggle.com/robikscube/nfl-big-data-bowl-plotting-player-position

def create_football_field(linenumbers=True,

                          endzones=True,

                          highlight_line=False,

                          highlight_line_number=50,

                          highlighted_name='Line of Scrimmage',

                          fifty_is_los=False,

                          figsize=(12, 6.33)):

    """

    Function that plots the football field for viewing plays.

    Allows for showing or hiding endzones.

    """

    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,

                             edgecolor='r', facecolor='lightgreen', zorder=0)



    fig, ax = plt.subplots(1, figsize=figsize)

    ax.add_patch(rect)



    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,

              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],

             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,

              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],

             color='white')

    if fifty_is_los:

        plt.plot([60, 60], [0, 53.3], color='gold')

        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')

    # Endzones

    if endzones:

        ez1 = patches.Rectangle((0, 0), 10, 53.3,

                                linewidth=0.1,

                                edgecolor='r',

                                facecolor='blue',

                                alpha=0.2,

                                zorder=0)

        ez2 = patches.Rectangle((110, 0), 120, 53.3,

                                linewidth=0.1,

                                edgecolor='r',

                                facecolor='blue',

                                alpha=0.2,

                                zorder=0)

        ax.add_patch(ez1)

        ax.add_patch(ez2)

    plt.xlim(0, 120)

    plt.ylim(-5, 58.3)

    plt.axis('off')

    if linenumbers:

        for x in range(20, 110, 10):

            numb = x

            if x > 50:

                numb = 120 - x

            plt.text(x, 5, str(numb - 10),

                     horizontalalignment='center',

                     fontsize=20,  # fontname='Arial',

                     color='white')

            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),

                     horizontalalignment='center',

                     fontsize=20,  # fontname='Arial',

                     color='white', rotation=180)

    if endzones:

        hash_range = range(11, 110)

    else:

        hash_range = range(1, 120)



    for x in hash_range:

        ax.plot([x, x], [0.4, 0.7], color='white')

        ax.plot([x, x], [53.0, 52.5], color='white')

        ax.plot([x, x], [22.91, 23.57], color='white')

        ax.plot([x, x], [29.73, 30.39], color='white')



    if highlight_line:

        hl = highlight_line_number + 10

        plt.plot([hl, hl], [0, 53.3], color='yellow')

        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),

                 color='yellow')

    return fig, ax
import matplotlib.pyplot as plt

import matplotlib.patches as patches

import matplotlib.path as mpath



def create_football_field(linenumbers=True,

                          endzones=True,

                          los_line=False,

                          los_line_number=50,

                          highlight_line=False,

                          highlight_line_number=50,

                          highlighted_name='Line of Scrimmage',

                          fifty_is_los=False,

                          figsize=(15, 7)):

    """

    Function that plots the football field for viewing plays.

    Allows for showing or hiding endzones.

    """

    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,

                             edgecolor='r', facecolor='darkgreen', zorder=0)



    fig, ax = plt.subplots(1, figsize=figsize)

    ax.add_patch(rect)



    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,

              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],

             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,

              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],

             color='white')

    if fifty_is_los:

        plt.plot([60, 60], [0, 53.3], color='gold')

        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')

    # Endzones

    if endzones:

        ez1 = patches.Rectangle((0, 0), 10, 53.3,

                                linewidth=0.1,

                                edgecolor='r',

                                facecolor='blue',

                                alpha=0.2,

                                zorder=0)

        ez2 = patches.Rectangle((110, 0), 120, 53.3,

                                linewidth=0.1,

                                edgecolor='r',

                                facecolor='darkblue',

                                alpha=0.2,

                                zorder=0)

        ax.add_patch(ez1)

        ax.add_patch(ez2)

    plt.xlim(0, 120)

    plt.ylim(-5, 58.3)

    plt.axis('off')

    if linenumbers:

        for x in range(20, 110, 10):

            numb = x

            if x > 50:

                numb = 120 - x

#             plt.text(x, 5, str(numb - 10),

#                      horizontalalignment='center',

#                      fontsize=20,  # fontname='Arial',

#                      color='white')

#             plt.text(x - 0.95, 53.3 - 5, str(numb - 10),

#                      horizontalalignment='center',

#                      fontsize=20,  # fontname='Arial',

#                      color='white', rotation=180)

    if endzones:

        hash_range = range(11, 110)

    else:

        hash_range = range(1, 120)



    for x in hash_range:

        ax.plot([x, x], [0.4, 0.7], color='white')

        ax.plot([x, x], [53.0, 52.5], color='white')

#         ax.plot([x, x], [22.91, 23.57], color='white')

#         ax.plot([x, x], [29.73, 30.39], color='white')



    if los_line:

        hl = los_line_number + 10

        plt.plot([hl, hl], [0, 53.3], color='orange')

    

    if highlight_line:

        hl = highlight_line_number + 10

        plt.plot([hl, hl], [0, 53.3], color='yellow')

#         plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),

#                  color='yellow')

    return fig, ax





import random



plays = list(train.PlayId.values)

play = random.sample(plays,1)[0]



# Michael Lopez's sample plays in his kernel with gif of play 

# play = 20170910000081

play = 20170910001102

play_df = train.loc[train.PlayId==play]



# line of scrimmage

los = play_df.YardLine.mode()[0]

distance = play_df.Distance.mode()[0]

first_down = los + distance

fig, ax = create_football_field(highlight_line=True,

                      highlight_line_number=first_down,

                      los_line=True,

                      los_line_number=los

                     )

bc = play_df.loc[play_df.IsBc==1]

offense = play_df.loc[(play_df.Offense==1)&(play_df.IsBc==0)]

defense = play_df.loc[(play_df.Offense==0)]

        



for index, row in offense.iterrows():

    ax.arrow(row['X'], row['Y'], (row['Uo']-row['X']), (row['Wo']-row['Y']), head_width=0.2, head_length=0.7, ec='lightblue')

    ax.arrow(row['X'], row['Y'], (row['U']-row['X']), (row['W']-row['Y']), head_width=0.2, head_length=0.7, ec='blue')

#     ax.arrow(row['X'], row['Y'], row['U2'], row['W2'], head_width=0.2, head_length=0.7, ec='lightblue')

# plt.scatter(offense.X,offense.Y, s=35, color='blue', zorder=5)



for index, row in defense.iterrows():

    ax.arrow(row['X'], row['Y'], (row['Uo']-row['X']), (row['Wo']-row['Y']), head_width=0.2, head_length=0.7, ec='pink')

    ax.arrow(row['X'], row['Y'], (row['U']-row['X']), (row['W']-row['Y']), head_width=0.2, head_length=0.7, ec='red')

#     ax.arrow(row['X'], row['Y'], row['U2'], row['W2'], head_width=0.2, head_length=0.7, ec='pink')

plt.scatter(defense.X,defense.Y, s=35, color='red', zorder=5)



# plt.scatter(bc.X.values[0],bc.Y.values[0],s=35,color='black')

# # ball carrier

ax.arrow(bc.X.values[0], bc.Y.values[0], (bc.U.values[0]-bc.X.values[0]), (bc.W.values[0]-bc.Y.values[0]), head_width=0.2, head_length=0.7, ec='black')

ax.arrow(bc.X.values[0], bc.Y.values[0], (bc.Uo.values[0]-bc.X.values[0]), (bc.Wo.values[0]-bc.Y.values[0]), head_width=0.2, head_length=0.7, ec='white')

# ax.arrow(bc.X.values[0], bc.Y.values[0], bc.U2.values[0], bc.W2.values[0], head_width=0.2, head_length=0.7, ec='black')



plt.title('Speed/Dir/Orientation Vector Plot')

plt.legend(loc=4)

plt.show()