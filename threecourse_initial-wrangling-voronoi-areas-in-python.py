import pandas as pd

import numpy as np

pd.options.display.max_columns = 100

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import seaborn as sns

import datetime, tqdm

# from kaggle.competitions import nflrush

import math

from ipywidgets import interact
train = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", low_memory=False)



# preprocess

train['ToLeft'] = train.PlayDirection == "left"

train['IsBallCarrier'] = train.NflId == train.NflIdRusher



train.loc[train.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"

train.loc[train.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

train.loc[train.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"

train.loc[train.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

train.loc[train.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"

train.loc[train.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

train.loc[train.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"

train.loc[train.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"



# dir => rad

train['Dir_rad'] = np.mod(90 - train.Dir, 360) * math.pi/180.0



# preprocess - standarized direction

train['TeamOnOffense'] = "home"

train.loc[train.PossessionTeam != train.HomeTeamAbbr, 'TeamOnOffense'] = "away"

train['IsOnOffense'] = train.Team == train.TeamOnOffense # Is player on offense?

train['YardLine_std'] = 100 - train.YardLine

train.loc[train.FieldPosition.fillna('') == train.PossessionTeam, 'YardLine_std'] = (

    train.loc[train.FieldPosition.fillna('') == train.PossessionTeam, 'YardLine'])

train['X_std'] = train.X

train.loc[train.ToLeft, 'X_std'] = 120 - train.loc[train.ToLeft, 'X'] 

train['Y_std'] = train.Y

train.loc[train.ToLeft, 'Y_std'] = 160/3 - train.loc[train.ToLeft, 'Y'] 

#train['Orientation_std'] = -90 + train.Orientation

#train.loc[train.ToLeft, 'Orientation_std'] = np.mod(180 + train.loc[train.ToLeft, 'Orientation_std'], 360)

train['Dir_std'] = train.Dir_rad

train.loc[train.ToLeft, 'Dir_std'] = np.mod(np.pi + train.loc[train.ToLeft, 'Dir_rad'], 2*np.pi)

def create_football_field(linenumbers=True,

                          endzones=True,

                          highlight_line=False,

                          highlight_line_number=50,

                          highlighted_name='Line of Scrimmage',

                          highlight_line2=False,

                          highlight_line_number2=50,

                          fifty_is_los=False,

                          figsize=(12*2, 6.33*2)):

    """

    Function that plots the football field for viewing plays.

    Allows for showing or hiding endzones.

    """

    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,

                             edgecolor='r', facecolor='darkgreen', zorder=0,  alpha=0.5)



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

    if highlight_line2:

        hl = highlight_line_number2 + 10

        plt.plot([hl, hl], [0, 53.3], color='red')

    return fig, ax



def get_dx_dy(radian_angle, dist):

    dx = dist * math.cos(radian_angle)

    dy = dist * math.sin(radian_angle)

    return dx, dy



def show_play_std(play_id, train=train):

    """standarized direction display"""

    df = train[train.PlayId == play_id]

    YardLine = df.YardLine_std.values[0]

    rusher_row = df[df.NflIdRusher == df.NflId]

    yards_covered = rusher_row["Yards"].values[0]

    

    fig, ax = create_football_field(highlight_line=True,

                                    highlight_line_number=YardLine,

                                    highlight_line2=True,

                                    highlight_line_number2=YardLine + yards_covered,

                                   )

    ax.scatter(df.X_std, df.Y_std, cmap='rainbow', c=~df.IsOnOffense, s=100)

    ax.scatter(rusher_row.X_std, rusher_row.Y_std, color='black', s=100)



    for (x, y, Dir, S) in zip(df.X_std, df.Y_std, df.Dir_std, df.S):       

        dx, dy = get_dx_dy(Dir, S)

        ax.arrow(x, y, dx, dy, length_includes_head=False, width=0.2, color='#333333', alpha=0.7)

    left = 'left' if df.ToLeft.sum() > 0 else 'right'

    

    r = df.iloc[0, :]

    msg11 = f'game#:{r["GameId"]}' 

    msg12 = f'play#:{r["PlayId"]} moving to {left}' 

    msg2 = f'{r["HomeTeamAbbr"]}{r["HomeScoreBeforePlay"]}-{r["VisitorTeamAbbr"]}{r["VisitorScoreBeforePlay"]} possesion:{r["PossessionTeam"]}'

    msg3 = f'{r["Quarter"]}Q-{r["GameClock"]} {r["Down"]}&{r["Distance"]} scrimage is {r["YardLine_std"]}'

    msg4 = f' yard distance is {r["Yards"]}'

    title = f"{msg11}\n{msg12}\n{msg2}\n{msg3}\n{msg4}"

    

    # plt.title(f'Play # {play_id} moving to {left}, yard distance is {yards_covered}, scrimage is {YardLine} ', fontsize=20)

    plt.title(title, fontsize=20)

    plt.legend()

    plt.show()
play_ids = train["PlayId"].unique()

@interact(n=(0, len(play_ids)-1))

def plot_rolling_mean(n):

     show_play_std(play_ids[n])
# only gained yards >= 10



play_ids_big_gain = train[train["Yards"] >= 10]["PlayId"].unique()

@interact(n=(0, len(play_ids_big_gain)-1))

def plot_rolling_mean(n):

     show_play_std(play_ids_big_gain[n])
# only gained yards <= 0



play_ids_no_gain = train[train["Yards"] <= 0]["PlayId"].unique()

@interact(n=(0, len(play_ids_no_gain)-1))

def plot_rolling_mean(n):

     show_play_std(play_ids_no_gain[n])