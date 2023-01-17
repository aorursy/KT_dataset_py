import os

import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

from IPython.display import HTML

from matplotlib import animation



data_dir = '../input/nfl-big-data-bowl-2020/'

df = pd.read_csv(data_dir + 'train.csv', low_memory=False)

drop_cols = ['Team', 'PlayerHeight', 'PlayerWeight', 'PlayerBirthDate', 'PlayerCollegeName',

           'Position', 'Stadium', 'Location', 'StadiumType', 'Turf', 'GameWeather',

           'Temperature', 'Humidity', 'WindSpeed', 'WindDirection']

df = df.drop(drop_cols, axis=1)



games = np.unique(df['GameId'].values)

plays = np.unique(df['PlayId'].values)



def update(i, df_game, play_list):

    plt.cla()

    df_play = df_game[df_game['PlayId']==play_list[i]]



    # player info

    acc        = df_play['A'].values

    x_coord    = df_play['X'].values

    y_coord    = df_play['Y'].values

    direction  = df_play['Dir'].values

    nfl_id     = df_play['NflId'].values



    # play info

    game_id         = df_play['GameId'].values[0]

    play_id         = df_play['PlayId'].values[0]

    yards           = df_play['Yards'].values[0]

    yard_line       = df_play['YardLine'].values[0]

    quarter         = df_play['Quarter'].values[0]

    down            = df_play['Down'].values[0]

    distance        = df_play['Distance'].values[0]

    home_score      = df_play['HomeScoreBeforePlay'].values[0]

    visitor_score   = df_play['VisitorScoreBeforePlay'].values[0]

    rusher_id       = df_play['NflIdRusher'].values[0]

    play_dir        = df_play['PlayDirection'].values[0]

    field_position  = df_play['FieldPosition'].values[0]

    possession_team = df_play['PossessionTeam'].values[0]

    game_clock      = df_play['GameClock'].values[0]

    season          = df_play['Season'].values[0]

    week            = df_play['Week'].values[0]

    home_team       = df_play['HomeTeamAbbr'].values[0]

    visitor_team    = df_play['VisitorTeamAbbr'].values[0]

    

    a_x =  np.cos(direction * ((2 * np.pi)/360 + (np.pi/2)))

    a_y =  np.sin(direction * ((2 * np.pi)/360 + (np.pi/2)))



    rusher_idx = np.where(rusher_id == nfl_id)[0][0]

    

    plt.ylim((0, 53))

    plt.xlim((0, 120))

    norm = matplotlib.colors.Normalize(vmin=acc.min(),vmax=acc.max())

    plt.grid()

    hw = 1.2



    q_away = ax.quiver(x_coord[0:10], y_coord[0:10],

                       a_x[0:10], a_y[0:10],acc[0:10],

                       cmap='autumn', norm=norm, 

                       scale=30, headwidth=hw)

    q_home = ax.quiver(x_coord[11:21], y_coord[11:21],

                       a_x[11:21], a_y[11:21],acc[11:21],

                       cmap='winter', norm=norm, 

                       scale=30, headwidth=hw)



    plt.plot(x_coord[0:10],y_coord[0:10],'o',color='red', label="home")

    plt.plot(x_coord[11:21],y_coord[11:21],'o',color='blue', label="away")



    plt.plot(x_coord[rusher_idx],y_coord[rusher_idx],'d',color='brown', markersize=16)



    ax.axvline(10,c='gray') #Home Endzone

    ax.axvline(60,c='gray') #Center line

    ax.axvline(110,c='gray') #Away Endzone

    

    #PlayDirection arrow

    if play_dir == 'right':

        ax.arrow(x=3,y=51,dx=3,dy=0,width=0.4,color='b')

    else:

        ax.arrow(x=8,y=51,dx=-3,dy=0,width=0.4,color='r')



    #Score Board

    ax.text(1, 47, game_clock[:5], size=22, color='orange')

    ax.text(1, 44, 'HOME TOGO GUEST', size=13, color='k')

    score = format(home_score, '02') + '    ' + format(distance, '02') + '    ' + format(visitor_score, '02')

    ax.text(1, 42, score, size=18, color='orange')

    ax.text(1, 38, str(down) + ' down', size=18, color='k')

    ax.text(1, 35, str(quarter) + ' quarter', size=18, color='k')



    #Start line

    own = (field_position == possession_team)

    if (own and play_dir == 'left') or (own == False and play_dir == 'right'):

        start_line = 110 - yard_line

    elif (own and play_dir == 'right') or (own == False and play_dir == 'left'):

        start_line = 10 + yard_line

    #Gain line

    #10 yard line

    if play_dir == 'right':

        gain_line = start_line + yards

        ten_yard_line = start_line + distance

    elif play_dir == 'left':

        gain_line = start_line - yards

        ten_yard_line = start_line - distance

    

    plt.vlines([start_line], 0, 53, 'royalblue', label="YardLine")

    plt.vlines([gain_line], 0, 53, 'red', label="YardLine + Yards")

    plt.vlines([ten_yard_line], 0, 53, 'orange', label="First Down Line")



    plt.legend(loc="lower left", fontsize=16)

    plt.title(f"frame: {i}, Season:{season}, Week:{week}, {home_team} vs {visitor_team}, GameId:{game_id}, PlayId:{play_id}")
def show_gameplay(game_id):

    # Animation interval

    anim_interval = 400

    

    df_game = df[df['GameId']==game_id]

    play_list = df[df['GameId']==game_id]['PlayId'].unique()



    anim = animation.FuncAnimation(

          fig, update, 

          fargs = (df_game, play_list), 

          interval = anim_interval, 

          frames = play_list.size

    )



    return anim.to_jshtml()
# Chose GameId to plot

game_id = 2017091100



fig, ax = plt.subplots(figsize=(20,8.9))

HTML(show_gameplay(game_id))
# Chose which game to plot(0 - 511)

game_index = 71

game_id = games[game_index]



fig, ax = plt.subplots(figsize=(20,8.9))

HTML(show_gameplay(game_id))