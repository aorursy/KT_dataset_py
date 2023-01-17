import os

import matplotlib

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import math

import scipy

from random import choice

from scipy.spatial.distance import euclidean

from scipy.special import expit

from IPython.display import HTML

from matplotlib import animation

from tqdm import tqdm





train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
def standardize_dataset(train):

    train['ToLeft'] = train.PlayDirection == "left"

    train['IsBallCarrier'] = train.NflId == train.NflIdRusher

    train['TeamOnOffense'] = "home"

    train.loc[train.PossessionTeam != train.HomeTeamAbbr, 'TeamOnOffense'] = "away"

    train['IsOnOffense'] = train.Team == train.TeamOnOffense # Is player on offense?

    train['YardLine_std'] = 100 - train.YardLine

    train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  

            'YardLine_std'

             ] = train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  

              'YardLine']

    train['X_std'] = train.X

    train.loc[train.ToLeft, 'X_std'] = 120 - train.loc[train.ToLeft, 'X'] 

    train['Y_std'] = train.Y

    train.loc[train.ToLeft, 'Y_std'] = 53.3 - train.loc[train.ToLeft, 'Y'] 

    train['Orientation_std'] = train.Orientation

    train.loc[train.ToLeft, 'Orientation_std'] = np.mod(180 + train.loc[train.ToLeft, 'Orientation_std'], 360)

    train['Dir_std'] = train.Dir

    train.loc[train.ToLeft, 'Dir_std'] = np.mod(180 + train.loc[train.ToLeft, 'Dir_std'], 360)

    train.loc[train['Season'] == 2017, 'Orientation'] = np.mod(90 + train.loc[train['Season'] == 2017, 'Orientation'], 360)    

    

    return train
dominance_df = standardize_dataset(train_df)

dominance_df['Rusher'] = dominance_df['NflIdRusher'] == dominance_df['NflId']



dominance_df.head(3)
def radius_calc(dist_to_ball):

    ''' I know this function is a bit awkward but there is not the exact formula in the paper,

    so I try to find something polynomial resembling

    Please consider this function as a parameter rather than fixed

    I'm sure experts in NFL could find a way better curve for this'''

    return 4 + 6 * (dist_to_ball >= 15) + (dist_to_ball ** 3) / 560 * (dist_to_ball < 15)

@np.vectorize

def compute_influence(x_point, y_point, player_id, play_id):

    my_play = dominance_df[dominance_df['PlayId']==play_id]

    '''Compute the influence of a certain player over a coordinate (x, y) of the pitch

    '''

    point = np.array([x_point, y_point])

    theta = math.radians(my_play.loc[player_id]['Orientation_std'])

    speed = my_play.loc[player_id]['S']

    player_coords = my_play.loc[player_id][['X_std', 'Y_std']].values

    ball_coords = my_play[my_play['IsBallCarrier']][['X_std', 'Y_std']].values

    

    dist_to_ball = euclidean(player_coords, ball_coords)



    S_ratio = (speed / 13) ** 2    # we set max_speed to 13 m/s

    RADIUS = radius_calc(dist_to_ball)  # updated



    S_matrix = np.matrix([[RADIUS * (1 + S_ratio), 0], [0, RADIUS * (1 - S_ratio)]])

    R_matrix = np.matrix([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    COV_matrix = np.dot(np.dot(np.dot(R_matrix, S_matrix), S_matrix), np.linalg.inv(R_matrix))

    

    norm_fact = (1 / 2 * np.pi) * (1 / np.sqrt(np.linalg.det(COV_matrix)))    

    mu_play = player_coords + speed * np.array([np.cos(theta), np.sin(theta)]) / 2

    

    intermed_scalar_player = np.dot(np.dot((player_coords - mu_play),

                                    np.linalg.inv(COV_matrix)),

                             np.transpose((player_coords - mu_play)))

    player_influence = norm_fact * np.exp(- 0.5 * intermed_scalar_player[0, 0])

    

    intermed_scalar_point = np.dot(np.dot((point - mu_play), 

                                    np.linalg.inv(COV_matrix)), 

                             np.transpose((point - mu_play)))

    point_influence = norm_fact * np.exp(- 0.5 * intermed_scalar_point[0, 0])



    return point_influence / player_influence
@np.vectorize

def pitch_control(x_point, y_point, play_id):

    my_play = dominance_df[dominance_df['PlayId']==play_id]

    '''Compute the pitch control over a coordinate (x, y)

    '''

    home_ids = my_play[my_play['IsOnOffense']==True].index

    home_control = np.around(compute_influence(x_point, y_point, home_ids, play_id), 2)

    home_score = np.sum(home_control)

    

    away_ids = my_play[my_play['IsOnOffense']==False].index

    away_control = np.around(compute_influence(x_point, y_point, away_ids, play_id), 2)

    away_score = np.sum(away_control)

    

    return expit(home_score - away_score)
# animation interval

ANIM_INTERVAL = 600



# don't make it too large otherwise it'll be long to run

GRID_SIZE = 10
def plot_pitch_control(my_play):

    front = 25

    behind = 15

    left = right = 20

    num_points_meshgr = (GRID_SIZE, GRID_SIZE)

    

    x_coords = my_play['X_std'].values

    y_coords = my_play['Y_std'].values

    player_coords = my_play[my_play['Rusher']][['X_std', 'Y_std']].values[0]

    

    play_id = my_play['PlayId'].values[0]



    X, Y = np.meshgrid(np.linspace(player_coords[0] - behind, 

                                   player_coords[0] + front, 

                                   num_points_meshgr[0]), 

                       np.linspace(player_coords[1] - left, 

                                   player_coords[1] + right, 

                                   num_points_meshgr[1]))



    #fill all

    #X, Y = np.meshgrid(np.linspace(0, 120, 24), np.linspace(0, 53.3, 10))



    # infl is an array of shape num_points with values in [0,1] accounting for the pitch control

    infl = pitch_control(X, Y, play_id)



    plt.contourf(X, Y, infl, cmap ='bwr')

    plt.plot(player_coords[0] ,player_coords[1], markeredgecolor='black', c='snow', marker='o', markersize=14, label='Rusher')

    plt.scatter(x_coords[11:21] ,y_coords[11:21], c='orange', marker=',', s=120, label='Diffence')

    plt.scatter(x_coords[0:10] ,y_coords[0:10], c='purple', marker='>', s=120, label='Offence')
games = np.unique(dominance_df['GameId'].values)

plays = np.unique(dominance_df['PlayId'].values)



def update(i, df_game, play_list):

    my_play = df_game[df_game['PlayId']==play_list[i]]

    

    # parameters

    game_id         = my_play['GameId'].values[0]

    play_id         = my_play['PlayId'].values[0]

    game_clock      = my_play['GameClock'].values[0]

    distance        = my_play['Distance'].values[0]

    yards           = my_play['Yards'].values[0]

    season          = my_play['Season'].values[0]

    week            = my_play['Week'].values[0]

    down            = my_play['Down'].values[0]

    quarter         = my_play['Quarter'].values[0]

    yard_line       = 10 + my_play['YardLine_std'].values[0]

    gain_line       = yard_line + yards

    first_down_line = yard_line + distance

    

    # plot

    plt.cla()

    plt.grid()

    plt.ylim((0, 53))

    plt.xlim((0, 120))



    #Pitch Control

    plot_pitch_control(my_play)    

    

    ax.axvline(10,c='gray')

    ax.axvline(60,c='gray')

    ax.axvline(110,c='gray')



    plt.vlines([yard_line], 0, 53, 'royalblue', label="YardLine")

    plt.vlines([gain_line], 0, 53, 'red', label="YardLine + Yards")

    plt.vlines([first_down_line], 0, 53, 'orange', label="First Down Line")



    plt.legend(loc="lower left", fontsize=16)

    

    plt.title(f"frame: {i}, Season:{season}, Week:{week}, Clock:{game_clock[:5]}, Distance:{distance:02}, Yards:{yards:02}, {down} down, {quarter} quarter, PlayId:{play_id}")





def show_gameplay(game_id):

    df_game = dominance_df[dominance_df['GameId']==game_id]

    play_list = dominance_df[dominance_df['GameId']==game_id]['PlayId'].unique()



    anim = animation.FuncAnimation(

          fig, update, 

          fargs = (df_game, play_list), 

          interval = ANIM_INTERVAL, 

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