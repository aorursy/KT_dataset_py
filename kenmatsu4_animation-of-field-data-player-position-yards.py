import math

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

from IPython.display import Image

from matplotlib import animation as ani



plt.rcParams["patch.force_edgecolor"] = True

#rc('text', usetex=True)

from IPython.display import display # Allows the use of display() for DataFrames

import seaborn as sns

sns.set(style="whitegrid", palette="muted", color_codes=True)

sns.set_style("whitegrid", {'grid.linestyle': '--'})

red = sns.xkcd_rgb["light red"]

green = sns.xkcd_rgb["medium green"]

blue = sns.xkcd_rgb["denim blue"]

TARGET_COL_NAME = "Yards"

train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False, dtype={"GameId":object, "PlayId":object})

train.TimeHandoff = pd.to_datetime(train.TimeHandoff)

train.TimeSnap = pd.to_datetime(train.TimeSnap)
DEBUG = False



X_MAX = 120

Y_MAX = 53.3

down_disp_dict = {i+1: v for i, v in enumerate(["1st", "2nd", "3rd", "4th"])}

arrow_length_coef = 5



def draw_scatter_arrow(df, color, team, rusher_id):

    rusher_color = np.where(df.NflId == rusher_id, "r", "k")

    plt.scatter(x=df.X, y=df.Y, color=color, alpha=0.7, s=30, label=team, edgecolors=rusher_color)

    for i, df_s in df.iterrows():

        distance_each = df_s.Dis

        dx = math.cos(math.radians(df_s.Dir))*distance_each*arrow_length_coef

        dy = math.sin(math.radians(df_s.Dir))*distance_each*arrow_length_coef

        if dx!=0 or dy!=0:

            plt.arrow(x=df_s.X, y=df_s.Y, dx=dx, dy=dy, length_includes_head=True, color='k')



def draw_field_animation(game_id):

    def animate(nframe):

        global num_frame

        plt.clf()



        df = df_list[nframe]

        play_dir = df.iloc[0].PlayDirection

        quater = df.iloc[0].Quarter

        yards = df.iloc[0].Yards

        field_position  = df.iloc[0].FieldPosition

        possession_team = df.iloc[0].PossessionTeam

        yard_line = df.iloc[0].YardLine

        down = df.iloc[0].Down

        distance = df.iloc[0].Distance

        handoff_snap = df.iloc[0].TimeHandoff.value//1000//1000 - df.iloc[0].TimeSnap.value//1000//1000

        rusher = df.iloc[0].NflIdRusher

        

        score_home = df.iloc[0].HomeScoreBeforePlay

        score_vistor = df.iloc[0].VisitorScoreBeforePlay



        # ref: https://www.kaggle.com/hookbook/nfl-data-visualization-of-plays

        own = (field_position == possession_team)

        if (own and play_dir == 'left') or (own == False and play_dir == 'right'):

            start_line = 110 - yard_line

        elif (own and play_dir == 'right') or (own == False and play_dir == 'left'):

            start_line = 10 + yard_line

        if play_dir == 'right':

            gain_line = start_line + yards

        elif play_dir == 'left':

            gain_line = start_line - yards

            

        df_home = df[df.Team=="home"]

        df_away = df[df.Team=="away"]



        if play_dir == 'right':

            arrow_x = 30

            arrow_dx = 60

        else:

            arrow_x = 90

            arrow_dx = -60

            

        plt.arrow(x=arrow_x, y=25, dx=arrow_dx, dy=0, length_includes_head=True, width=10, head_length=10,head_width=20, 

                  color='pink', edgecolor=None, alpha=0.2)

        plt.fill_between([0,10], Y_MAX, color="pink", alpha=0.3)

        plt.fill_between([110,120], Y_MAX, color="pink", alpha=0.3)

        

        plt.vlines(start_line, 0, Y_MAX, "r", label="YardLine")

        plt.vlines(gain_line, 0, Y_MAX, "purple", label="YardLine + Yards")

        

        draw_scatter_arrow(df_home, color="b", team="home", rusher_id=rusher)

        draw_scatter_arrow(df_away, color="g", team="away", rusher_id=rusher)

        

        plt.xlim(0, X_MAX)

        plt.ylim(0, Y_MAX)

        plt.title(f"frame: {nframe}, Q:{quater},home{score_home} vs visitor{score_vistor}, {down_disp_dict[down]}&{distance} Yards:{yards}, PlayDir:{play_dir}, own:{own}, handoff-snap:{handoff_snap} ")

        plt.legend(loc="lower left")

        plt.xticks(np.arange(0,120,10), np.arange(-10,110,10))



    train_game = train[train.GameId==game_id]

    df_list = [g for _, g in train_game.groupby("PlayId")]

    num_frame = 3 if DEBUG else len(df_list)

    

    fig = plt.figure(figsize=(12,5))

    anim = ani.FuncAnimation(fig, animate, frames=int(num_frame))

    

    save_name = f'field_anim_{game_id}.gif'

    anim.save(save_name, writer='imagemagick', fps=1, dpi=128)

    plt.close()

    return save_name
n_game = 1 if DEBUG else 10

for game_id in  train.GameId.unique()[:n_game]:

    print(f"GameId: {game_id}")

    save_name = draw_field_animation(game_id)

    with open(save_name, 'rb') as file:

        display(Image(file.read()))