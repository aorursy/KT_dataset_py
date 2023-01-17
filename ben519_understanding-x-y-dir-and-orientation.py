import numpy as np

import pandas as pd

import getpass

import matplotlib.pyplot as plt



# ==== Setup =======================================================================================



np.set_printoptions(suppress=True)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 0)



if getpass.getuser() in ['Ben']:

    input_file = 'input/train.csv'

else:

    from kaggle.competitions import nflrush

    env = nflrush.make_env()

    input_file = '/kaggle/input/nfl-big-data-bowl-2020/train.csv'



# read train

train = pd.read_csv(filepath_or_buffer = input_file, low_memory = False)
def plot_play(playId, arrows_as = 'Dir'):

    """

    Plot a single play

    :param playId: id of the play 

    :param arrows_as: should be either 'Dir' or 'Orientation'

    :return: plot of players with arrows

    """



    # Get 'play players' and 'play' from train

    play_players = train.loc[train.PlayId == playId].copy()

    play = play_players.iloc[[0]]



    # Determine which players are on offense by identifying which team the rusher is on

    rusher = play_players.loc[play_players.NflId == play_players.NflIdRusher]

    play_players.loc[:, 'PlayerOnOffense'] = play_players.Team.values == rusher.Team.values



    # Create field 'ArrowAngle'

    play_players['ArrowAngle'] = play_players['Dir'] if arrows_as == 'Dir' else play_players['Orientation']

    if (play.Season.values[0] == 2017 and arrows_as == 'Orientation'):

        play_players['ArrowAngle'] = (360 - play_players.ArrowAngle).mod(360)

    else:

        play_players['ArrowAngle'] = (360 - play_players.ArrowAngle + 90).mod(360)



    # Create fields Arrow_dx, Arrow_dy

    play_players['Arrow_dx'] = np.cos(play_players.ArrowAngle * (np.pi/180))

    play_players['Arrow_dy'] = np.sin(play_players.ArrowAngle * (np.pi/180))



    # Split offense and defense players

    play_players_offense = play_players.loc[play_players.PlayerOnOffense].copy()

    play_players_defense = play_players.loc[~play_players.PlayerOnOffense].copy()



    # Plot

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.axvline(x=10, linewidth=1)

    ax.axvline(x=110, linewidth=1)

    ax.axhline(y=0, linewidth=1)

    ax.axhline(y=53.5, linewidth=1)

    ax.scatter(play_players_offense.X, play_players_offense.Y, color="red", label="offense", s = 20)

    ax.scatter(play_players_defense.X, play_players_defense.Y, color="blue", label="defense", s = 20)

    for i in range(0, play_players.shape[0]):

        ax.arrow(

            x = play_players.X.values[i],

            y = play_players.Y.values[i],

            dx = play_players.Arrow_dx.values[i],

            dy = play_players.Arrow_dy.values[i],

            head_width=0.5,

            head_length=0.5,

            fc='k',

            ec='k'

        )

    ax.text(60, 0 - 5, 'Home Sideline', horizontalalignment='center', verticalalignment='bottom')

    ax.text(60, 53.5 + 5, 'Away Sideline', horizontalalignment='center', verticalalignment='top', rotation=180)

    ax.set_xlim(0, 120)

    ax.set_ylim(0 - 5, 53.5 + 5)

    ax.set_title(f"PlayId: {play.PlayId.values[0]} (moving {play.PlayDirection.values[0]})\n{arrows_as}")

    fig.legend()

    fig.show()
# Derrick Henry, 2018, moving right, https://www.youtube.com/watch?v=tlZvgdcIXvI

plot_play(20181206001238, 'Orientation')
# Lamar Miller, 2018, moving left, https://www.youtube.com/watch?v=p-ptA3nQxCA

plot_play(20181126001222, 'Orientation')
# Nick Chubb, 2018, moving left, https://www.youtube.com/watch?v=NvQiykZIBNA

plot_play(20181111022155, 'Orientation')
# Adrian Peterson, 2018, moving left, https://www.youtube.com/watch?v=AMLKvNs2Ec8

plot_play(20181203001224, 'Orientation')
# Leonard Fournette, 2017, moving right, https://youtu.be/Dp3zkB3NRDA?t=114

plot_play(20171008074020, 'Orientation')
# Melvin Gordon, 2017, moving right, https://www.youtube.com/watch?v=oUHaQKmyn7U

plot_play(20171029030426, 'Orientation')
# Bilal Powell, 2017, moving left, https://www.youtube.com/watch?v=zDtDanILhAc

plot_play(20171001080397, 'Orientation')
# Saquon Barkley, 2018, moving left, https://www.youtube.com/watch?v=E4IesbDwpq4

plot_play(20181209081494, 'Orientation')
# Kerryon Johnson, 2018, moving right, https://youtu.be/cgZnUFAtd0c?t=27

plot_play(20181021060782, 'Orientation')
# Alvin Kamara, 2017, moving right, https://www.youtube.com/watch?v=4XAYJKiT2rc

plot_play(20171126070740, 'Orientation')