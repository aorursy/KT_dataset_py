# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nfl_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv')

nfl_df.head()
import matplotlib.pyplot as plt
def position_marker(position):

    position_map = {

        'SS': ".",

        'DE': ",",

        'ILB': "o",

        'FS': "v",

        'CB': "^",

        'DT': "<",

        'WR': ">",

        'TE': "1",

        'T': "2",

        'QB': "3",

        'RB': "4",

        'G': "8",

        'C': "s",

        'OLB': "p",

        'NT': "P",

        'FB': "*",

        'MLB': "h",

        'LB': "H",

        'OT': "+",

        'OG': "x",

        'HB': "X",

        'DB': "D",

        'S': "d",

        'DL': "|",

        'SAF': "_"

    }

    return position_map[position]
def prepare_field_plot():

    plt.figure(figsize=(4 * 5.9602649, 4 * 2.5000000))



    # play.plot.scatter(x='X',  y='Y', c=colors)

    plt.xlim(0, 120)

    plt.ylim(0, 53 + 1/3)



    # Vertical lines form 10-th yard to 110-th spaced for 5 yards

    for p in range(10, 115, 5):

        plt.axvline(p, c="#D3D3D3D3")



    plt.xticks(range(20, 110, 10), [10, 20, 30, 40, 50, 40, 30, 20, 10])

    plt.yticks([])

prepare_field_plot()



play = nfl_df[nfl_df['PlayId'] == nfl_df['PlayId'][np.random.randint(0, len(nfl_df))]]



for i, player in play.iterrows():

    # Offense team RED, defense team BLUE

    if (player.HomeTeamAbbr == player.PossessionTeam):

        if (player.Team ==  'home'):

            p_color = 'red'

        else:

            p_color = 'blue'

    else:

        if (player.Team ==  'home'):

            p_color = 'blue'

        else:

            p_color = 'red'

    

    p_marker = position_marker(player.Position)

    plt.scatter(player.X, player.Y, c=p_color, marker=p_marker)

    pass



plt.show()