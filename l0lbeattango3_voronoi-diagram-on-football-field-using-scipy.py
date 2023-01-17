import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import matplotlib as mpl

import math

%matplotlib inline
train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv')
#correct the team names

train['VisitorTeamAbbr'] = train['VisitorTeamAbbr'].replace('ARI','ARZ')

train['HomeTeamAbbr'] = train['HomeTeamAbbr'].replace('ARI','ARZ')

train['VisitorTeamAbbr'] = train['VisitorTeamAbbr'].replace('BAL','BLT')

train['HomeTeamAbbr'] = train['HomeTeamAbbr'].replace('BAL','BLT')

train['VisitorTeamAbbr'] = train['VisitorTeamAbbr'].replace('CLE','CLV')

train['HomeTeamAbbr'] = train['HomeTeamAbbr'].replace('CLE','CLV')

train['VisitorTeamAbbr'] = train['VisitorTeamAbbr'].replace('HOU','HST')

train['HomeTeamAbbr'] = train['HomeTeamAbbr'].replace('HOU','HST')



# Player is on offense

train['home_possession'] = (train['PossessionTeam'] == train['HomeTeamAbbr'])

train['team_binary'] = [True if i=="home" else False for i in train['Team']]

train['on_offense'] = (train['team_binary'] == train['home_possession'])



#correct X Y and Dir

mask = train["PlayDirection"] != "right"

train.loc[mask, "X"] = 120 - train.loc[mask, "X"]

train["X"] -= 10

train.loc[mask, "Y"] = 160/3 - train.loc[mask, "Y"]

train.loc[mask, "Dir"] = (train.loc[mask, "Dir"] + 180) % 360

train.loc[mask, "Orientation"] = np.mod(180 + train.loc[mask, 'Orientation'], 360)



#Yrds from own goal

train['YardsFromOwnGoal'] = train['YardLine']

train.loc[(train.FieldPosition!=train.PossessionTeam), 'YardsFromOwnGoal'] = (50 + (50 - train.loc[(train.FieldPosition!=train.PossessionTeam), 'YardsFromOwnGoal']))

train.loc[(train.YardLine==50), 'YardsFromOwnGoal'] = 50



#is Rusher

train['isRusher'] = (train['NflIdRusher'] == train['NflId'])

train["TeamOnOffense"] = "away"

train.loc[train["HomeTeamAbbr"] == train["PossessionTeam"], "TeamOnOffense"] = "home"

train["IsOnOffense"] = train["Team"] == train["TeamOnOffense"]
def create_football_field(linenumbers=True,

                          endzones=True,

                          highlight_line=False,

                          highlight_line_number=50,

                          highlighted_name='Line of Scrimmage',

                          fifty_is_los=False,

                          figsize=(12*2, 6.33*2)):

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
def get_moving(angle,speed):

    import math

    cartesianAngleRadians = (90-angle)*math.pi/180.0

    dx = speed * math.cos(cartesianAngleRadians)

    dy = speed * math.sin(cartesianAngleRadians) 

    return dx,dy
def show_play_std(playid, train=train):

    from scipy.spatial import Voronoi, voronoi_plot_2d



    df = train[train.PlayId == playid]

    

    StdYardLine = df.YardsFromOwnGoal.values[0]-10



    fig, ax = create_football_field(highlight_line=True,highlight_line_number = StdYardLine)

    ax.scatter(df.X, df.Y, cmap='seismic', c=~df.IsOnOffense)

    

    points = df[["X","Y"]].to_numpy()

    vor = Voronoi(points)



    for index, simplex in enumerate(vor.ridge_vertices):

        simplex = np.asarray(simplex)

        if np.all(simplex >= 0):

            ax.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], 'b-')



    center = points.mean(axis=0)

    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):

        simplex = np.asarray(simplex)

        if np.any(simplex < 0):

            i = simplex[simplex >= 0][0] # finite end Voronoi vertex

            t = points[pointidx[1]] - points[pointidx[0]] # tangent

            t /= np.linalg.norm(t)

            n = np.array([-t[1], t[0]]) # normal

            midpoint = points[pointidx].mean(axis=0)

            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100

            ax.plot([vor.vertices[i,0], far_point[0]], [vor.vertices[i,1], far_point[1]], 'b--')

    

    rusher_row = df[df.isRusher]

    yards_covered = rusher_row["Yards"].values[0]

    x = rusher_row["X"].values[0]

    y = rusher_row["Y"].values[0]

    rusher_dir = rusher_row["Dir"].values[0]

    rusher_speed = rusher_row["S"].values[0]

    

    if (yards_covered ==0):

        dx,dy = get_moving(rusher_dir,rusher_speed)

    else:

        dx,dy = get_moving(rusher_dir,yards_covered)

        

    ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.2, color = 'yellow')



    plt.title('Play # ' +  str(playid) + "and yard distance is " + str(yards_covered))

    plt.legend(loc = 'lower right')

    plt.show()

    

    return
myplayid = 20170907000118

show_play_std(myplayid, train=train)