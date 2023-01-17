import pandas as pd

pd.options.display.max_rows = 50

import numpy as np



import matplotlib.pyplot as plt

import matplotlib.patches as patches

import seaborn as sns



import math



from scipy.spatial import ConvexHull, convex_hull_plot_2d

from scipy.spatial import Voronoi, voronoi_plot_2d
train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
#Preprocessing

train_df['ball_carrier']=(train_df['NflId']==train_df['NflIdRusher'])



def flip_x_same_direction(x):

    if x['PlayDirection']=='left':

        return 120.0 - x['X']

    else:

        return x['X']

train_df['X_same_way']=train_df.apply(flip_x_same_direction, axis=1)



def flip_y_same_direction(x):

    if x['PlayDirection']=='left':

        return 53.3 - x['Y']

    else:

        return x['Y']

train_df['Y_same_way']=train_df.apply(flip_y_same_direction, axis=1)



def corrected_2017_orientation(x):

    if x['Season']!=2017:

        return x['Orientation']

    else:

        return np.mod(90+x['Orientation'],360)

train_df['Orientation_corrected']=train_df.apply(corrected_2017_orientation, axis=1)



def flip_Orientation_same_direction(x):

    if x['PlayDirection']=='left':

        return 360.0 - x['Orientation_corrected']

    else:

        return x['Orientation_corrected']

train_df['Orientation_same_way']=train_df.apply(flip_Orientation_same_direction, axis=1)



def flip_Dir_same_direction(x):

    if x['PlayDirection']=='left':

        return 360.0 - x['Dir']

    else:

        return x['Dir']

train_df['Dir_same_way']=train_df.apply(flip_Dir_same_direction, axis=1)



train_df.loc[train_df.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"

train_df.loc[train_df.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"



train_df.loc[train_df.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"

train_df.loc[train_df.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"



train_df.loc[train_df.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"

train_df.loc[train_df.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"



train_df.loc[train_df.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"

train_df.loc[train_df.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"



def side_of_ball(x):

    if x['Team']=='away':

        if x['VisitorTeamAbbr']==x['PossessionTeam']:

            return 'OFF' 

        else:

            return 'DEF'

    elif x['Team']=='home':

        if x['VisitorTeamAbbr']==x['PossessionTeam']:

            return 'DEF'

        else:

            return 'OFF'

    else:

        return 'UNK'

        

train_df['side_of_ball']=train_df.apply(side_of_ball, axis=1)



def calculate_yards_to_end_zone(x):

    if x['PossessionTeam']==x['FieldPosition']:

        return 100-x['YardLine']

    else:

        return x['YardLine']

train_df['Yards_to_end_zone']=train_df.apply(calculate_yards_to_end_zone, axis=1)

train_df['X_to_YardLine']=train_df['X_same_way']-(110-train_df['Yards_to_end_zone'])
#Based off od code from https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python

train_df['Dir_rad'] = np.mod(90 - train_df.Dir_same_way, 360) * math.pi/180.0  

train_df['Orientation_rad'] = np.mod(90 - train_df.Orientation_same_way, 360) * math.pi/180.0

train_df['ToLeft'] = train_df.PlayDirection == "left"



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

    return fig, ax



def get_dx_dy(radian_angle, dist):

    dx = dist * math.cos(radian_angle)

    dy = dist * math.sin(radian_angle)

    return dx, dy



def show_play(play_id, train=train_df):

    df = train[train.PlayId == play_id]

    fig, ax = create_football_field()

    ax.scatter(df.X_same_way, df.Y_same_way, cmap='rainbow', c=~(df.Team == 'home'), s=100)

    rusher_row = df[df.NflIdRusher == df.NflId]

    ax.scatter(rusher_row.X_same_way, rusher_row.Y_same_way, color='black')

    yards_covered = rusher_row["Yards"].values[0]

    x = rusher_row['X_same_way'].values[0]

    y = rusher_row["Y_same_way"].values[0]

    rusher_dir = rusher_row["Dir_rad"].values[0]

    rusher_orientation = rusher_row["Orientation_rad"].values[0]

    #rusher_orientation_orig = rusher_row["Orientation"].values[0]

    rusher_speed = rusher_row["S"].values[0]

    dx, dy = get_dx_dy(rusher_dir, rusher_speed)

    dx_o, dy_o = get_dx_dy(rusher_orientation, rusher_speed)

    #dx_oo, dy_oo = get_dx_dy(rusher_orientation_orig, rusher_speed)

    

    ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.3, color='black')

    ax.arrow(x, y, dx_o, dy_o, length_includes_head=True, width=0.3, color='yellow')

    #ax.arrow(x, y, dx_oo, dy_oo, length_includes_head=True, width=0.3, color='brown')

    left = 'left' if df.ToLeft.sum() > 0 else 'right'

    plt.title(f'Play # {play_id} moving to {left}, yard distance is {yards_covered}', fontsize=20)

    def label_point(x, y, val, ax):

        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)

        for i, point in a.iterrows():

            ax.text(point['x']+.02, point['y'], str(point['val']))



    label_point(df.X_same_way, df.Y_same_way, df.JerseyNumber, ax)

    

    runner_v_def=train_df.query("PlayId==%d and (side_of_ball=='DEF' or ball_carrier==True)" % play_id)[['Team','X_same_way','Y_same_way','Position','JerseyNumber','side_of_ball','ball_carrier']].sort_values(by='Y_same_way')

    #plt.plot(runner_v_def['X_same_way'], runner_v_def['Y_same_way'], 'o')

    runner_v_def_points=runner_v_def.copy()

    layer_colors=['green','yellow','red','black']

    for layer_of_def in range(0,4):



        #define hull

        try:

            layer_hull = ConvexHull(runner_v_def_points[['X_same_way','Y_same_way']])

            #calculate hull measures

                #area

                #max_X_to_yardline

                #related voronoi max_X_to_yardline

            #plot simplices

            for simplex in layer_hull.simplices:

                plt.plot(runner_v_def_points['X_same_way'].iloc[simplex], runner_v_def_points['Y_same_way'].iloc[simplex], 'k-', c=layer_colors[layer_of_def])

            #define rusher defender points for next layer

            runner_v_def_points = runner_v_def_points.drop(index=runner_v_def_points.iloc[layer_hull.vertices].query("side_of_ball=='DEF'").index)

        except:

            pass

            #calculate hull measures - fill with previous layer metric? Or would that 

                #area

                #max_X_to_yardline

                #related voronoi max_X_to_yardline

    

    plt.legend()

    plt.show()
play_id=20170907000345

show_play(play_id)
play_id=20170907000118

show_play(play_id)
play_id=20170907000139

show_play(play_id)
play_id=20170907000189

show_play(play_id)
layer_hull.simplices[:,1]
runner_v_def_points
layer_hull.simplices
for simplex in layer_hull.simplices:

    print(runner_v_def_points['X_same_way'].iloc[simplex], runner_v_def_points['Y_same_way'].iloc[simplex])
runner_v_def_points[['X_same_way','Y_same_way']].iloc[layer_hull.vertices]
runner_v_def_points['X_same_way'].iloc[simplex]
runner_v_def_points['Y_same_way'].iloc[simplex]
vor.vertices
vor.vertices[:,0].max()
layer_voronoi_hull=ConvexHull(vor.vertices)
voronoi_points
runner_v_def=train_df.query("PlayId==%d and (side_of_ball=='DEF' or ball_carrier==True)" % play_id)[['Team','X_to_YardLine','Y_same_way','Position','JerseyNumber','side_of_ball','ball_carrier']].sort_values(by='Y_same_way')

#plt.plot(runner_v_def['X_same_way'], runner_v_def['Y_same_way'], 'o')

runner_v_def_points=runner_v_def.copy()

layer_colors=['green','yellow','red','black']

hull_values=[play_id]

for layer_of_def in range(0,4):

    try:

        #define hull

        layer_hull = ConvexHull(runner_v_def_points[['X_to_YardLine','Y_same_way']])

        #calculate hull measures

        hull_area=layer_hull.area

        hull_expected_gain=runner_v_def_points.X_to_YardLine.max()

        hull_width=runner_v_def_points.Y_same_way.max()-runner_v_def_points.Y_same_way.min()

        hull_defenders=len(layer_hull.vertices)-1

        

        hull_values.append(hull_area)#area

        hull_values.append(hull_expected_gain)#depth (max_X-min_X)

        hull_values.append(hull_width)#width (Y_max-Ymin)

        hull_values.append(hull_defenders)

        

        #related voronoi max_X_to_yardline

        runner_vertice_backstop=runner_v_def_points.query("side_of_ball=='OFF'")[['X_to_YardLine','Y_same_way']]

        runner_vertice_backstop2=runner_vertice_backstop.copy()

        runner_vertice_backstop['X_to_YardLine']=runner_vertice_backstop['X_to_YardLine']-1.0

        runner_vertice_backstop['Y_same_way']=runner_vertice_backstop['Y_same_way']-1.0

        runner_vertice_backstop2['X_to_YardLine']=runner_vertice_backstop2['X_to_YardLine']-1.0

        runner_vertice_backstop2['Y_same_way']=runner_vertice_backstop2['Y_same_way']+1.0

        

        voronoi_points=runner_v_def_points[['X_to_YardLine','Y_same_way']].iloc[layer_hull.vertices].append(runner_vertice_backstop).append(runner_vertice_backstop2) 

        vor = Voronoi(voronoi_points)

        layer_voronoi_hull=ConvexHull(vor.vertices)

        voronoi_area=layer_voronoi_hull.area

        voronoi_expected_gain=vor.vertices[:,0].max()

        voronoi_width=vor.vertices[:,1].max()-vor.vertices[:,1].min()

        hull_values.append(voronoi_area)#area

        hull_values.append(voronoi_expected_gain)#depth (max_X-min_X)

        hull_values.append(voronoi_width)#width (Y_max-Ymin)

        fig = voronoi_plot_2d(vor)

        #plot simplices

        for simplex in layer_hull.simplices:

            plt.plot(runner_v_def_points['X_to_YardLine'].iloc[simplex], runner_v_def_points['Y_same_way'].iloc[simplex], 'k-', c=layer_colors[layer_of_def])

        #define rusher defender points for next layer

        runner_v_def_points = runner_v_def_points.drop(index=runner_v_def_points.iloc[layer_hull.vertices].query("side_of_ball=='DEF'").index) 

    except:

        #calculate hull measures - fill with previous layer metric? Or would that 

        hull_values.append(hull_area)#area

        hull_values.append(hull_expected_gain)#depth (max_X-min_X)

        hull_values.append(hull_width)#width (Y_max-Ymin)

        hull_values.append(0)

        hull_values.append(voronoi_area)#area

        hull_values.append(voronoi_expected_gain)#depth (max_X-min_X)

        hull_values.append(voronoi_width)#width (Y_max-Ymin)

            #related voronoi max_X_to_yardline



hull_columns=['PlayId',

'hull_secondary_area', 'hull_secondary_depth','hull_secondary_width','hull_secondary_defenders','voronoi_secondary_area', 'voronoi_secondary_depth','voronoi_secondary_width',

'hull_contain_area', 'hull_contain_depth','hull_contain_width','hull_contain_defenders','voronoi_contain_area', 'voronoi_contain_depth','voronoi_contain_width',

'hull_2nd_attack_area', 'hull_2nd_attack_depth','hull_2nd_attack_width','hull_2nd_attack_defenders','voronoi_2nd_attack_area', 'voronoi_2nd_attack_depth','voronoi_2nd_attack_width',

'hull_1st_attack_area', 'hull_1st_attack_depth','hull_1st_attack_width','hull_1st_attack_defenders','voronoi_1st_attack_area', 'voronoi_1st_attack_depth','voronoi_1st_attack_width'

             ]

pd.DataFrame([hull_values],columns=hull_columns)
layer_hull = ConvexHull(runner_v_def_points[['X_same_way','Y_same_way']])

#calculate hull measures

hull_area=layer_hull.area

hull_depth=runner_v_def_points.X_same_way.max()-runner_v_def_points.X_same_way.min()

hull_width=runner_v_def_points.Y_same_way.max()-runner_v_def_points.Y_same_way.min()

hull_defenders=len(layer_hull.vertices)-1



hull_values.append(hull_area)#area

hull_values.append(hull_depth)#depth (max_X-min_X)

hull_values.append(hull_width)#width (Y_max-Ymin)

hull_values.append(hull_defenders)

    #related voronoi max_X_to_yardline

runner_vertice_backstop=runner_v_def_points[['X_same_way','Y_same_way']].query("ball_carrier==True")

runner_vertice_backstop['X_same_way']=runner_vertice_backstop['X_same_way']-1.0

voronoi_points=runner_v_def_points[['X_same_way','Y_same_way']].iloc[layer_hull.vertices].append(runner_vertice_backstop, ignore_index=True)    

vor = Voronoi(runner_v_def_points[['X_same_way','Y_same_way']].iloc[layer_hull.vertices])

fig = voronoi_plot_2d(vor)

#plot simplices

for simplex in layer_hull.simplices:

    plt.plot(runner_v_def_points['X_same_way'].iloc[simplex], runner_v_def_points['Y_same_way'].iloc[simplex], 'k-', c=layer_colors[layer_of_def])
def calc_def_hull_measures(runner_v_def):

    runner_v_def_points=runner_v_def.copy()

    layer_colors=['green','yellow','red','black']

    layer_names=['secondary','contain','red','black']

    hull_values=[]

    for layer_of_def in range(0,4):

        try:

            #define hull

            layer_hull = ConvexHull(runner_v_def_points[['X_same_way','Y_same_way']])

            #calculate hull measures

            hull_area=layer_hull.area

            hull_depth=runner_v_def_points.X_same_way.max()-runner_v_def_points.X_same_way.min()

            hull_width=runner_v_def_points.Y_same_way.max()-runner_v_def_points.Y_same_way.min()

            hull_defenders=len(layer_hull.vertices)-1



            data['hull_secondary_area']=hull_area#hull_values.append(hull_area)#area

            data['hull_secondary_depth']=hull_depth#hull_values.append(hull_depth)#depth (max_X-min_X)

            data['hull_secondary_width']=hull_width#hull_values.append(hull_width)#width (Y_max-Ymin)

            data['hull_secondary_defenders']=hull_defenders#hull_values.append(hull_defenders)

                #related voronoi max_X_to_yardline



            #plot simplices

            #for simplex in layer_hull.simplices:

            #    plt.plot(runner_v_def_points['X_same_way'].iloc[simplex], runner_v_def_points['Y_same_way'].iloc[simplex], 'k-', c=layer_colors[layer_of_def])

            #define rusher defender points for next layer

            runner_v_def_points = runner_v_def_points.drop(index=runner_v_def_points.iloc[layer_hull.vertices].query("side_of_ball=='DEF'").index) 

        except:

            #calculate hull measures - fill with previous layer metric? Or would that 

            hull_values.append(hull_area)#area

            hull_values.append(hull_depth)#depth (max_X-min_X)

            hull_values.append(hull_width)#width (Y_max-Ymin)

            hull_values.append(0)

                    #related voronoi max_X_to_yardline



    hull_columns=[

'hull_secondary_area', 'hull_secondary_depth','hull_secondary_width','hull_secondary_defenders',

'hull_contain_area', 'hull_contain_depth','hull_contain_width','hull_contain_defenders',

'hull_2nd_attack_area', 'hull_2nd_attack_depth','hull_2nd_attack_width','hull_2nd_attack_defenders',

'hull_1st_attack_area', 'hull_1st_attack_depth','hull_1st_attack_width','hull_1st_attack_defenders']

    return pd.DataFrame([hull_values],columns=hull_columns)

test=train_df.query("PlayId in ([20170907000189,20170907000139]) and (side_of_ball=='DEF' or ball_carrier==True)")[['PlayId','X_same_way','Y_same_way','side_of_ball']].groupby('PlayId').apply(calc_def_hull_measures).droplevel(level=1)
test
def calc_def_hull_measures(runner_v_def_points):

    #runner_v_def_points=runner_v_def.copy()

    hull_values=[]

    for layer_of_def in range(0,4):

        try:

            #define hull

            layer_hull = ConvexHull(runner_v_def_points[['X_same_way','Y_same_way']])

            #calculate hull measures

            hull_area=layer_hull.area

            hull_depth=runner_v_def_points.X_same_way.max()-runner_v_def_points.X_same_way.min()

            hull_width=runner_v_def_points.Y_same_way.max()-runner_v_def_points.Y_same_way.min()

            hull_defenders=len(layer_hull.vertices)-1



            hull_values.append(hull_area)#area

            hull_values.append(hull_depth)#depth (max_X-min_X)

            hull_values.append(hull_width)#width (Y_max-Ymin)

            hull_values.append(hull_defenders)

                #related voronoi max_X_to_yardline

                

            #define rusher defender points for next layer

            runner_v_def_points = runner_v_def_points.drop(index=runner_v_def_points.iloc[layer_hull.vertices].query("side_of_ball=='DEF'").index) 

        except:

            #calculate hull measures - fill with previous layer metric? Or would that 

            hull_values.append(hull_area)#area

            hull_values.append(hull_depth)#depth (max_X-min_X)

            hull_values.append(hull_width)#width (Y_max-Ymin)

            hull_values.append(0)

                    #related voronoi max_X_to_yardline



    hull_columns=[

'hull_secondary_area', 'hull_secondary_depth','hull_secondary_width','hull_secondary_defenders',

'hull_contain_area', 'hull_contain_depth','hull_contain_width','hull_contain_defenders',

'hull_2nd_attack_area', 'hull_2nd_attack_depth','hull_2nd_attack_width','hull_2nd_attack_defenders',

'hull_1st_attack_area', 'hull_1st_attack_depth','hull_1st_attack_width','hull_1st_attack_defenders']

    return pd.DataFrame([hull_values],columns=hull_columns)

defender_hulls_df=train_df.query("(side_of_ball=='DEF' or ball_carrier==True)")[['PlayId','X_same_way','Y_same_way','side_of_ball']].groupby('PlayId').apply(calc_def_hull_measures).droplevel(level=1)
defender_hulls_df