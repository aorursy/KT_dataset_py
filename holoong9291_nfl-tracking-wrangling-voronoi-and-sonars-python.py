import sys,os



import pandas as pd, numpy as np, matplotlib.pyplot as plt



%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
train  = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv')

train['ToLeft'] = train.PlayDirection.apply(lambda play_direction:play_direction=='left')

train['IsBallCarrier'] = train[['NflId','NflIdRusher']].apply(lambda row:row.NflId==row.NflIdRusher, axis=1)



team_abbr_map = {

    'ARI':'ARZ',

    'BAL':'BLT',

    'CLE':'CLV',

    'HOU':'HST'

}

train.VisitorTeamAbbr = train.VisitorTeamAbbr.apply(lambda vta:team_abbr_map[vta] if vta in team_abbr_map.keys() else vta)

train.HomeTeamAbbr = train.HomeTeamAbbr.apply(lambda vta:team_abbr_map[vta] if vta in team_abbr_map.keys() else vta)
sample_chart_v1 = train[train.ToLeft==True][['PlayId','ToLeft']].sample(3).merge(train, how='inner')

sample_chart_v1 = sample_chart_v1.append(train[train.ToLeft==False][['PlayId','ToLeft']].sample(3).merge(train, how='inner'))

sample_chart_v1
plt.figure(figsize=(30, 15))

plt.suptitle('Sample plays')

i=1

for gp,chance in sample_chart_v1.groupby('PlayId'):

    play_id = gp

    rusher = chance[chance.NflId==chance.NflIdRusher]

    home = chance[chance.Team=='home']

    away = chance[chance.Team=='away']

    #yard_line_left = offense.YardLine.iloc[0]+10 # yard_line 加10偏移量，这个10是左侧的达阵区

    #yard_line_right = offense.YardLine.iloc[0]+2*(50-offense.YardLine.iloc[0])+10

    #yard_line = yard_line_left if np.abs(yard_line_left-rusher.X.iloc[0])<=(yard_line_right-rusher.X.iloc[0]) else yard_line_right

    

    plt.subplot(3,2,i)

    i+=1

    plt.xlim(0,120)# 0~120已经包含了达阵区，实际场内只有100码，码线也是0~100的范围

    plt.ylim(-10,63)

    plt.scatter(list(home.X),list(home.Y),marker='o',c='red',s=55,alpha=0.5,label='HOME')

    plt.scatter(list(away.X),list(away.Y),marker='o',c='green',s=55,alpha=0.5,label='AWAY')

    plt.scatter(list(rusher.X),list(rusher.Y),marker='o',c='black',s=30,label='RUSHER')

    

    for line in range(10,130,10):

        plt.plot([line,line],[-100,100],c='black',linewidth=1,linestyle=':')

    

    #plt.plot([yard_line,yard_line],[-100,100],c='orange',linewidth=1.5)

    plt.plot([10,10],[-100,100],c='green',linewidth=3) # down zone left

    plt.plot([110,110],[-100,100],c='green',linewidth=3) # down zone right

    

    plt.title(play_id)

    plt.legend()



plt.show()
train_1 = train.copy()

train_1['TeamOnOffense'] = train_1[['PossessionTeam','HomeTeamAbbr']].apply(lambda row:'home' if row.PossessionTeam==row.HomeTeamAbbr else 'away', axis=1)

train_1['IsOnOffense'] = train_1[['TeamOnOffense','Team']].apply(lambda row:row.TeamOnOffense==row.Team , axis=1)

train_1['YardsFromOwnGoal'] = train_1[['FieldPosition','PossessionTeam','YardLine']].apply(lambda row:row.YardLine if row.FieldPosition==row.PossessionTeam else 50+(50-row.YardLine), axis=1)

train_1['YardsFromOwnGoal'] = train_1[['YardsFromOwnGoal','YardLine']].apply(lambda row:50 if row.YardLine==50 else row.YardsFromOwnGoal, axis=1)

train_1['X_std'] = train_1[['ToLeft','X']].apply(lambda row:120-row.X-10 if row.ToLeft else row.X-10, axis=1)

train_1['Y_std'] = train_1[['ToLeft','Y']].apply(lambda row:160/3-row.Y if row.ToLeft else row.Y, axis=1)

train_1[['Team','FieldPosition','PossessionTeam','TeamOnOffense','IsOnOffense','YardLine','YardsFromOwnGoal','X','X_std','Y','Y_std']].sample(10)
sample_chart_v2 = sample_chart_v1.merge(train_1, how='inner')

sample_chart_v2
plt.figure(figsize=(30, 15))

plt.suptitle("Sample plays, standardized, Offense moving left to right")

plt.xlabel("Distance from offensive team's own end zone")

plt.ylabel("Y coordinate")



i=1

for gp,chance in sample_chart_v2.groupby('PlayId'):

    play_id = gp

    rusher = chance[chance.NflId==chance.NflIdRusher].iloc[0]

    offense = chance[chance.IsOnOffense]

    defense = chance[~chance.IsOnOffense]

    

    plt.subplot(3,2,i)

    i+=1

    plt.xlim(0,120)

    plt.ylim(-10,63)

    

    plt.scatter(offense.X_std,offense.Y_std,marker='o',c='red',s=55,alpha=0.5,label='OFFENSE')

    plt.scatter(defense.X_std,defense.Y_std,marker='o',c='green',s=55,alpha=0.5,label='DEFENSE')

    plt.scatter([rusher.X_std],[rusher.Y_std],marker='o',c='black',s=30,label='RUSHER')

    

    for line in range(10,130,10):

        plt.plot([line,line],[-100,100],c='silver',linewidth=0.8,linestyle='-')

    

    plt.plot([rusher.YardsFromOwnGoal,rusher.YardsFromOwnGoal],[-100,100],c='black',linewidth=1.5,linestyle=':')

    plt.plot([10,10],[-100,100],c='black',linewidth=2)

    plt.plot([110,110],[-100,100],c='black',linewidth=2)

    

    plt.title(play_id)

    plt.legend()



plt.show()
def voronoi_finite_polygons_2d(vor, radius=None):

    """

    Reconstruct infinite voronoi regions in a 2D diagram to finite

    regions.



    Parameters

    ----------

    vor : Voronoi

        Input diagram

    radius : float, optional

        Distance to 'points at infinity'.



    Returns

    -------

    regions : list of tuples

        Indices of vertices in each revised Voronoi regions.

    vertices : list of tuples

        Coordinates for revised Voronoi vertices. Same as coordinates

        of input vertices, with 'points at infinity' appended to the

        end.



    """



    if vor.points.shape[1] != 2:

        raise ValueError("Requires 2D input")



    new_regions = []

    new_vertices = vor.vertices.tolist()



    center = vor.points.mean(axis=0)

    if radius is None:

        radius = vor.points.ptp().max()



    # Construct a map containing all ridges for a given point

    all_ridges = {}

    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):

        all_ridges.setdefault(p1, []).append((p2, v1, v2))

        all_ridges.setdefault(p2, []).append((p1, v1, v2))



    # Reconstruct infinite regions

    for p1, region in enumerate(vor.point_region):

        vertices = vor.regions[region]



        if all(v >= 0 for v in vertices):

            # finite region

            new_regions.append(vertices)

            continue



        # reconstruct a non-finite region

        ridges = all_ridges[p1]

        new_region = [v for v in vertices if v >= 0]



        for p2, v1, v2 in ridges:

            if v2 < 0:

                v1, v2 = v2, v1

            if v1 >= 0:

                # finite ridge: already in the region

                continue



            # Compute the missing endpoint of an infinite ridge



            t = vor.points[p2] - vor.points[p1] # tangent

            t /= np.linalg.norm(t)

            n = np.array([-t[1], t[0]])  # normal



            midpoint = vor.points[[p1, p2]].mean(axis=0)

            direction = np.sign(np.dot(midpoint - center, n)) * n

            far_point = vor.vertices[v2] + direction * radius



            new_region.append(len(new_vertices))

            new_vertices.append(far_point.tolist())



        # sort region counterclockwise

        vs = np.asarray([new_vertices[v] for v in new_region])

        c = vs.mean(axis=0)

        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])

        new_region = np.array(new_region)[np.argsort(angles)]



        # finish

        new_regions.append(new_region.tolist())



    return new_regions, np.asarray(new_vertices)
from scipy.spatial import Voronoi



plt.figure(figsize=(12, 8))

plt.suptitle("Sample plays, standardized, Offense moving left to right")

plt.xlabel("Distance from offensive team's own end zone")

plt.ylabel("Y coordinate")



sample_20171120000963 = train_1[train_1.PlayId==20171120000963].copy()

for gp,chance in sample_20171120000963.groupby('PlayId'):

    play_id = gp

    rusher = chance[chance.NflId==chance.NflIdRusher].iloc[0]

    offense = chance[chance.IsOnOffense]

    defense = chance[~chance.IsOnOffense]

    

    plt.subplot(1,1,1)

    i+=1

    

    x_min, x_max = chance.X_std.min()-2, chance.X_std.max()+2

    y_min, y_max = chance.Y_std.min()-2, chance.Y_std.max()+2

    #plt.xlim(8,50) # 特定

    plt.xlim(x_min,x_max)

    #plt.ylim(5,40) # 特定

    plt.ylim(y_min,y_max)

    #plt.plot([x_min,x_min,x_max,x_max,x_min],[y_min,y_max,y_max,y_min,y_min],c='black',linewidth=1.5)

    

    vor = Voronoi(np.array([[row.X_std,row.Y_std] for index, row in chance.iterrows()]))

    regions, vertices = voronoi_finite_polygons_2d(vor)

    for region in regions:

        polygon = vertices[region]

        plt.plot(*zip(*polygon),c='black',alpha=0.8)

    

    plt.scatter(offense.X_std,offense.Y_std,marker='o',c='green',s=55,alpha=0.5,label='OFFENSE')

    plt.scatter(defense.X_std,defense.Y_std,marker='o',c='red',s=55,alpha=0.5,label='DEFENSE')

    plt.scatter([rusher.X_std],[rusher.Y_std],marker='o',c='black',s=30,label='RUSHER')

    

    for line in range(10,130,10):

        plt.plot([line,line],[-100,100],c='silver',linewidth=0.8,linestyle='-')

    

    plt.plot([rusher.YardsFromOwnGoal,rusher.YardsFromOwnGoal],[-100,100],c='black',linewidth=1.5,linestyle=':')

    plt.plot([10,10],[-100,100],c='black',linewidth=2)

    plt.plot([110,110],[-100,100],c='black',linewidth=2)

    

    plt.title(play_id)

    plt.legend()



plt.show()
# plt.figure(figsize=(12, 8))

# plt.suptitle("Sample plays, standardized, Offense moving left to right")

# plt.xlabel("Distance from offensive team's own end zone")

# plt.ylabel("Y coordinate")



for gp,chance in sample_20171120000963.groupby('PlayId'):

    play_id = gp

    rusher = chance[chance.NflId==chance.NflIdRusher].iloc[0]

    offense = chance[chance.IsOnOffense]

    defense = chance[~chance.IsOnOffense]

    

#     plt.subplot(1,1,1)

    i+=1

    x_min, x_max = chance.X_std.min()-2, chance.X_std.max()+2

    y_min, y_max = chance.Y_std.min()-2, chance.Y_std.max()+2

#     plt.xlim(x_min,x_max)

#     plt.ylim(y_min,y_max)

    

    vor = Voronoi(np.array([[row.X_std,row.Y_std] for index, row in defense.append(rusher).iterrows()]))

    from scipy.spatial import voronoi_plot_2d

    fig = voronoi_plot_2d(vor,show_vertices=False,point_size=0.1,linewidth=2)

    fig.set_size_inches(12,8)

    #vor = Voronoi(np.array([[row.X_std,row.Y_std] for index, row in offense.iterrows()]))

    #vor = Voronoi(np.array([[row.X_std,row.Y_std] for index, row in chance.iterrows()]))

#     regions, vertices = voronoi_finite_polygons_2d(vor)

#     for region in regions:

#         polygon = vertices[region]

#         plt.plot(*zip(*polygon),c='black',alpha=0.8)

    

    #plt.scatter(offense.X_std,offense.Y_std,marker='o',c='green',s=55,alpha=0.5,label='OFFENSE')

    plt.scatter(defense.X_std,defense.Y_std,marker='o',c='red',s=55,alpha=0.5,label='DEFENSE')

    plt.scatter([rusher.X_std],[rusher.Y_std],marker='o',c='black',s=30,label='RUSHER')

    

    for line in range(10,130,10):

        plt.plot([line,line],[-100,100],c='silver',linewidth=0.8,linestyle='-')

    

    plt.plot([rusher.YardsFromOwnGoal,rusher.YardsFromOwnGoal],[-100,100],c='black',linewidth=1.5,linestyle=':')

    plt.plot([10,10],[-100,100],c='black',linewidth=2)

    plt.plot([110,110],[-100,100],c='black',linewidth=2)

    

#     plt.title(play_id)

#     plt.legend()



plt.show()
df_bc = train_1[train_1.IsBallCarrier][['DisplayName','PossessionTeam','PlayId','Dir','ToLeft','PlayDirection','IsOnOffense','X_std','Y_std','YardsFromOwnGoal','Down','Distance','Yards']].copy()
import seaborn as sns

plt.figure(figsize=(10, 5))

g = sns.FacetGrid(df_bc, col="ToLeft", height=4, aspect=.5)

g = g.map(plt.hist, "Dir")

plt.show()
df_bc['Dir_std_1'] = df_bc[['ToLeft','Dir']].apply(lambda row:row.Dir+360 if (row.ToLeft and row.Dir<90) else row.Dir, axis=1)

df_bc['Dir_std_1'] = df_bc[['ToLeft','Dir','Dir_std_1']].apply(lambda row:row.Dir-360 if ((not row.ToLeft) and row.Dir>270) else row.Dir_std_1, axis=1)
plt.figure(figsize=(10, 5))

g = sns.FacetGrid(df_bc, col="ToLeft", height=4, aspect=.5)

g = g.map(plt.hist, "Dir_std_1")

plt.show()
df_bc['Dir_std_2'] = df_bc[['ToLeft','Dir_std_1']].apply(lambda row:row.Dir_std_1-180 if row.ToLeft else row.Dir_std_1, axis=1)
plt.figure(figsize=(10, 5))

g = sns.FacetGrid(df_bc, col="ToLeft", height=4, aspect=.5)

g = g.map(plt.hist, "Dir_std_2")

plt.show()
df_bc[df_bc.PlayId==20170910001102][['PlayId','DisplayName','Dir','ToLeft','PlayDirection','X_std','Y_std','YardsFromOwnGoal','Dir_std_1','Dir_std_2','Yards']]
df_bc[df_bc.PlayId==20170910000081][['PlayId','DisplayName','Dir','ToLeft','PlayDirection','X_std','Y_std','YardsFromOwnGoal','Dir_std_1','Dir_std_2','Yards']]
train_2 = train_1.copy()

train_2['Dir_std_1'] = train_2[['ToLeft','Dir']].apply(lambda row:row.Dir+360 if (row.ToLeft and row.Dir<90) else row.Dir, axis=1)

train_2['Dir_std_1'] = train_2[['ToLeft','Dir','Dir_std_1']].apply(lambda row:row.Dir-360 if ((not row.ToLeft) and row.Dir>270) else row.Dir_std_1, axis=1)

train_2['Dir_std_2'] = train_2[['ToLeft','Dir_std_1']].apply(lambda row:row.Dir_std_1-180 if row.ToLeft else row.Dir_std_1, axis=1)

train_2['X_std_end'] = train_2[['Dir_std_2','X_std','S']].apply(lambda row:row.S*np.cos((90-row.Dir_std_2)*np.pi/180.)+row.X_std, axis=1)

train_2['Y_std_end'] = train_2[['Dir_std_2','Y_std','S']].apply(lambda row:row.S*np.sin((90-row.Dir_std_2)*np.pi/180.)+row.Y_std, axis=1)



sample_20170910001102 = train_2[train_2.PlayId==20170910001102].copy()

sample_20170910001102
plt.figure(figsize=(12, 8))

plt.suptitle("Playid:20170910001102")

plt.xlabel("Distance from offensive team's own end zone")

plt.ylabel("Y coordinate")



for gp,chance in sample_20170910001102.groupby('PlayId'):

    play_id = gp

    rusher = chance[chance.NflId==chance.NflIdRusher].iloc[0]

    offense = chance[chance.IsOnOffense]

    defense = chance[~chance.IsOnOffense]

    

    plt.subplot(1,1,1)

    i+=1

    

    x_min, x_max = chance.X_std.min()-5, chance.X_std.max()+5

    y_min, y_max = chance.Y_std.min()-5, chance.Y_std.max()+5

    plt.xlim(x_min,x_max)

    plt.ylim(y_min,y_max)

    

    plt.scatter(offense.X_std,offense.Y_std,marker='o',c='green',s=55,alpha=0.5,label='OFFENSE')

    plt.scatter(defense.X_std,defense.Y_std,marker='o',c='red',s=55,alpha=0.5,label='DEFENSE')

    plt.scatter([rusher.X_std],[rusher.Y_std],marker='o',c='black',s=30,label='RUSHER')

    

    for idx, row in chance.iterrows():

        _color='black' if row.IsBallCarrier else('green' if row.IsOnOffense else 'red')

        plt.arrow(row.X_std,row.Y_std,row.X_std_end-row.X_std,row.Y_std_end-row.Y_std,width=0.05,head_width=0.3,ec=_color,fc=_color)

    

    for line in range(10,130,10):

        plt.plot([line,line],[-100,100],c='silver',linewidth=0.8,linestyle='-')

    

    plt.plot([rusher.YardsFromOwnGoal,rusher.YardsFromOwnGoal],[-100,100],c='black',linewidth=1.5,linestyle=':')

    plt.plot([10,10],[-100,100],c='black',linewidth=2)

    plt.plot([110,110],[-100,100],c='black',linewidth=2)

    

    plt.title(play_id)

    plt.legend()



plt.show()
df_bc['IsSuccess'] = df_bc[['Down','Distance','Yards']].apply(lambda row: row.Yards>=(row.Distance/2) if row.Down in [1,2] else row.Yards>=row.Distance, axis=1)
df_bc['AngleRound'] = df_bc.Dir_std_2.apply(lambda ds2:np.round(ds2/15)*15)
plt.figure(figsize=(30, 70))

# plt.suptitle("Team senor(FAKE)")

# plt.xlabel("Team")

# plt.ylabel("Success rate")

plt.subplots_adjust(wspace=0.5, hspace=0.5)

i=1

for idx,row in df_bc[['PossessionTeam','AngleRound','IsSuccess']].groupby(['PossessionTeam']):

    plt.subplot(9,4,i)

    i+=1

    row.groupby('AngleRound').IsSuccess.mean().plot.bar()

    plt.title(row.PossessionTeam.iloc[0])

    plt.legend()

plt.show()