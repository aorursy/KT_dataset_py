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
from kaggle.competitions import nflrush

from time import time

import matplotlib.pyplot as plt
#https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl

def in_hull(p, hull):

    """

    Test if points in `p` are in `hull`



    `p` should be a `NxK` coordinates of `N` points in `K` dimensions

    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 

    coordinates of `M` points in `K`dimensions for which Delaunay triangulation

    will be computed

    """

    from scipy.spatial import Delaunay

    if not isinstance(hull,Delaunay):

        hull = Delaunay(hull)



    return hull.find_simplex(p)>=0

def plot_in_hull(p, hull):

    """

    plot relative to `in_hull` for 2d data

    """

    import matplotlib.pyplot as plt

    from matplotlib.collections import PolyCollection, LineCollection



    from scipy.spatial import Delaunay

    if not isinstance(hull,Delaunay):

        hull = Delaunay(hull)



    # plot triangulation

    poly = PolyCollection(hull.points[hull.vertices], facecolors='w', edgecolors='b')

    plt.clf()

    plt.title('in hull')

    plt.gca().add_collection(poly)

    #plt.plot(hull.points[:,0], hull.points[:,1], 'o', hold=1)

    plt.plot(hull.points[:,0], hull.points[:,1], 'o')





    # plot the convex hull

    edges = set()

    edge_points = []



    def add_edge(i, j):

        """Add a line between the i-th and j-th points, if not in the list already"""

        if (i, j) in edges or (j, i) in edges:

            # already added

            return

        edges.add( (i, j) )

        edge_points.append(hull.points[ [i, j] ])



    for ia, ib in hull.convex_hull:

        add_edge(ia, ib)



    lines = LineCollection(edge_points, color='g')

    plt.gca().add_collection(lines)

    



    # plot tested points `p` - black are inside hull, red outside

    inside = in_hull(p,hull)

    

    plt.plot(p[ inside,0],p[ inside,1],'.k')

    plt.plot(p[inside==False,0],p[inside==False,1],'.r')

    plt.show()    
tested = np.random.rand(20,2)

cloud  = np.random.rand(50,2)



print (in_hull(tested,cloud))

''' Voronoi stuff '''

from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull



def normalize_coordinates(points):

    points = np.array(points)

    #extra_space = 5

    extra_space = EXTRA_SPACE + 0.1

    xmin, xmax, ymin, ymax = points[:,0].min()-extra_space, points[:,0].max()+extra_space, points[:,1].min()-1, points[:,1].max()+1

    #override boundaries to make area consistent

    ymin, ymax = 0, 53.333 # some plays runner go to sides 

    xmax = xmin + 25 # consider only players within 25 yards of runner

    ###################

    # delete points outside boundaries

    points = points[points[:,0]<xmax]

    points = points[points[:,1]<ymax]

    

    ###################

    xrange = xmax - xmin

    yrange = ymax - ymin

    norm = np.zeros_like(points)

    norm[:,0] = (points[:,0] - xmin) / xrange

    norm[:,1] = (points[:,1] - ymin) / yrange

    return norm



def mirror(points):

    # a hacky method to bound voronoi in a square area inspired from https://stackoverflow.com/questions/28665491/getting-a-bounded-polygon-coordinates-from-voronoi-cells

    # this is simplified to fit NFL's problem

    left = points.copy()

    left[:,0] = left[:,0] * -1

    right = left.copy()

    right[:,0] = right[:,0] + 2

    up = points.copy()

    up[:,1] = 2- up[:,1]

    down = points.copy()

    down[:,1] = down[:,1] * -1

    

    mirrored = np.vstack([points, left, right, up, down])

    return mirrored





def voronoi_area(v):

    #v = Voronoi(points)

    vol = np.zeros(v.npoints)

    for i, reg_num in enumerate(v.point_region[:1]):

        indices = v.regions[reg_num]

        if -1 in indices: 

            vol[i] = np.inf

        else:

            vol[i] = ConvexHull(v.vertices[indices]).area

    return vol[:1]





def compute_area(check):

    playid = check['PlayId'][0]

    check = check.sort_values('IsBallCarrier', ascending=False).reset_index(drop=True)# make sure runner is at top 

    points = check[['X_std', 'Y_std']] # get all player's coordinate

    norm = normalize_coordinates(points)

    norm_mirror = mirror(norm)

    vor = Voronoi(norm_mirror, qhull_options='Qbb Qc Qx')

    vols_incl_offense = voronoi_area(vor) # first calculation is if we include offense



    runnervdef = check[(check['IsBallCarrier']==True) | (check['IsOnOffense']==False)] # consider ball carrier only and defense team

    points = runnervdef[['X_std', 'Y_std']]

    norm = normalize_coordinates(points)

    norm_mirror = mirror(norm)

    vor = Voronoi(norm_mirror, qhull_options='Qbb Qc Qx')

    vols = voronoi_area(vor)# This is the area if we consider only ball carrier with remaining defense

    return [vols[0], vols_incl_offense[0]]
def initial_features(train):

    # code taken from https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python

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

    

    train['TeamOnOffense'] = "home"

    train.loc[train.PossessionTeam != train.HomeTeamAbbr, 'TeamOnOffense'] = "away"

    train['IsOnOffense'] = train.Team == train.TeamOnOffense # Is player on offense?

    train['YardLine_std'] = 100 - train.YardLine

    train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  

              'YardLine_std'

             ] = train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  

              'YardLine']

    train['X_std'] = train.X #(range is 0 to 120 which covers the 100 yards and 10 yards each sides)

    train.loc[train.ToLeft, 'X_std'] = 120 - train.loc[train.ToLeft, 'X'] 

    train['X_std'] = train['X_std'] - 10 #-10 to align with yardline range(0-100)

    

    train['Y_std'] = train.Y

    train.loc[train.ToLeft, 'Y_std'] = 160/3 - train.loc[train.ToLeft, 'Y'] 

    

    return train

train = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv")



train = initial_features(train)

start = time()

EXTRA_SPACE=5

plays = train['PlayId'].unique()



check = train[train['PlayId']== plays[402]]

check = check.sort_values(['IsBallCarrier','IsOnOffense'], ascending=False).reset_index(drop=True)



runnervdef = check[(check['IsBallCarrier']==True) | (check['IsOnOffense']==False)]



pointsorig = check[['X_std', 'Y_std']]



points = runnervdef[['X_std', 'Y_std']]

norm = normalize_coordinates(points)

norm.shape

norm_mirror = mirror(norm)

vorraw = Voronoi(pointsorig, qhull_options='Qbb Qc Qx')



vor = Voronoi(norm_mirror, qhull_options='Qbb Qc Qx')

vols = voronoi_area(vor)





print('compute ',time()-start)



fig = voronoi_plot_2d(vorraw)

fig.set_figheight(5.3*2)

fig.set_figwidth(2.5*2)

plt.scatter(check.loc[check['IsOnOffense']==True, ['X_std']], check.loc[check['IsOnOffense']==True, [ 'Y_std']])







fig = voronoi_plot_2d(vor)

fig.set_figheight(5.3*2)

fig.set_figwidth(2.5*2)

plt.scatter(norm[0,0], norm[0,1])

plt.xlim(0, 1)

plt.ylim(0, 1)

plt.show()



print(time()-start)



#vor.npoints

#list(zip(vor.point_region,vor.regions, vols))

#print( len(vor.point_region), len(vor.regions) )

#vor.points

#vor.ridge_points



'''

points	(ndarray of double, shape (npoints, ndim)) Coordinates of input points.

vertices	(ndarray of double, shape (nvertices, ndim)) Coordinates of the Voronoi vertices.

ridge_points	(ndarray of ints, shape (nridges, 2)) Indices of the points between which each Voronoi ridge lies.

ridge_vertices	(list of list of ints, shape (nridges, *)) Indices of the Voronoi vertices forming each Voronoi ridge.

regions	(list of list of ints, shape (nregions, *)) Indices of the Voronoi vertices forming each Voronoi region. -1 indicates vertex outside the Voronoi diagram.

point_region	(list of ints, shape (npoints)) Index of the Voronoi region for each input point. If qhull option “Qc” was not specified, the list will contain -1 for points that are not associated with a Voronoi region.

'''
i=408
#convex hull and checking if a point inside an area

i = i + 1

check = train[train['PlayId']== plays[i]]

check = check.sort_values(['IsBallCarrier','IsOnOffense'], ascending=False).reset_index(drop=True)



runnervdef = check[(check['IsBallCarrier']==True) | (check['IsOnOffense']==False)]



pointsorig = check[['X_std', 'Y_std']]



print( 'nr defenders inside offense ',  np.sum(in_hull(pointsorig.values[11:],pointsorig.values[1:11]) ) , ' Yards ', check.loc[0,'Yards'] )

print( 'nr offense inside defenders ',  np.sum(in_hull(pointsorig.values[1:11],pointsorig.values[11:]) ) , ' Yards ', check.loc[0,'Yards'] )

print( 'is runner inside defense ',  np.sum(in_hull(pointsorig.values[0],pointsorig.values[11:]) ) , ' Yards ', check.loc[0,'Yards'] )

print( 'is runner inside offense ',  np.sum(in_hull(pointsorig.values[0],pointsorig.values[1:11]) ) , ' Yards ', check.loc[0,'Yards'] )



plot_in_hull(pointsorig.values[11:],pointsorig.values[1:11])

plot_in_hull(pointsorig.values[1:11],pointsorig.values[11:])

plot_in_hull(pointsorig.values[0],pointsorig.values[11:])

plot_in_hull(pointsorig.values[0],pointsorig.values[1:11])
from itertools import chain
start =time()

voronoi_cols = []

for i in range(1):

    EXTRA_SPACE = i

    result = train.groupby('PlayId')['PlayId','IsBallCarrier', 'X_std', 'Y_std', 'IsOnOffense'].apply(compute_area)

    result = result.reset_index()

    result = pd.concat([result['PlayId'],  pd.DataFrame(list(chain(result.iloc[:,1].values)))  ], axis=1  ) 

    result.columns = ['PlayId', 'Runner_area' + str(EXTRA_SPACE), 'Runner_area_w_offense'+str(EXTRA_SPACE)]

    voronoi_cols = voronoi_cols + ['Runner_area' + str(EXTRA_SPACE), 'Runner_area_w_offense'+str(EXTRA_SPACE)]

    train = train.merge(result, how='left', on='PlayId', suffixes=('',''))

print(time()-start,' seconds')



result.head(10)
plt.scatter(train[voronoi_cols[0]], train['Yards'])
np.corrcoef(train[voronoi_cols[0]], train['Yards'])
plt.scatter(train[voronoi_cols[1]], train['Yards'])
np.corrcoef(train[voronoi_cols[1]], train['Yards'])
train[['Yards']+ voronoi_cols].corr()