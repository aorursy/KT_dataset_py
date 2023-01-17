%load_ext autoreload
%autoreload 2
%matplotlib inline

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=DeprecationWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.spatial.distance import cdist, euclidean

df_cities = pd.read_csv('../input/cities.csv')
df_cities.tail()
def isPrime(n):
    #from https://stackoverflow.com/questions/4114167/checking-if-a-number-is-a-prime-number-in-python
    from itertools import count, islice
    return n > 1 and all(n%i for i in islice(count(2), int(np.sqrt(n)-1)))

def total_distance(path):
    #initialize counters
    prev_city_num = path[0]
    total_distance = 0
    step_num = 1
    for city_num in path[1:]:
        #get distance
        new_city = (df_cities.X[city_num], df_cities.Y[city_num])
        prev_city = (df_cities.X[prev_city_num], df_cities.Y[prev_city_num])
        distance = euclidean(new_city, prev_city)
        
        #check for 10% penalty
        if step_num % 10 == 0 and not isPrime(prev_city_num):
            distance = distance * 1.1
        
        total_distance += distance
        
        #increment counters
        prev_city_num = city_num
        step_num = step_num + 1
    return total_distance
%%time
def nearest_neighbour(cities):
    ids = cities.CityId.values[1:]
    xy = np.array([cities.X.values, cities.Y.values]).T[1:]
    path = [0,]
    while len(ids) > 0:
        last_x, last_y = cities.X[path[-1]], cities.Y[path[-1]]
        last_point = [[last_x, last_y]]
        nearest_index = cdist(last_point, xy, metric='euclidean').argmin()
        
        path.append(ids[nearest_index])
        ids = np.delete(ids, nearest_index, axis=0)
        xy = np.delete(xy, nearest_index, axis=0)
    path.append(0)
    return path

nnpath = nearest_neighbour(df_cities)

print(f"Total Distance: {total_distance(nnpath):.1f}")
df_path = pd.DataFrame({'CityId':nnpath}).merge(df_cities,how = 'left')
fig, ax = plt.subplots(figsize=(10,7))
ax.plot(df_path['X'], df_path['Y'])
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

n_clusters = 445
#more clusters == worse solution, fewer resources
#my experience:
#   - 1000 clusters = 1593976
#   - 445  clusters = 1585990
#   - 100  clusters = 1577067

num_iterations_per_solve = 250
#higher is gives a better solution, but I've set it low so this kernel solves quickly. 
#500 is a pretty good value in my exp
%%time
cities = df_cities.copy()
mclusterer = GaussianMixture(n_components=n_clusters)
cities['cluster'] = mclusterer.fit_predict(cities[['X', 'Y']].values)
cities['cluster_'] = cities['cluster'].astype(str) + "_"

plt.figure(figsize=(10,7))
clusters = sns.scatterplot(x=cities.X, y=cities.Y, alpha = 0.1, marker='.', hue=cities.cluster_, legend=False)
#plot number of points in each cluster
plt.suptitle('Points Per Cluster')
ax = cities.groupby('cluster')['CityId'].count().hist()
from scipy.spatial.distance import pdist, squareform
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

#%% functions
def create_mat(df):
    mat = pdist(locations, 'euclidean')
    return squareform(mat)

def create_distance_callback(dist_matrix):
    def distance_callback(from_node, to_node):
      return int(dist_matrix[from_node][to_node])
    return distance_callback

def optimize(df, startnode=None, stopnode=None):     
    num_nodes = df.shape[0]
    dist_matrix = create_mat(df)
    routemodel = pywrapcp.RoutingModel(num_nodes, 1, [startnode], [stopnode])
    
    dist_callback = create_distance_callback(dist_matrix)
    routemodel.SetArcCostEvaluatorOfAllVehicles(dist_callback)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.solution_limit = num_iterations_per_solve
    
    assignment = routemodel.SolveWithParameters(search_parameters)
    # print(f"Solved points:{num_nodes}, distance:{assignment.ObjectiveValue()}")
    return routemodel, assignment
    
def get_route(df, startnode, stopnode): 
    routing, assignment = optimize(df, int(startnode), int(stopnode))
    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        node_index= routing.IndexToNode(index)
        route.append(node_index)
        index = assignment.Value(routing.NextVar(index))      
    route.append(routing.IndexToNode(index))
    return route
%%time
nnode = int(cities.loc[0, 'cluster'])
center_df = cities.groupby('cluster')['X', 'Y'].agg('mean').reset_index()
locations = center_df[['X', 'Y']]
segment = get_route(locations, nnode, nnode)

fig, ax = plt.subplots()
clusters = sns.scatterplot(x=cities.X, y=cities.Y, alpha = 0.1, marker='.', hue=cities.cluster_, legend=False, ax=ax)
centers = sns.scatterplot(x=center_df.X, y=center_df.Y, marker='x', ax=ax)

ordered_clusters = center_df.loc[segment].reset_index(drop=True)
center_route = ax.plot(ordered_clusters.X, ordered_clusters.Y)
#get entries and exits for each cluster
last_exit_cityId = 0
entry_cityIds = []
exit_cityIds = []
for i,m in enumerate(ordered_clusters.cluster, 0):
    cluster = cities.loc[cities.cluster==m].reset_index(drop=True)
    if i < len(ordered_clusters)-1:
        next_center = ordered_clusters.loc[i+1, ['X', 'Y']]
    else:
        next_center = ordered_clusters.loc[0, ['X', 'Y']]
    
    #cluster entry is based on the nearest neighbor to the exit fo last cluster
    last_exit = cities.loc[last_exit_cityId, ['X', 'Y']]
    entry = cdist([last_exit], cluster[['X','Y']], metric='euclidean').argmin()
    entry_cityID = cluster.iloc[entry].CityId
    entry_cityIds.append(entry_cityID)
    
    #cluster exit is based on nearest neighbor to center of next cluster
    exit = cdist([next_center], cluster[['X','Y']], metric='euclidean').argmin()
    exit_cityID = cluster.iloc[exit].CityId
    exit_cityIds.append(exit_cityID)
    
    last_exit_cityId = exit_cityID

ordered_clusters['entry_cityId'] = entry_cityIds
ordered_clusters['exit_cityId'] = exit_cityIds
ordered_clusters.head()   
%%time 
seglist = []
#total_cities = cities.shape[0]
cities['cluster_index'] = cities.groupby('cluster').cumcount()

for i,m in enumerate(ordered_clusters.cluster):
    if i % 25 == 0: print(f"finished {i} clusters of {ordered_clusters.shape[0]-1}")
    district = cities[cities.cluster == m]
    
    clstart = ordered_clusters.loc[i, 'entry_cityId']
    nnode = district.loc[clstart, 'cluster_index']
    clstop = ordered_clusters.loc[i, 'exit_cityId']
    pnode = district.loc[clstop, 'cluster_index']
    locations = district[['X', 'Y']].values
    
    segnodes = get_route(locations, nnode, pnode) #output is type list
    ord_district =  district.iloc[segnodes]
    segment = ord_district.index.tolist()
    seglist.append(segment)

seglist.append([0])
ortools_path = np.concatenate(seglist)
print(f"Total Distance: {total_distance(ortools_path):.1f}")
df_ortools = df_cities.loc[ortools_path].drop_duplicates()
#df_ortools = pd.DataFrame({'CityId':ortools_path}).merge(df_cities,how = 'left')
fig, ax = plt.subplots(figsize=(10,7))
ax = ax.plot(df_ortools['X'], df_ortools['Y'])
from concorde.tsp import TSPSolver
from concorde.tests.data_utils import get_dataset_path
%%time
start_cities = pd.read_csv('../input/cities.csv') # 1533418.5
solver = TSPSolver.from_data(
    start_cities.X,
    start_cities.Y,
    norm="EUC_2D"
)
tour_data = solver.solve(time_bound = 60.0, verbose = True)
tour = start_cities.loc[tour_data.tour]
tour = tour.append({'CityId':0, 'X':316.836739, 'Y': 2202.340707}, ignore_index=True)
print(f"Total Distance: {total_distance(tour.CityId.tolist()):.1f}")
fig, ax = plt.subplots(figsize=(10,7))
ax = ax.plot(tour.X,tour.Y)
sub_df = pd.DataFrame(tour_data.tour,columns=['Path']).drop_duplicates()
sub_df=sub_df.append({'Path':0}, ignore_index=True)
sub_df.to_csv("submission.csv", index=False)
