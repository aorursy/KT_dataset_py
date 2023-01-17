import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import KMeans

from sklearn.utils.testing import ignore_warnings

from sklearn.exceptions import ConvergenceWarning



import datetime

import math

%matplotlib inline

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [25, 15]

import seaborn as sns

sns.set()

import networkx as nx

import csv



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
trips = pd.read_csv("/kaggle/input/mrt_trips_sampled.csv")

print(trips.isna().sum())

trips
trips.applymap(lambda x : x.str.strip() if type(x) == 'str' else x)
print(trips.groupby(['destination','destination_tm','origin','origin_tm']).size().reset_index(name='count')['count'].value_counts())



def remove_repeats(trips):

    """

    scales df by one third

    """

    trips = trips.groupby(['destination','destination_tm','origin','origin_tm']).size().reset_index(name='count')

    trips = pd.concat([trips,trips[trips['count']>3],trips[trips['count']==9]])

    return trips.reset_index().drop(columns=['count','index'])

trips = remove_repeats(trips)



print(trips.groupby(['destination','destination_tm','origin','origin_tm']).size().reset_index(name='count')['count'].value_counts())
print('There are {} and {} unique destination and origin stations respectively.'.format(trips['destination'].nunique(),trips['origin'].nunique()))

fig, axes = plt.subplots(nrows=2, ncols=2,figsize=[15, 7],tight_layout=True)

trips['destination'].value_counts(ascending=True)[-20:].plot(kind='barh',title='Top 20 Destinations by frequency',ax=axes[0,0])

trips['destination'].value_counts(ascending=True)[:20].plot(kind='barh',title='Bottom 20 Destinations by frequency',ax=axes[0,1])

trips['origin'].value_counts(ascending=True)[-20:].plot(kind='barh',title='Top 20 Origins by frequency',ax=axes[1,0])

trips['origin'].value_counts(ascending=True)[:20].plot(kind='barh',title='Bottom 20 Origins by frequency',ax=axes[1,1])

fig.show()
print('The top {} origin stations account for {:.2f}% of the trips.'.format(20,trips['origin'].value_counts().values[:20].sum() / len(trips)*100))

print('The top {} destination stations account for {:.2f}% of the trips.'.format(20,trips['destination'].value_counts().values[:20].sum() / len(trips)*100))
n_trips = 50

y1 = trips['destination'].value_counts().values[:n_trips] 

y2 = trips['origin'].value_counts().values[:n_trips]

x = np.arange(n_trips)

plt.bar(x,y1,color='r',label='Destination',width=.45)

plt.bar(x+.5,y2,color='b',label='Origin',width=.45)

plt.title('Frequency distribution of top {} stations'.format(n_trips))

plt.legend()

plt.xlabel('Top nth station by frequency')

plt.ylabel('Number of trips')

plt.figure(figsize=(8,5))

plt.show()
stations = trips[["origin", "destination"]].values.ravel()

stations = pd.unique(stations)

stations.sort()

print('There are',len(stations),'unique stations.')



origin_stations = trips['origin'].unique()

dest_stations = trips['destination'].unique()

o_unique = []

d_unique = []

for station in stations:

    if station not in origin_stations: o_unique.append(station)

    if station not in dest_stations: d_unique.append(station)

print('Stations with trips only starting from in dataset:',o_unique)

print('Stations with trips only ending from in dataset:',d_unique)
d_time = pd.to_datetime(trips['destination_tm'])

o_time = pd.to_datetime(trips['origin_tm'])

print('Destination timestamp ranges from',d_time.min(),'to',d_time.max())

print('Origin timestamp ranges from',o_time.min(),'to',o_time.max())
# test = pd.to_datetime(trips['destination_tm'],format='%H%M%S')

test =trips['destination_tm'].astype('datetime64[ns]')

test
d_time.hist(bins=100,figsize= [10, 7],label='Histogram of {} trips, ranging from {} to {}'.format(len(d_time),d_time.min(),d_time.max()))
def add_duration(trips): 

    d_time = pd.to_datetime(trips['destination_tm'])

    o_time = pd.to_datetime(trips['origin_tm'])

    diff = d_time - o_time

    assert len(trips[diff <= datetime.timedelta(0)]) == 0 # Sanity check that all destination times recorded are after origin times

    diff = diff.astype('timedelta64[s]')

    trips.loc[:,'duration'] = diff

    return trips



trips = add_duration(trips)

trips
dest_ori_counts = trips.groupby(['destination','origin']).size().reset_index(name='freq')

max_d, max_o = dest_ori_counts.iloc[dest_ori_counts.freq.idxmax()].destination, dest_ori_counts.iloc[dest_ori_counts.freq.idxmax()].origin

trips[(trips.destination == max_d) & (trips.origin == max_o)].duration.hist(bins=100,figsize= [10, 7],label='{} trips from {} to {}'.format(dest_ori_counts.freq.max(),max_o,max_d))
def filter_outlier(s):

    """ 

    Filters outliers in given series. Outliers defined as 1.5 * interquartile range smaller than Q1 or larger than Q3 (i.e. mild outliers). We also filter for values appearing less than 3 times. 

    Inspired from https://datascience.stackexchange.com/questions/33632/remove-local-outliers-from-dataframe-using-pandas

    

    Parameters:

    s (pandas series): Input series to identify outliers.

    

    Returns:

    boolean of whether we keep a value

    """

    if s.size < 2:

        return s.between(0,0)

    Q1 = s.quantile(0.25)

    Q3 = s.quantile(0.75)

    IQR = Q3 - Q1

    LOW = Q1 - 1.5 * IQR

    HIGH = Q3 + 1.5 * IQR

    return s.between(LOW, HIGH)
# Filters outliers in given series. Outliers defined as 3 * interquartile range smaller than Q1 or larger than Q3 (i.e. extreme outliers)

Q1 = trips['duration'].quantile(0.25)

Q3 = trips['duration'].quantile(0.75)

IQR = Q3 - Q1

LOW = Q1 - 3 * IQR

HIGH = Q3 + 3 * IQR



df = trips[trips['duration'].apply(lambda x: (LOW <= x <= HIGH))]

print('We removed {} trips.'.format(len(trips) - len(df)))
d_time = pd.to_datetime(df['destination_tm'])

o_time = pd.to_datetime(df['origin_tm'])

print('Destination timestamp ranges from',d_time.min(),'to',d_time.max())

print('Origin timestamp ranges from',o_time.min(),'to',o_time.max())

df.loc[d_time.idxmax()]
df = df[df.groupby(['origin','destination']).duration.apply(filter_outlier)]

print('We removed {} trips.'.format(len(trips) - len(df)))

df
plt.figure(figsize=(8,5))

dest_ori_counts = trips.groupby(['destination','origin']).size().reset_index(name='freq')

max_d, max_o = dest_ori_counts.iloc[dest_ori_counts.freq.idxmax()].destination, dest_ori_counts.iloc[dest_ori_counts.freq.idxmax()].origin

ax = sns.distplot(trips[(trips.destination == max_d) & (trips.origin == max_o)].duration/60)

plt.title('Original dataset: {} trips from {} to {}'.format(dest_ori_counts.freq.max(),max_o,max_d))

plt.xlabel('Duration in minutes')

plt.ylabel('Proportion of trips')

plt.show()
plt.figure(figsize=(8,5))

dest_ori_counts = df.groupby(['destination','origin']).size().reset_index(name='freq')

max_d, max_o = dest_ori_counts.iloc[dest_ori_counts.freq.idxmax()].destination, dest_ori_counts.iloc[dest_ori_counts.freq.idxmax()].origin

ax = sns.distplot(df[(df.destination == max_d) & (df.origin == max_o)].duration/60)

plt.title('After removing outliers: {} trips from {} to {}'.format(dest_ori_counts.freq.max(),max_o,max_d))

plt.xlabel('Duration in minutes')

plt.ylabel('Proportion of trips')

plt.show()
plt.figure(figsize=(8,5))

ax = sns.distplot(trips.duration/60)

plt.title('Original dataset of {:,} trips'.format(len(trips)))

plt.xlabel('Duration in minutes')

plt.ylabel('Proportion of trips')

plt.show()
plt.figure(figsize=(8,5))

ax = sns.distplot(df.duration/60)

plt.title('Filtered dataset of {:,} trips'.format(len(df)))

plt.xlabel('Duration in minutes')

plt.ylabel('Proportion of trips')

plt.show()
def get_trip_pairs(trips,direction='o2d',quantile=0.5):

    """

    modes: o2d, d2o, both, default=o2c, else=both

    undirected trip pairs: combine origin-destination pairs with destination-origin pairs

    d2o: reverse destination with origin columns

    quantile: Group by this quantile level

    """

    trips = trips[trips['origin'] != trips['destination']]

    if direction == 'o2d':

        return trips.groupby(['origin', 'destination']).duration.quantile(quantile).reset_index()

    else:

        trips_rev = trips.rename(columns={'origin':'destination','destination':'origin'})

        trips_rev = trips_rev.reindex(columns=['origin','destination','duration'])

        if direction == 'd2o': return trips_rev

        trip_pairs_combined = pd.concat([trips.reindex(columns=['origin','destination','duration']),trips_rev], ignore_index=True)

        trip_pairs_combined = trip_pairs_combined.groupby(['origin', 'destination']).duration.quantile(quantile).reset_index()

        return trip_pairs_combined
def get_nearest_n(trip_pairs,n):

    """

    input: trip_pairs = df of all connections, n = number of smallest trips

    output: df of nearest n stations for each unique origin station

    """

    adjacency = trip_pairs.groupby(['origin']).duration.nsmallest(n).reset_index()

    adjacency = adjacency.rename(columns={'level_1':'destination'})

    adj_dest = trip_pairs['destination'][adjacency['destination']].reset_index()

    return pd.DataFrame({'origin':adjacency['origin'],'destination':adj_dest['destination'],'duration':adjacency['duration']})
def filter_triangle_inequality(adjacency):

    """

    generates new adjacency df, filtering using triangle inequality rule

    """

    filtered_adj = {'origin':[],'destination':[],'duration':[]}

    for o,d,dur in zip(adjacency['origin'],adjacency['destination'],adjacency['duration']):

        candidate = adjacency[(adjacency['origin'] == o) & (adjacency['duration'] < dur)]

        if len(candidate) == 1:

            # directly add the nearest neighbour

            filtered_adj['origin'].append(o)

            filtered_adj['destination'].append(d)

            filtered_adj['duration'].append(dur)



        else:

            # compare all other pairs, add pair if for all other stations 

            add = True

            for dest in candidate['destination']:

                if adjacency[(adjacency['origin'] == dest) & (adjacency['destination'] == d)].empty: 

                    add = False

                    break

                a = adjacency[(adjacency['origin'] == o) & (adjacency['destination'] == dest)].duration.values[0]

                b = adjacency[(adjacency['origin'] == dest) & (adjacency['destination'] == d)].duration.values[0]

                if a+b < dur + 10:

                    add = False

                    break

            if add: 

                filtered_adj['origin'].append(o)

                filtered_adj['destination'].append(d)

                filtered_adj['duration'].append(dur)



    return pd.DataFrame.from_dict(filtered_adj)
def match_bidirectional(o2d,d2o):

    """

    default: finds matches in o2d and d2o adjacency lists, returns df of smaller durations

    """

    filtered_adj = {'origin':[],'destination':[],'duration':[]}

    for o,d,dur in zip(o2d['origin'],o2d['destination'],o2d['duration']):

        if not d2o[(d2o['origin'] == o) & (d2o['destination'] == d)].empty:

            filtered_adj['origin'].append(o)

            filtered_adj['destination'].append(d)

            d2o_dur = d2o[(d2o['origin'] == o) & (d2o['destination'] == d)].duration.values[0]

            filtered_adj['duration'].append(min(dur,d2o_dur))

    return pd.DataFrame.from_dict(filtered_adj)
def get_adjacency(trips,triangle=False,direction=1, match=False, n=2, quantile=0.5):

    """

    Choose adjacency list generating algorithm.

    nn: Selects nearest-2-neighbour as edges. Else, use triangle inequality to generate edges.

    direction: 1 or 2 to denote if we concat d2o into original o2d table to generate direction agnostic graph.

    match: If true, generates o2d edges and d2o edges, and returns intersection of both sets.

    quantile: Select quantile level for groupby operation for trips to trip_pairs

    

    Returns adjacency list from trips dataset

    """

    if match: 

        o2d, d2o = get_trip_pairs(trips, quantile=quantile), get_trip_pairs(trips,direction='d2o',quantile=quantile)

        o2d, d2o = get_nearest_n(o2d,5), get_nearest_n(d2o,5)

        if triangle: o2d, d2o = filter_triangle_inequality(o2d), filter_triangle_inequality(d2o)

        return match_bidirectional(o2d,d2o)

    else:

        trip_pairs = get_trip_pairs(trips) if direction == 1 else get_trip_pairs(trips,direction='both')

        trip_pairs = get_nearest_n(trip_pairs,n)

        return trip_pairs if triangle else filter_triangle_inequality(trip_pairs)
# adjacency = get_adjacency(df,triangle=False,n=2,direction=1, match=False)  # nearest 2

adjacency = get_adjacency(df,triangle=False,n=2,direction=2, match=False)  # bidirectional, unmatched

# adjacency = get_adjacency(df,triangle=True,n=5,direction=2, match=True)  # bidirectional, matched, triangle ineq. filtered

adjacency
derived_network = nx.Graph(name='Derived')

for origin, dest, weight in zip(adjacency['origin'], adjacency['destination'], adjacency['duration']):

    derived_network.add_edge(origin, dest, weight=weight)

print(nx.info(derived_network))

nx.draw_networkx(derived_network,font_size=10,node_size=800)
node_labels = pd.read_csv('/kaggle/input/mrt_network_nodes.csv')

node_labels['Node'] = node_labels['Node'].str.strip()

node_labels['Name'] = node_labels['Name'].str.strip()

node_labels
with open('/kaggle/input/mrt_network_edges.csv', 'r') as edgecsv:

    edgereader = csv.reader(edgecsv)

    next(edgereader)

    edges = []

    nodes = []

    for e in edgereader:

        name = e[0].strip()

        nodes.append(name)

        for dest in e[1][2:-1].split(','):

            edges.append((name,dest.strip()))

# print(edges)

# print(nodes)
actual_graph = nx.Graph(name='Actual')

actual_graph.add_nodes_from(nodes)

actual_graph.add_edges_from(edges)

print(nx.info(actual_graph))
def get_colors(G):

    color_map = {'NS':'#c3301d','EW':'#419246','CG':'#419246','NE':'#8c3aaf','DT':'#1e59b2','CC':'#f2a13e','CE':'#f2a13e'}

    return [color_map[node[0:2]] if node[0:2] in color_map else '#999999' for node in G]



def get_pos(nodes):

    """

    Unused, as this does not guarantee the positions between lines are fixed (unless we order them).

    """

    pos = {}

    added = {}

    y = 0

    for node in nodes:

        if node[:2] not in added:

            added[node[:2]] = y

            y += 1

        pos[node] = (int(node[2:]),added[node[:2]]) if node[2:].isdigit() else (1,added[node[:2]])

    return pos



def get_fixed_pos():

    """

    Hard-coded positions for neater visualizations

    """

    pos = {}

    j = 0

    for i in range(28):

        pos['NS'+str(i+1)] = (i,j+i/50)

    j += 1

    for i in range(29):

        pos['EW'+str(i+1)] = (i,j+i/50)

    j += 1

    for i in range(2):

        pos['CG'+str(i+1)] = (i,j+i/50)

    j += 1

    for i in range(17):

        pos['NE'+str(i+1)] = (i,j+i/50)

    j += 1

    for i in range(29):

        pos['CC'+str(i+1)] = (i,j+i/50)

    j += 1

    for i in range(2):

        pos['CE'+str(i+1)] = (i,j+i/50)

    j += 1

    for i in range(19):

        pos['DT'+str(i+1)] = (i,j+i/50)

    j += 1

    for i in range(14):

        pos['BP'+str(i+1)] = (i,j+i/50)

    j += 1

    for i in range(5):

        pos['SE'+str(i+1)] = (i,j+i/50)

    pos['STC'] = (6,j)

    j += 1

    for i in range(8):

        pos['SW'+str(i+1)] = (i,j+i/50)

    j += 1

    for i in range(7):

        pos['PE'+str(i+1)] = (i,j+i/50)

    pos['PTC'] = (8,j)

    pos['PW1'] = (9,j)

    pos['PW5'] = (10,j)

    pos['PW6'] = (11,j)

    pos['PW7'] = (12,j)

    return pos



fixed_pos = get_fixed_pos()
def name_to_labels(adjacency, node_labels):

    """

    in-place modification of adjacency df, adds two new columns origin_code and destination_code

    input: names = pandas df of adjacency to concvert origin & destination names; node_labels = pandas df to map names to mrt code

    output: original df

    """

    ans = [[] for _ in range(2)]

    skipped = [[] for _ in range(2)]

    # mapping of dirty data, i.e. name of interchanges: City Hall, Jurong East, Promenade, Raffles Place, Sengkang, Punggol, Dhoby Ghaut CCL

    name_dict = {'Bayfront CCL':'CE1','Bayfront DTL':'DT16','Bishan NSEW': 'NS17', 'Bugis NSEW': 'EW12', 'Bukit Panjang BPLRT': 'BP6', 'Buona Vista NSEW': 'EW21', 'Choa Chu Kang': 'NS4', 'Choa Chu Kang BPLRT': 'BP1', 'Dhoby Ghaut NSEW': 'NS24', 'Expo NSEW': 'CG1','Marina Bay': 'NS27','Marina Bay CCL': 'CE2', 'Newton NSEW': 'NS21', 'Outram Park NSEW': 'EW16', 'Paya Lebar NSEW': 'EW8', 'Tampines NSEW': 'EW2'}

    for i in range(2):

        names = adjacency['origin'] if i == 0 else adjacency['destination']

        for name in names:

            match = node_labels['Node'][node_labels['Name'] == name]

            if len(match) == 1:

                ans[i].append(match.values[0])

            else:

                first, end = ' '.join(name.split()[:-1]), name.split()[-1]

                if end[-1] == 'L':  # CCL, DTL

                    if first == 'Bayfront' or first == 'Marina Bay': ans[i].append(name_dict[name])

                    else: 

                        for n in node_labels['Node'][node_labels['Name'] == first]:

                            if n[:2] == end[:-1]: 

                                ans[i].append(n)

                                continue

                else:

                    if name not in name_dict:

                        skipped[i].append(len(ans[i]))

                        ans[i].append(name)

                    else:

                        ans[i].append(name_dict[name])

    for i in range(2):

        for idx in skipped[i]:

            name = ans[i][idx]

            other = ans[abs(i-1)][idx]

            changed=False

            if other[:1].isupper():

                for n in node_labels['Node'][node_labels['Name'] == name]:

                    if n[:1] == other[:1]:

                        changed=True

                        ans[i][idx] = n

                if not changed: 

                    default = node_labels['Node'][node_labels['Name'] == name].values[0]

                    ans[i][idx] = default

                    print(name,other,default)

    adjacency['origin_code'], adjacency['destination_code'] = ans[0], ans[1]

    return adjacency
derived_network = nx.Graph(name='Derived')

name_to_labels(adjacency, node_labels)

derived_network.add_nodes_from(adjacency['origin_code'])

for origin, dest, weight in zip(adjacency['origin_code'],adjacency['destination_code'],adjacency['duration']):

    derived_network.add_edge(origin, dest, weight=weight)



# manually add interchanges    

# for edge in edges:

#     code0, code1 = code_to_name(edge[0]), code_o_name(edge[1])

#     if code0 in interchanges and code1 in interchanges and code0 == code1:

#         print('interchange',code0,code1)

#         derived_network.add_edge(origin, dest)

        

pos = get_pos(adjacency['origin_code'])

print(nx.info(derived_network))
nx.draw_networkx(actual_graph,node_color=get_colors(actual_graph),font_size=10,node_size=800, pos=fixed_pos)
nx.draw_networkx(derived_network,node_color=get_colors(derived_network),font_size=10,node_size=800, pos=fixed_pos)
def similarity(g, h, mode='jaccard', nodes=False):

    """

    measure: jaccard, precision, recall, f1

    g: ground truth network, h: derived network

    returns float value of similarity between two graphs range [0,1] 

    """

    tp, tn, fp, fn = [], [], [], []

    if nodes:

        g, h, gh = set(g.nodes()), set(h.nodes()), nx.compose(g,h)

        for node in gh.nodes():

            if node in g and node in h:

                tp.append(node)

            elif node in g and node not in h:

                fn.append(node)

            elif node not in g and node in h:

                fp.append(node)

            elif node not in g and node not in h:

                tn.append(node)

    else:

        for edge in nx.compose(g,h).edges():

            if g.has_edge(*edge) and h.has_edge(*edge):

                tp.append(edge)

            elif g.has_edge(*edge) and not h.has_edge(*edge):

                fn.append(edge)

            elif not g.has_edge(*edge) and h.has_edge(*edge):

                fp.append(edge)

            elif not g.has_edge(*edge) and not h.has_edge(*edge):

                tn.append(edge)

        g, h = g.edges(), h.edges()



    assert len(tn) == 0, 'True negative set: {}'.format(tn)

    

    p = len(tp) / (len(tp) + len(fp))

    r = len(tp) / (len(tp) + len(fn))

    

    if mode == 'jaccard':

        return len(tp) / (len(g) + len(h) - len(tp))

    elif mode == 'precision':

        return p

    elif mode == 'recall':

        return r

    elif mode == 'f1':

        return 2 * p * r / (p + r)

    else:

        raise Exception(mode,' mode not supported.')

    
def plot_comparison(G,H, pos=None):

    GH = nx.compose(G,H)

    

    # set edge colors

    edge_colors = dict()

    for edge in GH.edges():

        if G.has_edge(*edge):

            if H.has_edge(*edge):

                edge_colors[edge] = 'green'

                continue

            edge_colors[edge] = 'blue'

        elif H.has_edge(*edge):

            edge_colors[edge] = 'red'



    # set node colors

    G_nodes = set(G.nodes())

    H_nodes = set(H.nodes())

    node_colors = []

    for node in GH.nodes():

        if node in G_nodes:

            if node in H_nodes:

                node_colors.append('green')

                continue

            node_colors.append('blue')

        if node in H_nodes:

            node_colors.append('red')



    nx.draw_networkx(GH, pos=pos, 

            nodelist=GH.nodes(),

            node_color=node_colors,

            edgelist=edge_colors.keys(), 

            edge_color=edge_colors.values(),

            node_size=800,

            width=5,alpha=0.6,

            with_labels=True)
print(nx.info(actual_graph))

print(nx.info(derived_network))

print('Nodes:')

print('Jaccard Similarity: {:3f}'.format(similarity(actual_graph, derived_network, mode='jaccard', nodes=True)))

print('Precision: {:3f}'.format(similarity(actual_graph, derived_network, mode='precision', nodes=True)))

print('Recall: {:3f}'.format(similarity(actual_graph, derived_network, mode='recall', nodes=True)))

print('F1: {:3f}'.format(similarity(actual_graph, derived_network, mode='f1', nodes=True)))

print('Edges:')

print('Jaccard Similarity: {:3f}'.format(similarity(actual_graph, derived_network, mode='jaccard')))

print('Precision: {:3f}'.format(similarity(actual_graph, derived_network, mode='precision')))

print('Recall: {:3f}'.format(similarity(actual_graph, derived_network, mode='recall')))

print('F1: {:3f}'.format(similarity(actual_graph, derived_network, mode='f1')))



plot_comparison(actual_graph,derived_network,pos=fixed_pos)
def remove_block_letters(s):

    """

    removes block letter MRT codes (e.g. CCL) from pandas series. Use as filter function to change row values.

    """

    ret = []

    for substr in s.split():

        if not substr.isupper(): ret.append(substr)

    return ' '.join(ret)



def code_to_name(s):

    return node_labels[node_labels['Node'] == s]['Name'].values[0]
from collections import Counter

name_count = Counter()

for name in df['origin'].unique():

    cleaned = remove_block_letters(name)

    name_count[cleaned] += 1

interchanges = {name for name in name_count if name_count[name]>1}

print('Interchange stations:',interchanges)



actual_named_graph = nx.Graph(name='Actual-named')

actual_named_graph.add_edges_from([(code_to_name(e1), code_to_name(e2)) for (e1, e2) in edges])



derived_named_graph = nx.Graph(name='Derived-named')

for origin, dest, weight in zip(adjacency['origin'], adjacency['destination'], adjacency['duration']):

    o_name, d_name = remove_block_letters(origin), remove_block_letters(dest)

    if o_name != d_name:

        derived_named_graph.add_edge(o_name, d_name, weight=weight)

    elif o_name in interchanges and d_name in interchanges:

        derived_named_graph.add_edge(o_name, d_name, weight=0)



print(nx.info(actual_named_graph))

print(nx.info(derived_named_graph))

print('Nodes:')

print('Jaccard Similarity: {:3f}'.format(similarity(actual_named_graph, derived_named_graph, mode='jaccard', nodes=True)))

print('Precision: {:3f}'.format(similarity(actual_named_graph, derived_named_graph, mode='precision', nodes=True)))

print('Recall: {:3f}'.format(similarity(actual_named_graph, derived_named_graph, mode='recall', nodes=True)))

print('F1: {:3f}'.format(similarity(actual_named_graph, derived_named_graph, mode='f1', nodes=True)))

print('Edges:')

print('Jaccard Similarity: {:3f}'.format(similarity(actual_named_graph, derived_named_graph, mode='jaccard')))

print('Precision: {:3f}'.format(similarity(actual_named_graph, derived_named_graph, mode='precision')))

print('Recall: {:3f}'.format(similarity(actual_named_graph, derived_named_graph, mode='recall')))

print('F1: {:3f}'.format(similarity(actual_named_graph, derived_named_graph, mode='f1')))



plot_comparison(actual_named_graph,derived_named_graph)
def name_to_code(s):

    """

    return list of all possible codes for input station names

    """

    return list(node_labels[node_labels['Name'] == s]['Node'].values)
# cache for get_same_line

names2codes = {}



def get_same_line(o,d):

    """

    takes in pairs of station names, return mrt code if they are on same line, else False

    """ 

    o,d = remove_block_letters(o), remove_block_letters(d)

    if (o,d) in names2codes: return names2codes[(o,d)]

    ol,dl = name_to_code(o), name_to_code(d)

    if not ol or not dl: return (False,False)

    for i in ol:

        for j in dl:

            if i[:2] == j[:2]: 

                names2codes[(o,d)] = (i,j)

                return (i,j)

    names2codes[(o,d)] = (False,False)

    return (False,False)



def get_same_line_row(s):

    return get_same_line(s['origin'],s['destination'])
line_range = {}

for n in node_labels['Node']:

    if n[:2] not in line_range:

        if n[2:].isnumeric(): line_range[n[:2]] = (int(n[2:]),int(n[2:]))

    else:

        if n[2:].isnumeric(): 

            l,h = line_range[n[:2]]

            line_range[n[:2]] = (min(l,int(n[2:])),max(h,int(n[2:])))

print(line_range)
codes = df.apply(get_same_line_row,axis=1)

direct = df[[True if x[0] else False for x in codes]]

codes = codes[[True if x[0] else False for x in codes]]

direct.loc[:,'origin'] = [int(x[0][2:]) for x in codes]

direct.loc[:,'destination'] = [int(x[1][2:]) for x in codes]

direct.loc[:,'line'] = [x[0][:2] for x in codes]

direct.loc[:,'destination_tm'] = direct.destination_tm.apply(pd.to_datetime)

direct.loc[:,'origin_tm'] = direct.origin_tm.apply(pd.to_datetime)

direct
def get_trips_A_B(line='EW',ori=23,des=10,time_min='9:00:00',time_max='10:00:00',plot=False):

    """

    input: line code, origin and destination station numbers, time range

    output: pd series of destination time x, list of durations y

    """

    assert line in line_range, '{} not valid'.format(line)

    assert line_range[line][0] <= ori <=line_range[line][1], 'Station {} not valid'.format(line)

    assert line_range[line][0] <= des <=line_range[line][1], 'Station {} not valid'.format(line)

    time_min = pd.to_datetime(time_min)

    time_max = pd.to_datetime(time_max)

    prev = des-1 if des > ori else des+1

    # prev = ori



    to_des = direct[(direct['line']==line) & (direct['origin'] == prev) & (direct['destination'] == des)]

    x = (to_des['destination_tm'] - time_min).astype('timedelta64[m]')

    y = to_des['duration']/60



    # i = ori

    i = direct[(direct['line']==line) & (direct['destination'] == des)].origin.min() if ori < des else direct[(direct['line']==line) & (direct['destination'] == des)].origin.max() 

    start = i



    while i != prev:

        to_des = direct[(direct['line']==line) & (direct['origin'] == i) & (direct['destination'] == des)]



        i = i+1 if ori < des else i-1



        time = (to_des['destination_tm']-time_min).astype('timedelta64[m]')

        dur = to_des['duration']



        x = x.append(time)

        y = y.append(dur/60)

        

    if plot:

        plt.figure(figsize=(8,8))

        # plt.title('Trips from {}{} {} to {}{} {}'.format(line,ori,code_to_name(line+str(ori)),line,des,code_to_name(line+str(des))))

        plt.title('Trips ending in {}{} {} for all origins along {} line ({}{} {} to {}{} {})'.format(line,des,code_to_name(line+str(des)),line,line,start,code_to_name(line+str(start)),line,prev,code_to_name(line+str(prev))))

        plt.xlabel('Time elapsed since 9:00 (min)')

        plt.ylabel('Duration of trips (min)')

        plt.scatter(x, y, color ='b') 

        plt.show()



    return x, y



x,y = get_trips_A_B(line='EW',ori=23,des=10,plot=True)

X = np.array(x).reshape(-1, 1)

Y = np.array(y).reshape(-1, 1)
k_low, k_high = int((max(x)-min(x))/7), int((max(x)-min(x))/2) # calculate bounds from assumptions

K = range(k_low,k_high)



@ignore_warnings(category=ConvergenceWarning)

def get_distortions(K,x,plot=False):

    """

    k-means 1-D array x in range K

    returns list of distortions

    """

    X = np.array(x).reshape(-1, 1)

    distortions = []

    for k in K:

        kmeans = KMeans(n_clusters=k)

        kmeans.fit(X)

        distortions.append(kmeans.inertia_)



    if plot:

        plt.figure(figsize=(7, 5))

        plt.plot(K, distortions,marker='o')

        plt.grid(True)

        plt.xlabel('k')

        plt.ylabel('Loss')

        plt.title('Elbow curve for deriving k')

        plt.show()

    

    return distortions



distortions = get_distortions(K,x,plot=True)
def get_k(distortions, K, plot=False):

    """

    Algorithm to find the maximum distance between line from k_low to k_high, and each k in range.

    input: list of loss for k-means in range K, K is a list from k_low to k_high

    output: optimum k

    """

    def shortest_distance(x1, y1, a, b, c):    

        return abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))



    a = distortions[0] - distortions[-1]

    b = K[-1] - K[0]

    c = K[0] * distortions[-1] - K[-1] * distortions[0]



    dist = [shortest_distance(K[k],distortions[k],a,b,c) for k in range(len(K))]

    if plot:

        plt.figure(figsize=(6,6))

        plt.xlabel('k')

        plt.ylabel('Distance')

        plt.title('Finding the peak for deriving k')

        plt.plot(K,dist)

    opt_k = K[dist.index(max(dist))]

    return opt_k



k = get_k(distortions, K)
def get_relevant_trains(x,y,k,line='EW',ori=23,des=10,time_min='9:00:00',time_max='10:00:00',plot=False,verbose=False):

    """

    Finds k clusters of arrival times from ori to des

    Assigns train arrivals at the 90th quantile of each cluster

    returns relevant trains

    """

    assert line in line_range, '{} not valid'.format(line)

    assert line_range[line][0] <= ori <=line_range[line][1], 'Station {} not valid'.format(line)

    assert line_range[line][0] <= des <=line_range[line][1], 'Station {} not valid'.format(line)

    time_min = pd.to_datetime(time_min)

    time_max = pd.to_datetime(time_max)



    X = np.array(x).reshape(-1, 1)

    Y = np.array(y).reshape(-1, 1)

    

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(X)

    

    if plot:

        plt.figure(figsize=(10,10))

        plt.scatter(X[:,0], Y, c=kmeans.labels_, cmap='rainbow')

        plt.xlabel('Time elapsed since 9:00 (min)')

        plt.ylabel('Duration of trips (min)')

        # start = direct[(direct['line']==line) & (direct['destination'] == des)].origin.min() if ori < des else direct[(direct['line']==line) & (direct['destination'] == des)].origin.max()

        # plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1] ,color='black')

        # plt.title('Trips from {}{} {} to {}{} {}'.format(line,start,code_to_name(line+str(start)),line,des,code_to_name(line+str(des))))

        # plt.title('Trips ending in {}{} {} for all origins along {} line ({}{} {} to {}{} {})'.format(line,des,code_to_name(line+str(des)),line,line,start,code_to_name(line+str(start)),line,prev,code_to_name(line+str(prev))))



    means = pd.DataFrame([kmeans.labels_,x,y]).T

    means = means.rename(columns={0:'label',1:'x',2:'y'})

    times = []

    for i in means.groupby('label').x.quantile(.9):

        i = int(i)

        if plot: plt.axvline(i)

        times.append(i)

    times.sort()



    # idx = []

    # for i,l in enumerate(kmeans.labels_[1:],1):

    #     if kmeans.labels_[i-1] != l: idx.append(i-1)

    # print(idx)



    # [plt.scatter(x,15,color='black') for x in kmeans.cluster_centers_[:,0]]



    timestamps = pd.Series([time_min + datetime.timedelta(minutes=time) for time in times])



    filtered = direct[(direct['line']==line) & (direct['origin'] == ori) & (direct['destination'] == des) & (direct['origin_tm']>= time_min)].reset_index()

    first_arrival = filtered.iloc[filtered['origin_tm'].idxmin()].destination_tm



    valid_trains = timestamps[timestamps.apply(lambda x : first_arrival <= x <= time_max)]

    if verbose: 

        print('There are {} trains arriving at {}{} {} between {} and {}'.format(len(timestamps),line,des,code_to_name(line+str(des)),timestamps.min().time(),timestamps.max().time()))

        print('There are {} trains leaving {}{} {} after {} and arriving at {}{} {} before {}'.format(len(valid_trains),line,ori,code_to_name(line+str(ori)),time_min.time(),line,des,code_to_name(line+str(des)),time_max.time()))

        print('These trains are:',[str(time.time()) for time in valid_trains])



    if plot: 

        if len(valid_trains) > 1:

            left = times[valid_trains.index[0]-1] if valid_trains.index[0] != 0 else 0

            plt.axvspan(left, times[valid_trains.index[-1]], alpha=0.2, color='green')

        plt.title('{} trains leave {}{} {} after {} and arrive at {}{} {} before {}'.format(len(valid_trains),line,ori,code_to_name(line+str(ori)),time_min.time(),line,des,code_to_name(line+str(des)),time_max.time()))

        plt.show()

        

    return valid_trains



valid_trains = get_relevant_trains(x,y,k,line='EW',ori=23,des=10,time_min='9:00:00',time_max='10:00:00',plot=True,verbose=True)
line, ori, des = 'NS', 16, 26

# line, ori, des = 'EW', 23, 10



time_min, time_max ='9:00','10:00'



x,y = get_trips_A_B(line=line,ori=ori,des=des,time_min=time_min,time_max=time_max,plot=False)



K = range(int((max(x)-min(x))/7), int((max(x)-min(x))/2)) # calculate bounds from assumptions



distortions = get_distortions(K,x,plot=False)

k = get_k(distortions, K,plot=False)



timestamps = get_relevant_trains(x,y,k,line=line,ori=ori,des=des,time_min=time_min,time_max=time_max,plot=True,verbose=True)
combined_adjacency = get_trip_pairs(df,direction='both', quantile=.1)



combined_adjacency['origin'] = combined_adjacency['origin'].apply(remove_block_letters)

combined_adjacency['destination'] = combined_adjacency['destination'].apply(remove_block_letters)



# combined_adjacency = get_nearest_n(combined_trip_pairs,20)

# name_to_labels(combined_adjacency, node_labels)



combined_adjacency
combined_network = nx.Graph(name='Combined')

missing_edges = []

for i,edge in enumerate(edges):

    origin,dest = code_to_name(edge[0]), code_to_name(edge[1])

    weight = combined_adjacency.loc[(combined_adjacency['origin'] == origin) & (combined_adjacency['destination'] == dest)]['duration']

    if weight.empty: 

        if origin == dest:

            combined_network.add_edge(edge[0], edge[1], weight=0) # assumption: Changing at interchanges takes 0 mins

        else:

            missing_edges.append((edge[0],origin,'->',edge[1],dest))

        continue

    combined_network.add_edge(edge[0], edge[1], weight=weight.values[0])

    

print(len(missing_edges),'Missing edges:',[' '.join(edge) for edge in missing_edges])

pos = get_pos([node for node,_ in edges])

print(nx.info(combined_network))
def get_path(network,o,d,weight='weight',same_line=False):

    """

    given name of station 1 and 2 in string, return shortest path of weighted network graph

    shortest path algorithm used: dijkstra's

    same_line: If True, origin-destination pairs must be from the same line.

    """

    ol,dl = name_to_code(o), name_to_code(d)

    if not ol: raise Exception(o+' is not a valid station')

    if not dl: raise Exception(d+' not found')

    for i in ol:

        for j in dl:

            if i[:2] == j[:2]:

                return nx.shortest_path(network,i,j,weight=weight)

    if same_line: raise Exception(o+' and '+d+' are not on the same line')

    min_path = nx.shortest_path(network,ol[0],dl[0],weight=weight)

    for i in ol:

        for j in dl:

            path = nx.shortest_path(network,i,j,weight=weight)

            min_path = min(min_path,path)

    return min_path
path = get_path(combined_network,'Clementi','Kallang',weight=None)

total_dur = 0

for i,stn in enumerate(path[1:],1): 

    o,d = path[i-1],stn

    dur = combined_network[o][d]['weight']

    total_dur +=  dur

    print(o,code_to_name(o),'to',d,code_to_name(d),'Duration:',dur/60,'min')

print('Total time taken',total_dur/60,'min')
nx.draw_networkx(combined_network,node_color=get_colors(combined_network),font_size=10,node_size=800, pos=fixed_pos)
print(nx.info(actual_graph))

print(nx.info(combined_network))

print('Jaccard Similarity',similarity(actual_graph, combined_network))

plot_comparison(actual_graph,combined_network,pos=fixed_pos)
def get_trips_to_des(line,ori,des):

    """

    Finds k clusters of arrival times from ori to des

    Assigns train arrivals at the 90th quantile of each cluster

    returns all trips to given destination

    """

    time_min = pd.to_datetime('9:00:00') # get all trips

    time_max = pd.to_datetime('12:00:00') # get all trips

    

    x,y = get_trips_A_B(line=line,ori=ori,des=des,time_min=time_min,time_max=time_max)

    K = range(int((max(x)-min(x))/7), int((max(x)-min(x))/2)) # calculate bounds from assumptions

    distortions = get_distortions(K,x)

    k = get_k(distortions, K)





    X = np.array(x).reshape(-1, 1)

    Y = np.array(y).reshape(-1, 1)

    

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(X)

    

    means = pd.DataFrame([kmeans.labels_,x,y]).T

    means = means.rename(columns={0:'label',1:'x',2:'y'})

    times = []

    for i in means.groupby('label').x.quantile(.9):

        times.append(int(i))

    times.sort()

    

    return pd.Series([time_min + datetime.timedelta(minutes=time) for time in times])
timestamps = get_trips_to_des(line='EW',ori=23,des=22)

timestamps
def get_time_diff(timestamps):

    """

    input: pd date_time series

    output: time difference

    """

    td_series = pd.Series([(time - timestamps[i-1]) for i,time in enumerate(timestamps[1:],1)]).astype('timedelta64[m]')

#     print('Average time between trains: {:.2f} mins with std dev {:.2f} mins.'.format(td_series.mean(),td_series.std()))

    return td_series



get_time_diff(timestamps)
EW_eastbound = {des: get_trips_to_des(line='EW',ori=line_range['EW'][1],des=des) for des in range(line_range['EW'][1]-5,line_range['EW'][0]-1,-1)}
plt.figure(figsize=(8,5))

plt.plot([len(timestamps) for timestamps in EW_eastbound.values()])

plt.title('Number of learned arrival times per station')

plt.ylabel('Number of arrival times')

plt.xlabel('Station Number')

plt.show()
plt.figure(figsize=(15,10))

[plt.scatter(np.ones(len(timestamps))*stn,timestamps) for stn, timestamps in EW_eastbound.items()]

    

plt.title('Eastbound stations on EW line by arrivals')

plt.xlabel('Station Number')

plt.ylabel('Arrival times')

plt.show()
def get_ew_east_trains(num, walking=120):

    assert num <= len(EW_eastbound[line_range['EW'][1]-5])

    trains = []

    for i in range(num):

        train = {}

        stn = line_range['EW'][1]-5

        curr = EW_eastbound[stn][i]

        train[stn] = curr

        while stn != line_range['EW'][0]:

            # print('From {} to {}'.format(stn,stn-1))

            o2d = direct[(direct['line']=='EW')&(direct['destination']==stn-1)&(direct['origin']==stn)] if stn != 14 else direct[(direct['line']=='NS')&(direct['destination']==25)&(direct['origin']==26)]

            match = curr + datetime.timedelta(seconds=o2d.duration.min()-walking)

            curr = EW_eastbound[stn-1].where(EW_eastbound[stn-1]>match).min()

            

            if i != 0:

                if (stn-1) in trains[i-1]:

                    while curr == trains[i-1][stn-1]:

                        match = trains[i-1][stn-1]

                        curr = EW_eastbound[stn-1].where(EW_eastbound[stn-1]>match).min()

                        if str(curr) =='NaT': break



            if str(curr) =='NaT': break

            train[stn-1] = curr

            stn -= 1

        trains.append(train)

    return trains



trains = get_ew_east_trains(10,walking=120)
plt.figure(figsize=(15,10))

[plt.scatter(np.ones(len(timestamps))*stn,timestamps) for stn, timestamps in EW_eastbound.items()]



[plt.plot([stn for stn in train.keys()],[time for time in train.values()]) for train in trains]

    

plt.title('Eastbound stations on EW line by arrivals')

plt.xlabel('Station Number')

plt.ylabel('Arrival times')

plt.show()
def get_train_from_station(ori=23,des=10,time='09:25'):

    time = pd.to_datetime(time)

    for i, train in enumerate(trains):

        if ori in train:

            if train[ori] > time:

                return (i,train)

               # return {key:time for key,time in train.items() if 10 <= key <= ori}
train_num, curr_train = get_train_from_station(ori=23,des=10,time='09:25')

prev_train, next_train = trains[train_num-1], trains[train_num+1]



i = 23

des = 10

people = len(direct[(direct['line']=='EW')&(direct['origin']==i+1)&(direct['destination']<i+1)&(direct['origin_tm'].between(prev_train[i+1],curr_train[i+1]))])



while i >= des:

    enter = len(direct[(direct['line']=='EW')&(direct['origin']==i)&(direct['destination']<i)&(direct['origin_tm'].between(prev_train[i],curr_train[i]))])

    exit = len(direct[(direct['line']=='EW')&(direct['destination']==i)&(direct['origin']>i)&(direct['destination_tm'].between(curr_train[i],next_train[i]))])

    print('EW{} {:13}: {:4} commuters on the train. {:2} entered the train, while {:3} exited.'.format(i,code_to_name('EW'+str(i)),people,enter,exit))

    people += enter-exit

    i -= 1
