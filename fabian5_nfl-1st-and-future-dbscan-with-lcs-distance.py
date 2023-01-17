import numpy as np

import pandas as pd

import functools

import matplotlib.pyplot as plt

import matplotlib.patches as patches

from collections import Counter
# this code handles the random subsampling of the player track data



def sample_player_track_data(

        path_player_tracks, path_plays, nr_random, downsampling=None,

        chunk_size=2*1e5):

    plays = pd.read_csv(path_plays)

    play_ids = np.random.choice(plays.PlayKey.unique(), nr_random)

    chunks = pd.read_csv(path_player_tracks, chunksize=chunk_size)

    player_df = pd.concat(valid_find(chunks, play_ids))

    print("Extracted %d rows" % len(player_df))

    return extract_trajectories(player_df, downsampling)





def valid_find(chunks, tracks):

    tracks_found = set()

    for chunk in chunks:

        mask = chunk.PlayKey.isin(tracks)

        if mask.all():

            res = chunk

        else:

            res = chunk.loc[mask]

        tracks_found.update(res.PlayKey.unique())

        if len(tracks_found) == len(set(tracks)) and not mask.any():

            break

        yield res

        

def extract_trajectories(player_df, downsampling):

    print("downsampling:", downsampling)

    trajectories = []

    keys = []

    for key, player_track in player_df.groupby('PlayKey'):

        keys.append(key)

        if downsampling is not None:

            trajectories.append(player_track[['x', 'y']].values[0::downsampling])

        else:

            trajectories.append(player_track[['x', 'y']].values)

    return trajectories, keys





def valid_find(chunks, tracks):

    tracks_found = set()

    for chunk in chunks:

        mask = chunk.PlayKey.isin(tracks)

        if mask.all():

            res = chunk

        else:

            res = chunk.loc[mask]

        tracks_found.update(res.PlayKey.unique())

        if len(tracks_found) == len(set(tracks)) and not mask.any():

            break

        yield res





def find_players_tracks(data_path, tracks, chunk_size=2*1e5):

    chunks = pd.read_csv(data_path, chunksize=chunk_size)

    player_df = pd.concat(valid_find(chunks, tracks))

    return extract_trajectories(player_df, None)
# This code follows the LCS implementation by GeeksforGeeks (see above for more information)



def lcs(X, Y, eps, delta):

    m = len(X)

    n = len(Y)

    L = np.zeros((m+1, n+1))

    for i in range(m + 1):

        for j in range(n + 1):

            if i == 0 or j == 0:

                L[i][j] = 0

            elif abs(X[i-1][0] - Y[j-1][0]) <= eps and abs(X[i-1][1] - Y[j-1][1]) <= eps and abs(i - j) < delta:

                L[i][j] = L[i-1][j-1]+1

            else:

                L[i][j] = max(L[i-1][j], L[i][j-1])

    return L[m][n]





# We need a distance function for two trajectories to be used in the clustering

# This follows the idea from Vlachos et al.

def lcs_dist(x, y, eps, delta):

    return 1 - lcs(x, y, eps, delta) / min(len(x), len(y))
# This code follow the DBSCAN implementation by Chris McCormick (see above for more information)

def dbscan(D, eps, MinPts, dist):

    labels = [0]*len(D)

    C = 0

    for P in range(0, len(D)):

        if not (labels[P] == 0):

           continue

        NeighborPts = regionQuery(D, P, eps, dist)

        if len(NeighborPts) < MinPts:

            labels[P] = -1

        else:

            C += 1

            growCluster(D, labels, P, NeighborPts, C, eps, MinPts, dist)

    return labels





def growCluster(D, labels, P, NeighborPts, C, eps, MinPts, dist):

    labels[P] = C

    i = 0

    while i < len(NeighborPts):

        Pn = NeighborPts[i]

        if labels[Pn] == -1:

            labels[Pn] = C

        elif labels[Pn] == 0:

            labels[Pn] = C

            PnNeighborPts = regionQuery(D, Pn, eps, dist)

            if len(PnNeighborPts) >= MinPts:

                NeighborPts = NeighborPts + PnNeighborPts

        i += 1



        

def regionQuery(D, P, eps, dist):

    neighbors = []

    for Pn in range(0, len(D)):

        if dist(D[P], D[Pn]) < eps:

            neighbors.append(Pn)

    return neighbors

nr_plays = 10

downsampling = 4



lcs_eps = 1.0

lcs_delta = 10

lcs_dist_param = functools.partial(lcs_dist, eps=lcs_eps, delta=lcs_delta)



dbscan_eps = 0.3

dbscan_min_samples = 5
trajectories, ids = sample_player_track_data("/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv", "/kaggle/input/nfl-playing-surface-analytics/PlayList.csv", nr_plays, downsampling=downsampling)
labels = dbscan(trajectories, dbscan_eps, dbscan_min_samples, lcs_dist_param)
print("Found %d clusters" % max(labels))

labels_df = pd.DataFrame(zip(ids, labels), columns=['Key', 'Cluster'])
# Code for plotting trajectory clusters on a football field

# The implementation of `create_football_field` is based on Rob Mulla's notebook (see above for more information)

def create_football_field(

        linenumbers=True, endzones=True, highlight_line=False,

        highlight_line_number=50, highlighted_name='Line of Scrimmage',

        fifty_is_los=False, figsize=(12, 6.33)):



    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1, edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)

    ax.add_patch(rect)

    plt.plot(

        [10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80, 80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],

        [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],

        color='white')

    if fifty_is_los:

        plt.plot([60, 60], [0, 53.3], color='gold')

        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')

    if endzones:

        ez1 = patches.Rectangle((0, 0), 10, 53.3, linewidth=0.1, edgecolor='r', facecolor='blue', alpha=0.2, zorder=0)

        ez2 = patches.Rectangle((110, 0), 120, 53.3, linewidth=0.1, edgecolor='r', facecolor='blue', alpha=0.2, zorder=0)

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

            plt.text(x, 5, str(numb - 10), horizontalalignment='center', fontsize=20, color='white')

            plt.text(x - 0.95, 53.3 - 5, str(numb - 10), horizontalalignment='center', fontsize=20, color='white', rotation=180)

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

        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name), color='yellow')

    return fig, ax



def plot_cluster_on_field(cluster, plays, trajectory_dict):

    fig, ax = create_football_field(figsize=(16, 6.33))

    c_trajectories = []

    for trajectory_id in cluster:

        c_trajectories.append(trajectory_dict[trajectory_id])

    count_injuries = 0

    positions = Counter()

    for i, t in enumerate(c_trajectories):

        play = plays[plays.PlayKey == cluster[i]]

        injury = injuries[injuries.PlayerKey == play.iloc[0].PlayerKey]

        injured = len(injury) > 0

        print(

            cluster[i], play.iloc[0].PlayerKey, play.iloc[0].RosterPosition,

            play.iloc[0].StadiumType, play.iloc[0].PlayType, injured)

        color = 'orange'

        if injured:

            color = 'red'

            count_injuries += 1

        positions[play.iloc[0].PositionGroup] += 1

        pd.DataFrame(t, columns=['x', 'y']).plot(

            kind='scatter', x='x', y='y', ax=ax, color=color, alpha=0.7, s=3)



    positions_str = 'Positions:\n'

    for key, value in positions.most_common():

        positions_str += "%s: %2d (%.2f%%)\n" % (

            key, value, value / len(c_trajectories) * 100)



    ax.text(122.5, 53.3, "%d Plays\n%d from injured (%2.f%%)\n%s" % (

        len(c_trajectories), count_injuries,

        count_injuries / len(c_trajectories) * 100, positions_str),

        verticalalignment='top')

    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)

    plt.show()

# the code in the above will produce a labeling akin to the following data structure:

labels_df = pd.DataFrame([

    ["26624-1-18", -1],

    ["27363-17-60", 0],

    ["26624-1-26", -1],

    ["26624-1-30", -1],

    ["27363-24-38", 0],

    ["26624-1-37", -1],

    ["26624-1-42", -1],

    ["35577-27-19", 0],

    ["33474-20-25", 0],

    ["26624-1-46", -1],

    ["32103-8-31", 0],

    ["26624-1-48", -1],

    ["35648-15-4", 0],

    ["33474-25-44", 0],

    ["26624-1-59", -1],

    ["34214-20-60", 0],

    ["26624-1-77", -1],

    ["33337-15-3", 0],

    ["34230-18-32", 0],

    ["36555-5-54", 0]], columns=['Key', 'Cluster'])

# Each positive number represents one cluster which can visualized with the following code



plays = pd.read_csv("/kaggle/input/nfl-playing-surface-analytics/PlayList.csv")

injuries = pd.read_csv("/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv")

trajectories, ids = find_players_tracks(

    "/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv",

    labels_df[labels_df.Cluster > -1].Key.tolist())

trajectory_dict = {}

for i, trajectory_id in enumerate(ids):

    trajectory_dict[trajectory_id] = trajectories[i]

# let's visualize a cluster

cluster_id = 0

keys_of_tracks = labels_df[labels_df.Cluster == cluster_id].Key.tolist()

plot_cluster_on_field(keys_of_tracks, plays, trajectory_dict)