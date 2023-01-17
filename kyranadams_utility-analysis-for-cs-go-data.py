import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import gc

import math

import pandas_profiling
grenade_df = pd.read_csv('/kaggle/input/csgo-matchmaking-damage/esea_master_grenades_demos.part1.csv')

grenade_df = grenade_df[['file', 'round', 'seconds', 'att_side', 'vic_side', 'is_bomb_planted', 'bomb_site', 'nade', 'nade_land_x', 'nade_land_y', 'att_pos_x', 'att_pos_y']]

meta_df = pd.read_csv('/kaggle/input/csgo-matchmaking-damage/esea_meta_demos.part1.csv')

grenade_df = pd.merge(grenade_df, meta_df, how='left', left_on=['file','round'], right_on = ['file','round'])

grenade_df['seconds'] -= grenade_df['start_seconds']



grenade_df.set_index(['file', 'round'], inplace=True)



map_df = pd.read_csv('/kaggle/input/csgo-matchmaking-damage/map_data.csv')

map_df = map_df.rename( columns={'Unnamed: 0':'map_name'}).set_index('map_name')

print(map_df)



print(grenade_df.info())

print(grenade_df.head())

from sklearn.cluster import AgglomerativeClustering, DBSCAN

from sklearn.neighbors.nearest_centroid import NearestCentroid

from time import time

from sklearn.linear_model import LogisticRegression



#Convert map coordinates to image coordinates, from Bill Freeman's analysis

def pointx_to_resolutionx(xinput,startX=-3217,endX=1912,resX=1024):

    sizeX = endX - startX

    if startX < 0:

        xinput += startX * (-1.0)

    else:

        xinput += startX

    xoutput = float((xinput / abs(sizeX)) * resX);

    return xoutput



def pointy_to_resolutiony(yinput,startY=-3401,endY=1682,resY=1024):

    sizeY=endY-startY

    if startY < 0:

        yinput += startY *(-1.0)

    else:

        yinput += startY

    youtput = float((yinput / abs(sizeY)) * resY);

    return resY-youtput



#grenade_df['att_pos_x'] = grenade_df['att_pos_x'].apply(pointx_to_resolutionx)

#grenade_df['att_pos_y'] = grenade_df['att_pos_y'].apply(pointy_to_resolutiony)



def cluster_utility(grenade_df, max_nades_to_process=-1, nade_types = ['Smoke', 'HE', 'Flash'], teams = ['Terrorist', 'CounterTerrorist'], map_name='de_mirage', planted=False):

    """

    Given a dataframe of grenade usage, clusters the grenades into the most common spots, and plots the results.

    """

    # Create graphics

    fig, axs = plt.subplots(len(nade_types), len(teams), figsize=(25,25))

    map_filenames = {map_name: f'/kaggle/input/csgo-matchmaking-damage/{map_name}.png' for map_name in grenade_df['map'].unique()}

    im = plt.imread(map_filenames[map_name])

    

    # Preprocess data on map of interest

    t_mirage_df = grenade_df[grenade_df['map'] == map_name]

    map_info = map_df.loc[map_name]

    t_mirage_df['cluster'] = None



    if planted is not None:

        t_mirage_df = t_mirage_df[t_mirage_df['is_bomb_planted'] == planted]

    

    

    # For each nade type and team, we will compute clusters and create a plot

    for i, nade_type in enumerate(nade_types):

        for j, team in enumerate(teams):

            before_time = time()

            t_mirage_nade_df = t_mirage_df[(t_mirage_df.nade==nade_type) & (t_mirage_df.att_side==team)]

            # Correct location

            t_mirage_nade_df['att_pos_y'] = t_mirage_nade_df['att_pos_y'].apply(pointy_to_resolutiony, args=(map_info['StartY'], map_info['EndY'], map_info['ResY']))

            t_mirage_nade_df['att_pos_x'] = t_mirage_nade_df['att_pos_x'].apply(pointx_to_resolutionx, args=(map_info['StartX'], map_info['EndX'], map_info['ResX']))

            t_mirage_nade_df['nade_land_y'] = t_mirage_nade_df['nade_land_y'].apply(pointy_to_resolutiony, args=(map_info['StartY'], map_info['EndY'], map_info['ResY']))

            t_mirage_nade_df['nade_land_x'] = t_mirage_nade_df['nade_land_x'].apply(pointx_to_resolutionx, args=(map_info['StartX'], map_info['EndX'], map_info['ResX']))

            # Determine how many points we will use from the data

            available_datapoints = len(t_mirage_nade_df)

            num_nades = max_nades_to_process

            if max_nades_to_process < 0:

                num_nades = available_datapoints

            num_nades = min(num_nades, available_datapoints)

            print(f"Using {num_nades}/{available_datapoints} available datapoints in {team} {nade_type}")

            t_mirage_nade_df = t_mirage_nade_df[:num_nades]

            ### Cluster into common smoke positions

            cluster = DBSCAN(eps=8, min_samples=num_nades/350, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)

            smoke_pts = pd.concat([t_mirage_nade_df['nade_land_x'], t_mirage_nade_df['nade_land_y']], axis=1)

            cluster.fit(smoke_pts)

            t_mirage_nade_df['cluster'] = cluster.labels_

            ### Find centroids of clusters

            centroids = NearestCentroid()

            centroids.fit(smoke_pts, cluster.labels_)

            nade_clusters = centroids.centroids_

            

            # Calculate distance to centroid

            def distance_to_centroid(row):

                if row['cluster'] == -1:

                    return np.nan

                centroid = nade_clusters[row['cluster']+1]

                return math.sqrt((row['nade_land_x'] - centroid[0]) ** 2 + (row['nade_land_y'] - centroid[1]) ** 2)

            #t_mirage_nade_df['distance_to_centroid'] = t_mirage_nade_df.apply(distance_to_centroid, axis=1)

            

            # Plot smokes and clusters

            axs[i, j].set_title(f"{map_name} {team} {nade_type}s")

            axs[i, j].set_xlim(0, 1000)

            axs[i, j].set_ylim(1000, 0)

            t = axs[i, j].imshow(im)

            # Create map from clusters to color pallete

            color_labels = np.unique(cluster.labels_)

            rgb_values = sns.color_palette("Set2", color_labels.shape[0])

            color_map = dict(zip(color_labels, rgb_values))

            color_map[-1] = (0, 0, 0)



            # Plot! Uncomment one of the following to plot clusters, distance to centroid, or winning side

            #axs[i, j].scatter(t_mirage_nade_df['nade_land_x'], t_mirage_nade_df['nade_land_y'],alpha=2500/num_nades,c=t_mirage_nade_df['cluster'].map(color_map), marker='.')

            #axs[i, j].scatter(t_mirage_nade_df['nade_land_x'], t_mirage_nade_df['nade_land_y'],alpha=2500/num_nades,c=t_mirage_nade_df['distance_to_centroid'], marker='.', cmap=plt.cm.autumn)

            axs[i, j].scatter(t_mirage_nade_df['nade_land_x'], t_mirage_nade_df['nade_land_y'],alpha=2500/num_nades,c=(t_mirage_nade_df['winner_side'] == team).map({True: (1, 0, 0), False: (0, 0, 1)}), marker='.', cmap=plt.get_cmap('plasma'))

            

            axs[i, j].scatter(nade_clusters[:, 0], nade_clusters[:, 1], alpha=1,c='yellow')

            for k, pos in enumerate(zip(nade_clusters[:, 0], nade_clusters[:, 1])):

                print(f"{k}: {pos}")

                axs[i, j].annotate(str(k), pos, color='white', fontsize=15)



            # Collect our results

            t_mirage_df = t_mirage_df[(t_mirage_df.nade!=nade_type) | (t_mirage_df.att_side!=team)]

            t_mirage_df = t_mirage_df.append(t_mirage_nade_df)

            elapsed_time = time() - before_time

            print(f"Clustering took {elapsed_time} seconds")

            # Save the data

            n_points = 10000

            print(t_mirage_nade_df.columns)

            t_mirage_nade_df = t_mirage_nade_df.reset_index()

            t_mirage_nade_df["file_round"] = t_mirage_nade_df["file"] + t_mirage_nade_df["round"].astype(str)

            t_mirage_nade_df = t_mirage_nade_df[['file_round', 'seconds', 'att_side', 'bomb_site', 'nade_land_x', 'nade_land_y', 'cluster', 'winner_side']]

            t_mirage_nade_df.sample(n_points).to_csv(f'{map_name}_{team}_{nade_type}_{n_points}_locs.csv')

    print( t_mirage_df.loc[("esea_match_13792436.dem", 10)])

    t_mirage_df.loc[("esea_match_13792436.dem", 9)].to_csv(f'{map_name}_example_round.csv')

    return t_mirage_df
from IPython.display import FileLink

map_name = 'de_mirage'

clustered_nade_df = cluster_utility(grenade_df, map_name=map_name, max_nades_to_process = 130000)
dat =  clustered_nade_df.loc[("esea_match_13779770.dem", 6)]

print(dat)

fname = f'{map_name}_example_round.csv'

clustered_nade_df.loc[("esea_match_13779770.dem", 6)].to_csv(fname)

FileLink(fname)
for map_name in ['de_mirage']:#['de_cache', 'de_cbble', 'de_dust2', 'de_inferno', 'de_mirage', 'de_overpass', 'de_train']:

    print(f"Calculating for {map_name}...")

    

    clustered_nade_df = cluster_utility(grenade_df, map_name=map_name, max_nades_to_process = 130000)

    perc_unclustered = 100 * len(clustered_nade_df[clustered_nade_df['cluster'] == -1]) / len(clustered_nade_df)

    print(f"{perc_unclustered}% of nades are unclustered")



    print(clustered_nade_df.columns)

    round_nade_df = pd.get_dummies(data=clustered_nade_df, columns=['cluster'])

    cluster_cols = [col for col in round_nade_df if col.startswith('cluster')]

    #print(t_mirage_df_one_hot[['att_side', 'nade'] + cols])

    round_nade_df = round_nade_df.drop(['round_type', 'winner_side'], axis=1).groupby(['file', 'round', 'att_side', 'nade'])

    round_nade_df = round_nade_df[cluster_cols].sum()

    round_nade_df = round_nade_df.unstack(level=['att_side', 'nade'])

    round_nade_df.columns = round_nade_df.columns.to_flat_index()

    round_nade_df = round_nade_df.reset_index()

    # Combine round win information

    meta_df = pd.read_csv('/kaggle/input/csgo-matchmaking-damage/esea_meta_demos.part1.csv')[['file', 'round', 'winner_side']]

    round_nade_df = pd.merge(round_nade_df, meta_df, how='left', left_on=['file','round'], right_on = ['file','round'])

    round_nade_df = round_nade_df.fillna(0)

    round_nade_df.columns = [tup if isinstance(tup, str) else f"{tup[2]}_{tup[1]}_{tup[0]}" for tup in round_nade_df.columns]

    # Remove clusters with no nades

    nade_counts = round_nade_df.drop(['file', 'round', 'winner_side'], axis=1).sum(axis=0)

    round_nade_df = round_nade_df.drop(nade_counts[nade_counts == 0].keys(), axis=1)

    round_nade_df.to_csv(f'{map_name}_clustered_nade_round_win.csv')

    gc.collect()
