import os

import numpy as np
import pandas as pd

from trackml.dataset import load_event
from trackml.score import score_event

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, Normalizer

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

from scipy import spatial
# function for arranging hits in a track (closest to origin, closest track to previous track, and so on...)
def arrange_track(track_points):
    arranged_track = pd.DataFrame()

    pt = [0, 0, 0]
    kdtree = spatial.KDTree(track_points)
    distance, index = kdtree.query(pt)

    arranged_track = arranged_track.append(track_points.iloc[index])
    track_points = track_points.drop(track_points.index[index]).reset_index(drop=True)

    while not track_points.empty:
        pt = arranged_track.iloc[-1]
        kdtree = spatial.KDTree(track_points)
        distance, index = kdtree.query(pt)

        arranged_track = arranged_track.append(track_points.iloc[index])
        track_points = track_points.drop(track_points.index[index]).reset_index(drop=True)
        
    return arranged_track

test_points = pd.DataFrame([[0, 0, 5], [0, 0, 1], [0, 0, 3], [0, 0, 2]])
arrange_track(test_points)
hits, cells, particles, truth = load_event('../input/train_1/event000001000')
hits.head()
truth.head()
tracks = truth.particle_id.unique()[1::500]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x', 'y', 'z']]
    ax.plot3D(t.z, t.x, t.y, '.', ms=10)
    
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
ax.set_title("True Tracks (Scatter Plot)", y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x', 'y', 'z']]
    t = arrange_track(t)
    ax2.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax2.set_xlabel('z (mm)')
ax2.set_ylabel('x (mm)')
ax2.set_zlabel('y (mm)')
ax2.set_title("True Tracks (Line Plot)", y=-.15, size=20)

plt.show()
tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x', 'y', 'z']]
    ax.plot3D(t.z, t.x, t.y, '.', ms=10)
    
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
ax.set_title("True Tracks (Scatter Plot)", y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x', 'y', 'z']]
    t = arrange_track(t)
    ax2.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax2.set_xlabel('z (mm)')
ax2.set_ylabel('x (mm)')
ax2.set_zlabel('y (mm)')
ax2.set_title("True Tracks (Line Plot)", y=-.15, size=20)

plt.show()
# DBSCAN benchmark preprocessing / coordinate transformation
x = hits.x.values
y = hits.y.values
z = hits.z.values

r = np.sqrt(x**2 + y**2 + z**2)
hits['x2'] = x/r
hits['y2'] = y/r

r = np.sqrt(x**2 + y**2)
hits['z2'] = z/r
tracks = truth.particle_id.unique()[1::500]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x2', 'y2', 'z2']]
    ax.plot3D(t.z2, t.x2, t.y2, '.', ms=10)
    
ax.set_xlabel('z2')
ax.set_ylabel('x2')
ax.set_zlabel('y2')
ax.set_title("True Tracks (Scatter Plot)", y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x2', 'y2', 'z2']]
    t = arrange_track(t)
    ax2.plot3D(t.z2, t.x2, t.y2, '.-', ms=10)
    
ax2.set_xlabel('z2')
ax2.set_ylabel('x2')
ax2.set_zlabel('y2')
ax2.set_title("True Tracks (Line Plot)", y=-.15, size=20)

plt.show()
tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x2', 'y2', 'z2']]
    ax.plot3D(t.z2, t.x2, t.y2, '.', ms=10)
    
ax.set_xlabel('z2)')
ax.set_ylabel('x2')
ax.set_zlabel('y2')
ax.set_title("True Tracks (Scatter Plot)", y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x2', 'y2', 'z2']]
    t = arrange_track(t)
    ax2.plot3D(t.z2, t.x2, t.y2, '.-', ms=10)
    
ax2.set_xlabel('z2')
ax2.set_ylabel('x2')
ax2.set_zlabel('y2')
ax2.set_title("True Tracks (Line Plot)", y=-.15, size=20)

plt.show()
X = hits[['x2', 'y2', 'z2']]
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

eps = 0.008
min_samp = 1
db = DBSCAN(eps=eps, min_samples=min_samp, metric='euclidean').fit(X)
labels = db.labels_

clustering = pd.DataFrame()
clustering['hit_id'] = truth['hit_id']
clustering['track_id'] = labels

score = score_event(truth, clustering)
print('track-ml custom metric score:', round(score, 4))

labels_true = truth['particle_id']
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print('\nOTHER CLUSTERING RESULTS:')
print('Estimated number of clusters: %d' % n_clusters)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
rej_perc = list(labels).count(-1) / float(hits.shape[0]) * 100
rej_perc = round(rej_perc, 2)
print ("Rejected samples %:", str(rej_perc) + '%')
rejected_count = list(labels).count(-1)
print ("Rejected samples:", rejected_count)
print ("Total samples:", hits.shape[0])
print ("Clustered samples:", hits.shape[0] - list(labels).count(-1))
tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')

tracks_hit_ids = truth[truth['particle_id'].isin(tracks)]['hit_id'] # all hits in tracks
clusters = clustering[clustering['hit_id'].isin(tracks_hit_ids)].track_id.unique() # all clusters containing the hits in tracks
for cluster in clusters:
    cluster_hit_ids = clustering[clustering['track_id'] == cluster]['hit_id'] # all hits in cluster
    plot_hit_ids = list(set(tracks_hit_ids) & set(cluster_hit_ids))
    t = hits[hits['hit_id'].isin(plot_hit_ids)][['x2', 'y2', 'z2']]
    if cluster == -1:
        ax.plot3D(t.z2, t.x2, t.y2, '.', ms=10, color='black')
    else:
        ax.plot3D(t.z2, t.x2, t.y2, '.-', ms=10)
    
ax.set_xlabel('z2')
ax.set_ylabel('x2')
ax.set_zlabel('y2')
ax.set_title('Clustered Hits (Predicted Tracks)', y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x2', 'y2', 'z2']]
    t = arrange_track(t)
    ax2.plot3D(t.z2, t.x2, t.y2, '.-', ms=10)
    
ax2.set_xlabel('z2')
ax2.set_ylabel('x2')
ax2.set_zlabel('y2')
ax2.set_title('True Tracks', y=-.15, size=20)

plt.show()
tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')

tracks_hit_ids = truth[truth['particle_id'].isin(tracks)]['hit_id'] # all hits in tracks
clusters = clustering[clustering['hit_id'].isin(tracks_hit_ids)].track_id.unique() # all clusters containing the hits in tracks
for cluster in clusters:
    cluster_hit_ids = clustering[clustering['track_id'] == cluster]['hit_id'] # all hits in cluster
    plot_hit_ids = list(set(tracks_hit_ids) & set(cluster_hit_ids))
    t = hits[hits['hit_id'].isin(plot_hit_ids)][['x', 'y', 'z']]
    if cluster == -1:
        ax.plot3D(t.z, t.x, t.y, '.', ms=10, color='black')
    else:
        ax.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
ax.set_title('Clustered Hits (Predicted Tracks)', y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x', 'y', 'z']]
    t = arrange_track(t)
    ax2.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax2.set_xlabel('z (mm)')
ax2.set_ylabel('x (mm)')
ax2.set_zlabel('y (mm)')
ax2.set_title('True Tracks', y=-.15, size=20)

plt.show()
X = hits[['x2', 'y2', 'z2']]
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

eps = 0.018
min_samp = 1
db = DBSCAN(eps=eps, min_samples=min_samp, metric='euclidean').fit(X)
labels = db.labels_

clustering = pd.DataFrame()
clustering['hit_id'] = truth['hit_id']
clustering['track_id'] = labels

score = score_event(truth, clustering)
print('track-ml custom metric score:', round(score, 4))

labels_true = truth['particle_id']
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print('\nOTHER CLUSTERING RESULTS:')
print('Estimated number of clusters: %d' % n_clusters)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
rej_perc = list(labels).count(-1) / float(hits.shape[0]) * 100
rej_perc = round(rej_perc, 2)
print ("Rejected samples %:", str(rej_perc) + '%')
rejected_count = list(labels).count(-1)
print ("Rejected samples:", rejected_count)
print ("Total samples:", hits.shape[0])
print ("Clustered samples:", hits.shape[0] - list(labels).count(-1))
tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')

tracks_hit_ids = truth[truth['particle_id'].isin(tracks)]['hit_id'] # all hits in tracks
clusters = clustering[clustering['hit_id'].isin(tracks_hit_ids)].track_id.unique() # all clusters containing the hits in tracks
for cluster in clusters:
    cluster_hit_ids = clustering[clustering['track_id'] == cluster]['hit_id'] # all hits in cluster
    plot_hit_ids = list(set(tracks_hit_ids) & set(cluster_hit_ids))
    t = hits[hits['hit_id'].isin(plot_hit_ids)][['x2', 'y2', 'z2']]
    if cluster == -1:
        ax.plot3D(t.z2, t.x2, t.y2, '.', ms=10, color='black')
    else:
        ax.plot3D(t.z2, t.x2, t.y2, '.-', ms=10)
    
ax.set_xlabel('z2')
ax.set_ylabel('x2')
ax.set_zlabel('y2')
ax.set_title('Clustered Hits (Predicted Tracks)', y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x2', 'y2', 'z2']]
    t = arrange_track(t)
    ax2.plot3D(t.z2, t.x2, t.y2, '.-', ms=10)
    
ax2.set_xlabel('z2')
ax2.set_ylabel('x2')
ax2.set_zlabel('y2')
ax2.set_title('True Tracks', y=-.15, size=20)

plt.show()
tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')

tracks_hit_ids = truth[truth['particle_id'].isin(tracks)]['hit_id'] # all hits in tracks
clusters = clustering[clustering['hit_id'].isin(tracks_hit_ids)].track_id.unique() # all clusters containing the hits in tracks
for cluster in clusters:
    cluster_hit_ids = clustering[clustering['track_id'] == cluster]['hit_id'] # all hits in cluster
    plot_hit_ids = list(set(tracks_hit_ids) & set(cluster_hit_ids))
    t = hits[hits['hit_id'].isin(plot_hit_ids)][['x', 'y', 'z']]
    if cluster == -1:
        ax.plot3D(t.z, t.x, t.y, '.', ms=10, color='black')
    else:
        ax.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
ax.set_title('Clustered Hits (Predicted Tracks)', y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x', 'y', 'z']]
    t = arrange_track(t)
    ax2.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax2.set_xlabel('z (mm)')
ax2.set_ylabel('x (mm)')
ax2.set_zlabel('y (mm)')
ax2.set_title('True Tracks', y=-.15, size=20)

plt.show()
X = hits[['x2', 'y2', 'z2']]
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

eps = 0.008
min_samp = 3
db = DBSCAN(eps=eps, min_samples=min_samp, metric='euclidean').fit(X)
labels = db.labels_

clustering = pd.DataFrame()
clustering['hit_id'] = truth['hit_id']
clustering['track_id'] = labels

score = score_event(truth, clustering)
print('track-ml custom metric score:', round(score, 4))

labels_true = truth['particle_id']
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print('\nOTHER CLUSTERING RESULTS:')
print('Estimated number of clusters: %d' % n_clusters)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
rej_perc = list(labels).count(-1) / float(hits.shape[0]) * 100
rej_perc = round(rej_perc, 2)
print ("Rejected samples %:", str(rej_perc) + '%')
rejected_count = list(labels).count(-1)
print ("Rejected samples:", rejected_count)
print ("Total samples:", hits.shape[0])
print ("Clustered samples:", hits.shape[0] - list(labels).count(-1), '\n')

print ('WITHOUT REJECTED SAMPLES:')
labels_true_wr = labels_true[labels != -1]
labels_wr = labels[labels != -1]
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true_wr, labels_wr))
print(("Completeness: %0.3f" % metrics.completeness_score(labels_true_wr, labels_wr)), '\n')
tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')

tracks_hit_ids = truth[truth['particle_id'].isin(tracks)]['hit_id'] # all hits in tracks
clusters = clustering[clustering['hit_id'].isin(tracks_hit_ids)].track_id.unique() # all clusters containing the hits in tracks
for cluster in clusters:
    cluster_hit_ids = clustering[clustering['track_id'] == cluster]['hit_id'] # all hits in cluster
    plot_hit_ids = list(set(tracks_hit_ids) & set(cluster_hit_ids))
    t = hits[hits['hit_id'].isin(plot_hit_ids)][['x2', 'y2', 'z2']]
    if cluster == -1:
        ax.plot3D(t.z2, t.x2, t.y2, '.', ms=10, color='black')
    else:
        ax.plot3D(t.z2, t.x2, t.y2, '.-', ms=10)
    
ax.set_xlabel('z2')
ax.set_ylabel('x2')
ax.set_zlabel('y2')
ax.set_title('Clustered Hits (Predicted Tracks)', y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x2', 'y2', 'z2']]
    t = arrange_track(t)
    ax2.plot3D(t.z2, t.x2, t.y2, '.-', ms=10)
    
ax2.set_xlabel('z2')
ax2.set_ylabel('x2')
ax2.set_zlabel('y2')
ax2.set_title('True Tracks', y=-.15, size=20)

plt.show()
tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')

tracks_hit_ids = truth[truth['particle_id'].isin(tracks)]['hit_id'] # all hits in tracks
clusters = clustering[clustering['hit_id'].isin(tracks_hit_ids)].track_id.unique() # all clusters containing the hits in tracks
for cluster in clusters:
    cluster_hit_ids = clustering[clustering['track_id'] == cluster]['hit_id'] # all hits in cluster
    plot_hit_ids = list(set(tracks_hit_ids) & set(cluster_hit_ids))
    t = hits[hits['hit_id'].isin(plot_hit_ids)][['x', 'y', 'z']]
    if cluster == -1:
        ax.plot3D(t.z, t.x, t.y, '.', ms=10, color='black')
    else:
        ax.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
ax.set_title('Clustered Hits (Predicted Tracks)', y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x', 'y', 'z']]
    t = arrange_track(t)
    ax2.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax2.set_xlabel('z (mm)')
ax2.set_ylabel('x (mm)')
ax2.set_zlabel('y (mm)')
ax2.set_title('True Tracks', y=-.15, size=20)

plt.show()
X = hits[['x', 'y', 'z']]
scaler = MaxAbsScaler().fit(X)
X = scaler.transform(X)
normalizer = Normalizer(norm='l2').fit(X)
X = normalizer.transform(X)

eps = 0.0022
min_samp = 3
db = DBSCAN(eps=eps, min_samples=min_samp, metric='euclidean').fit(X)
labels = db.labels_

clustering = pd.DataFrame()
clustering['hit_id'] = truth['hit_id']
clustering['track_id'] = labels

score = score_event(truth, clustering)
print('track-ml custom metric score:', round(score, 4))

labels_true = truth['particle_id']
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print('\nOTHER CLUSTERING RESULTS:')
print('Estimated number of clusters: %d' % n_clusters)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
rej_perc = list(labels).count(-1) / float(hits.shape[0]) * 100
rej_perc = round(rej_perc, 2)
print ("Rejected samples %:", str(rej_perc) + '%')
rejected_count = list(labels).count(-1)
print ("Rejected samples:", rejected_count)
print ("Total samples:", hits.shape[0])
print ("Clustered samples:", hits.shape[0] - list(labels).count(-1), '\n')

print ('WITHOUT REJECTED SAMPLES:')
labels_true_wr = labels_true[labels != -1]
labels_wr = labels[labels != -1]
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true_wr, labels_wr))
print(("Completeness: %0.3f" % metrics.completeness_score(labels_true_wr, labels_wr)), '\n')
tracks = truth.particle_id.unique()[1::100]
fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(121,projection='3d')

tracks_hit_ids = truth[truth['particle_id'].isin(tracks)]['hit_id'] # all hits in tracks
clusters = clustering[clustering['hit_id'].isin(tracks_hit_ids)].track_id.unique() # all clusters containing the hits in tracks
for cluster in clusters:
    cluster_hit_ids = clustering[clustering['track_id'] == cluster]['hit_id'] # all hits in cluster
    plot_hit_ids = list(set(tracks_hit_ids) & set(cluster_hit_ids))
    t = hits[hits['hit_id'].isin(plot_hit_ids)][['x', 'y', 'z']]
    if cluster == -1:
        ax.plot3D(t.z, t.x, t.y, '.', ms=10, color='black')
    else:
        ax.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax.set_xlabel('z (mm)')
ax.set_ylabel('x (mm)')
ax.set_zlabel('y (mm)')
ax.set_title('Clustered Hits (Predicted Tracks)', y=-.15, size=20)

ax2 = fig.add_subplot(122,projection='3d')
for track in tracks:
    hit_ids = truth[truth['particle_id'] == track]['hit_id']
    t = hits[hits['hit_id'].isin(hit_ids)][['x', 'y', 'z']]
    t = arrange_track(t)
    ax2.plot3D(t.z, t.x, t.y, '.-', ms=10)
    
ax2.set_xlabel('z (mm)')
ax2.set_ylabel('x (mm)')
ax2.set_zlabel('y (mm)')
ax2.set_title('True Tracks', y=-.15, size=20)

plt.show()