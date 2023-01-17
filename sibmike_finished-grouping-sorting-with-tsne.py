import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/train.csv').drop(columns=['ID'])
def build_histograms(df):
    df_X = (df.replace(0, np.nan).apply(np.log) * 10).round()
    start = int(df_X.min().min())
    stop = int(df_X.max().max())
    return pd.DataFrame(data={f'bucket{cnt}': (df_X == cnt).sum() for cnt in range(start, stop + 1)})
df = build_histograms(train)
tsne_res = TSNE(n_components=2, verbose=10, 
                perplexity=40, early_exaggeration=60, 
                learning_rate=150).fit_transform(df)
FEATURES40 = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', 
              '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 
              'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', 
              '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992', 
              'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', 
              '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', 
              '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', 
              '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']

def get_int_cols(df):
    return df.columns[df.dtypes == np.int64]

def get_colors(df):
    colors = pd.Series(index=df.columns, data='b')
    colors[FEATURES40] = 'y'
    colors[get_int_cols(train)] = 'g'
    colors['target'] = 'red'   
    return colors
#Simple function to find centroid of a cluster:

def centroids(X, lbls):
    
    centroids = np.zeros((len(np.unique(lbls)), 2))
    
    for l in np.unique(lbls):
        mask = lbls == l
        centroids[l] = np.mean(X[mask], axis=0)
    
    return centroids
# Sclae and clusterize, we use DBSCAN as it does not assume that cluster is convex:
X = StandardScaler().fit_transform(tsne_res)
db = DBSCAN(eps=0.085, min_samples=15).fit(X)
# Take a look at clusters' sizes:
unique, counts = np.unique(db.labels_, return_counts=True)
np.unique(counts, return_counts=True)
cluster_mask  = (counts <= 157) #& (counts >= 35)
dot_mask = np.array([(l in unique[cluster_mask]) for l in db.labels_])
np.sum(dot_mask)
fig, ax = plt.subplots(figsize=(20, 20))

# Split meaningful clusters from the noise:
plt.scatter(X[~dot_mask][:, 0], X[~dot_mask][:, 1], c='g')
plt.scatter(X[dot_mask][:, 0], X[dot_mask][:, 1], c='b')

# Add centroids to the picture:
db_centroids = centroids(X, db.labels_)
plt.scatter(db_centroids[cluster_mask][:, 0], 
            db_centroids[cluster_mask][:, 1], c='r')
plt.scatter(db_centroids[~cluster_mask][:, 0], 
            db_centroids[~cluster_mask][:, 1], c='orange')
tsne_no_noise = TSNE(n_components=2, verbose=0, 
                 perplexity=20, early_exaggeration=50, 
                 learning_rate=150).fit_transform(df[dot_mask])
#vis_x = tsne_no_noise[:, 0]
#vis_y = tsne_no_noise[:, 1]
#plt.figure(figsize=(20,20))
#plt.scatter(vis_x, vis_y, c=get_colors(train)[dot_mask]);
# Red = target, yellow = leak, green = ints, blue = floats
X4 = StandardScaler().fit_transform(tsne_no_noise)
dbf = DBSCAN(eps=0.095, min_samples=25).fit(X4)
# Mask labels for xlusters of size 40:
uniquef, countsf = np.unique(dbf.labels_, return_counts=True)
np.unique(countsf, return_counts=True)
#Simple function to find centroid of a cluster:

def centroids(X, lbls):
    
    cds_array = np.zeros((len(np.unique(lbls)), 2))
    
    for i, l in enumerate(np.unique(lbls)):
        mask = lbls == l
        cds_array[i] = np.mean(X[mask], axis=0)
    
    return cds_array
dbs_maskf  = (countsf <= 45) & (countsf >= 30)
dot_maskf = np.array([(l in uniquef[dbs_maskf]) for l in dbf.labels_])
dbf_c = centroids(X4, dbf.labels_)

# Add centroids to the picture:
fig, ax = plt.subplots(figsize=(20, 20))
plt.scatter(X4[~dot_maskf][:, 0], X4[~dot_maskf][:, 1], c='g')
plt.scatter(X4[dot_maskf][:, 0], X4[dot_maskf][:, 1], c='b')

plt.scatter(dbf_c[dbs_maskf][:, 0], dbf_c[dbs_maskf][:, 1], c='r')
plt.scatter(dbf_c[~dbs_maskf][:, 0], dbf_c[~dbs_maskf][:, 1], c='orange')
groups = pd.DataFrame([dbf.labels_[dot_maskf]]).T
groups.index = df.index[dot_mask][dot_maskf]

g_list = []
for g in pd.unique(groups[0]):
    g_list += [list(groups[groups[0] == g].index)]  

len(g_list), len(g_list[0])
with open('groups.txt', 'w') as file_handler:
    for item in g_list:
        file_handler.write("{}\n".format(item))
groups.shape
groups['x'] = X4[dot_maskf][:, 0]
groups['y'] = X4[dot_maskf][:, 1]
groups.columns = ['group', 'x', 'y']
def align_m(v1,v2):
    #Returns rotation matrix to align v1 to v2:
    
    # Unit vectors:
    x1, y1 = v1/np.sqrt(np.dot(v1,v1))
    x2, y2 = v2/np.sqrt(np.dot(v2,v2))
    
    #Cos of the angle between two vectors:
    cosv = x1*x2+y1*y2
    
    #Sin of the angle between two vectors:
    sinv = x1*y2-x2*y1
    
    #Rotation
    rotation_matrix = np.matrix([[cosv, -sinv],[sinv, cosv]])  
    
    return rotation_matrix
def direction(a):
    #Returns direction of a group:
    
    rng = np.max(a, axis=0) - np.min(a, axis=0)
    if rng[0] > rng[1]:
        a = a[a[:, 0].argsort()]
    else:
        a = a[a[:, 1].argsort()]
    
    return a[-1] - a[0]
#Lets align all groups with x-axis:

for g in groups['group'].unique():
    
    # Extract dots:
    t_dots = np.copy(groups[groups['group']==g].iloc[:,1:])

    # Find center of the group:
    t_center = np.mean(t_dots, axis=0)

    # Find direction of the goup and calculate rotation matrix:
    rotate = align_m(direction(t_dots),[1,0])

    # Rotate all dots:
    for i, d in enumerate(t_dots - t_center):
        t_dots[i] = (rotate*np.matrix(d).T).A1 + t_center

    # Save rotated dots in df:
    groups.loc[groups[groups['group']==g].index, ['x','y']] = t_dots
# Plot rotated groups:
fig, ax = plt.subplots(figsize=(15, 15))
plt.scatter(groups['x'], groups['y'], c='g')
#Save the labels, we'll use them a lot:
grps = groups['group'].unique()

#Array to hold standard deviations:
xy_std = np.zeros((len(grps), 2))

# Find standard deviations for every group:
for i, g in enumerate(grps):
    xy_std[i] = groups[groups['group']==g].describe()[['x','y']].loc['std',:]
# Mask groups that have standard deviations considerably higher than the mean:
k = 1.0
std_mask = xy_std[:,1] < xy_std.mean(axis=0)[1]*k

# Mask dots, plot dots:
dots_std_mask = groups['group'].apply(lambda x: x in grps[std_mask])

fig, ax = plt.subplots(figsize=(20, 20))

plt.scatter(groups[~dots_std_mask]['x'], groups[~dots_std_mask]['y'], c='g')
plt.scatter(groups[dots_std_mask]['x'], groups[dots_std_mask]['y'], c='b')
grps[std_mask]
groups[groups['group']==8].sort_values(by=['x'], ascending=False).index
with open('sorted_groups.txt', 'w') as file_handler:
    for good_group in grps[std_mask]:
        sorted_group = list(groups[groups['group']==good_group]\
                            .sort_values(by=['x'], ascending=False).index)
        file_handler.write("{}\n".format(sorted_group))
C = [1757, 3809,  511, 3798,  625, 3303, 4095, 1283, 4209, 1696, 3511, 816,  
     245, 1383, 2071, 3492,  378, 2971, 2366, 4414, 2790, 3979, 193, 1189, 
     3516,  810, 4443, 3697,  235, 1382, 4384, 3418, 4396, 921, 3176,  650]
g_list = []
for good_group in grps[std_mask]:
    g_list += [list(groups[groups['group']==good_group]\
                          .sort_values(by=['x'], ascending=False).index)]
cols_mask = [item for sublist in g_list for item in sublist]

col_labels = [[i]*len(g) for i, g in enumerate(g_list)]
col_labels = [item for sublist in col_labels for item in sublist]
len(col_labels), len(cols_mask)
Y = train.iloc[C,:][cols_mask].T
Y['labels'] = col_labels

sorted_col = Y.sort_values(list(C))
writer = pd.ExcelWriter('eyeball_notes.xlsx')
sorted_col.to_excel(writer,'Sheet1')
writer.save()