import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
def load_data(path):
    df = pd.read_csv(path, index_col=0)
    df['minute_of_day'] = (df.time + 0) % (60*24)
    df.drop('time', axis=1, inplace=True)
    return df

def mapk(actual, predicted):
    matched_rows, matched_pos = np.where((predicted-actual[:,np.newaxis])==0)
    _, unique_row_idx = np.unique(matched_rows, return_index=True)
    return (1./(1+matched_pos[unique_row_idx])).sum()/len(actual)
data = load_data('../input/train.csv')
le = LabelEncoder().fit(data.place_id)
data.place_id = le.transform(data.place_id)
train, valid = train_test_split(data, test_size=0.05)
def cluster(df, min_events=10, normed=False):
    df.is_copy = False
    keep_ids = df.groupby('place_id').x.count() >= min_events
    sub = df.loc[keep_ids[df.place_id]][['x','y','place_id']].copy()
    if normed:
        sub[['x','y']] = sub.groupby('place_id')[['x','y']].transform(lambda coord: (coord - coord.mean()) / coord.std())
    clstr = DBSCAN(eps=0.5, min_samples=min_events)
    for place_id, subg in sub.groupby('place_id'):
        clstr.fit(subg[['x','y']])
        df.loc[df.place_id==place_id,'label'] = clstr.labels_
cluster(valid, normed=True) #takes a long time
cluster_means = train.loc[train.label!=-1].groupby(['place_id','label'])[['x','y']].mean()
cluster_stds = train.loc[train.label!=-1].groupby(['place_id','label'])[['x','y']].std()
n_t_bin = 48
sigma = 120
dt = 60*24/n_t_bin
t_bin = np.linspace(-60*24+dt/2,2*60*24-dt/2,n_t_bin*3)
t_bin_ = t_bin[n_t_bin:n_t_bin*2]
def day_bin(minutes_of_day):
    return np.exp(-(minutes_of_day[:,np.newaxis]-t_bin[:np.newaxis])**2/(2*sigma**2)).reshape(-1,3,n_t_bin).sum((0,1))
t_hist = train.loc[train.label!=-1].groupby(['place_id','label']).minute_of_day.apply(day_bin)
def xy_bin(df, n_bins_x, n_bins_y):
    df.is_copy = False
    xbin = np.floor(df.x/10*n_bins_x).astype(int)
    xbin[xbin==n_bins_x] = n_bins_x - 1
    ybin = np.floor(df.y/10*n_bins_y).astype(int)
    ybin[ybin==n_bins_y] = n_bins_y - 1
    if 'xybin' in df:
        df.loc[:,'xybin'] = ybin + xbin * n_bins_y
    else:
        df['xybin'] = ybin + xbin * n_bins_y
nx, ny = 50,50
xy_bin(train, nx, ny)
xy_bin(valid, nx, ny)
ids_in_xybins = train.groupby('xybin').place_id.unique()
ids_in_xybins = ids_in_xybins.append(pd.Series([np.array([])],np.setdiff1d(np.arange(nx*ny),ids_in_xybins.index.values)))
actual = []
predicted = []
for xyb, subg in valid.groupby('xybin'):
    place_ids = list(ids_in_xybins['xyb'])
    means = cluster_means.loc[place_ids]
    stds = cluster_stds.loc[place_ids]
    prob_x = np.exp(- (subg.x[:,np.newaxis] - means.x[np.newaxis,:])**2 / (2*stds.x[np.newaxis,:]**2)) / stds.x[np.newaxis,:]
    prob_y = np.exp(- (subg.y[:,np.newaxis] - means.y[np.newaxis,:])**2 / (2*stds.y[np.newaxis,:]**2)) / stds.y[np.newaxis,:]
    t_bin_num = np.abs(subg.minute_of_day[:,np.newaxis] - t_bin_[np.newaxis,:]).argmin(1)
    prob_t = np.array(list(t_hist.loc[place_ids])).T[t_bin_num]
    probs = prob_x*prob_y*prob_t
    idxs = np.argpartition(-probs,2)[:,:3]
    selector = np.tile(np.arange(len(subg)),(3,1)).T
    idxs = idxs[selector, (-probs[selector,idxs]).argsort()]
    actual.append(subg.place_id)
    predicted.append(np.array([v[0] for v in means.index.values])[idxs])
mapk(np.array([ac for acs in actual for ac in acs]),np.vstack(predicted))