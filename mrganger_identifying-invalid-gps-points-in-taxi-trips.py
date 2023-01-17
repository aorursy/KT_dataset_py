import numpy as np

import pandas as pd

import geopandas as gpd

import json

from shapely.geometry import LineString, Point



def load_data(fname, nrows):

    df = pd.read_csv(fname, nrows=nrows)

    df['traj'] = json.loads('[' + df.POLYLINE.str.cat(sep=',') + ']')

    df = df[df.traj.str.len() > 1].copy()

    df['lines'] = gpd.GeoSeries(df.traj.apply(LineString))

    return gpd.GeoDataFrame(df, geometry='lines')



df = load_data('../input/train.csv', nrows=1000)

df.head()
%matplotlib inline

import matplotlib

matplotlib.rcParams['figure.figsize'] = [15,8]

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")



df.lines.plot(figsize=[15,15]);

plt.xlabel('Longitude')

plt.ylabel('Latitude');
df.iloc[[2]].lines.plot(figsize=[15,15]);

plt.xlabel('Longitude')

plt.ylabel('Latitude');
df.iloc[[3]].lines.plot(figsize=[15,15]);

plt.xlabel('Longitude')

plt.ylabel('Latitude');
from sklearn.neighbors import DistanceMetric

metric = DistanceMetric.get_metric('haversine')

def haversine(x):

    return metric.pairwise(np.radians(x)[:,::-1])

R = 6371 # radius of earth in km

dt = 15/3600 # coordinates are reported in 15 second intervals



def velocities(coords):

    return R/dt*np.diag(haversine(coords)[1:])



def plot_cdf(ax, values, xlabel, logscale=False):

    ax.set_ylabel('F(x)')

    ax.set_xlabel(xlabel)

    sns.lineplot(ax=ax, x=np.sort(values), y=np.arange(1,len(values)+1)/len(values), palette='tab10')

    if logscale:

        plt.xscale('log')



fig, ax = plt.subplots()

plot_cdf(ax, velocities(df.traj[2]), 'Speed (km/h)')

plot_cdf(ax, velocities(df.traj[3]), 'Speed (km/h)')
def velocity_graph(ax, coords):

    n = len(coords)

    dist = haversine(coords)

    interval = dt*np.abs(np.arange(n)[:,None] - np.arange(n)).clip(1)

    vel = R*dist/interval

    sns.heatmap(vel, ax=ax, square=True, robust=True)

    ax.set_title('Velocity (km/h)')

    ax.set_xlabel('GPS Sample Index')

    ax.set_ylabel('GPS Sample Index')



fig, axes = plt.subplots(1, 2, figsize=[25,10])

fig.suptitle("Average Velocities Between GPS Samples")

velocity_graph(axes[0], df.traj[2])

velocity_graph(axes[1], df.traj[3])
df.iloc[[2,3]].plot(figsize=[15,15]);

plt.xlabel('Longitude')

plt.ylabel('Latitude');
fig, ax = plt.subplots(1, 2, figsize=[25,10])

velocity_graph(ax[0], df.traj[45])

velocity_graph(ax[1], df.traj[39])
df.iloc[[45,39]].plot(figsize=[15,15]);

plt.xlabel('Longitude')

plt.ylabel('Latitude');
def offset_distances(coords, offset):

    dist = R*haversine(coords)

    dists = np.diag(dist[offset:])

    return dists



def plot_dists(ax, di):

    dists = np.hstack(df.traj.apply(lambda x: offset_distances(x, di)).values)

    sns.distplot(dists, ax=ax, kde=False, bins=300)

    ax.set_title('Time Offset: {} min'.format(di*dt*60))

    ax.set_xlabel('Distance (km)')

    ax.set_ylabel('Count')



fig, axes = plt.subplots(2,2, figsize=[20,17])

plot_dists(axes[0,0], 1)

plot_dists(axes[0,1], 10)

plot_dists(axes[1,0], 40)

plot_dists(axes[1,1], 100)
def dist_sequence(coords):

    n = len(coords)

    dist = R*metric.pairwise(np.radians(coords)).ravel()

    offsets = (np.arange(n)[:,None] - np.arange(n)).ravel()

    return pd.DataFrame([offsets[offsets>0]*dt*60,dist[offsets>0]], index=['time_offset', 'distance']).T



dist_ungrouped = pd.concat(df.traj.apply(dist_sequence).values).set_index('time_offset')
dists = np.sqrt((dist_ungrouped**2).groupby('time_offset').mean()/2)

sns.lineplot(data=dists)

plt.xlabel('Time Offset (minutes)')

plt.ylabel('Distance (km)');
def fit_rational(x,y,w=1):

    ws = np.sqrt(w)

    (a,b),_,_,_ = np.linalg.lstsq(np.column_stack([x,-y])*ws[:,None], x*y*ws, rcond=None)

    return a*x/(x+b), (a,b)



dists['curve'], coeffs = fit_rational(dists.index.values, dists.distance.values, ((1+np.arange(len(dists)))/(1+len(dists)))**-3)

print('a:', coeffs[0])

print('b:', coeffs[1])

print('a/b:', coeffs[0]/coeffs[1])

sns.lineplot(data=dists);

plt.xlabel('Time Offset (minutes)')

plt.ylabel('Distance (km)');
def likelihood(coords, ab):

    n = len(coords)

    a,b = coeffs

    dist = R*metric.pairwise(np.radians(coords))

    time = dt*60*np.abs(np.arange(n)[:,None] - np.arange(n))

    sigma = a*time/(time + b) + np.eye(n)

    lr = -0.5*(dist**2/sigma**2).sum(axis=1)

    return lr



def norm_lr(lr):

    return (lr-lr.max())/len(lr)



def plot_likelihood(ax, coords):

    lr = likelihood(coords,coeffs)

    lr = norm_lr(lr)

    sns.lineplot(x=np.arange(len(coords)),y=lr, ax=ax);

    ax.set_xlabel('Sequence ID')

    ax.set_ylabel('Normalized Likelihood')



fig,ax = plt.subplots(2,2, figsize=[20,10])

plot_likelihood(ax[0,0], df.traj[2])

plot_likelihood(ax[0,1], df.traj[3])

plot_likelihood(ax[1,0], df.traj[45])

plot_likelihood(ax[1,1], df.traj[39])
n = int(df.traj.str.len().quantile(0.9))

thresh = -2

invalid = df.traj.apply(lambda t: (norm_lr(likelihood(t,coeffs)) < thresh)[:n].tolist() + [False]*(n-len(t))).values.tolist()

plt.figure(figsize=[20,300])

sns.heatmap(data=invalid, cbar=False);

plt.xlabel('Sequence ID')

plt.ylabel('Row ID');
bad_routes = df.traj.apply(lambda t: (norm_lr(likelihood(t,coeffs)) < thresh).any()).values

print("Routes with invalid points: {} / 1000".format(bad_routes.sum()))
def spot_check(i):

    coords = np.array(df.iloc[i].traj)

    bad = norm_lr(likelihood(coords, coeffs)) < thresh

    df.iloc[[i]].plot()

    plt.scatter(x=coords[bad,0],y=coords[bad,1], color='red')

    plt.show()

spot_check(15)
spot_check(690)
spot_check(941)
spot_check(848)
spot_check(924)
spot_check(603)
spot_check(730)
spot_check(629)
spot_check(625)
df[~bad_routes].lines.plot(figsize=[15,15]);