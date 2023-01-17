%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import pandas as pd

import numpy as np



R = 6371 # Radius of earth in km
def load_trees(fname):

    df = pd.read_csv(fname)

    return df[['latitude', 'longitude', 'spc_common']].copy()

tree_df = load_trees('../input/new_york_tree_census_2015.csv')

plt.figure(figsize=[10,10])

sns.scatterplot(x='longitude', y='latitude', data=tree_df.sample(10000));
def to_cartesian(latlon):

    lat,lon = np.radians(latlon.T)

    clat = np.cos(lat)

    return R*np.column_stack([clat*np.cos(lon), clat*np.sin(lon), np.sin(lat)])



def density_estimate(coords, k, extent, grid_res=300, logscale=True, q=0.99):

    from matplotlib.colors import LogNorm

    from scipy.spatial import cKDTree as KDTree

    tree = KDTree(to_cartesian(coords))

    x0,x1,y0,y1 = extent

    dx = (x1-x0)/grid_res

    dy = (y1-y0)/grid_res

    g = np.moveaxis(np.mgrid[x0:x1:dx,y0:y1:dy], 0, -1)

    r = tree.query(to_cartesian(g.reshape(-1,2)), k, n_jobs=-1)[0][:,-1]

    d = (k/np.pi)*r**-2

    d = d.reshape(*g.shape[:-1])

    

    if logscale:

        norm = LogNorm()

    else:

        norm = None

    plt.figure(figsize=[15,12])

    plt.imshow(d, origin='lower', extent=(y0,y1,x0,x1), vmax=np.quantile(d, q), aspect='auto', norm=norm)

    plt.colorbar()

    plt.title("Density estimate of trees recorded in NYC tree census, in trees / km$^2$")

    plt.grid(False)
density_estimate(tree_df[['latitude', 'longitude']].dropna().values, 25, [40.48,40.92,-74.3,-73.65], grid_res=800, logscale=False)
density_estimate(tree_df[['latitude', 'longitude']].dropna().values, 25, [40.6,40.7,-74.0,-73.9], grid_res=800, logscale=False)
density_estimate(tree_df[['latitude', 'longitude']].dropna().values, 25, [40.62,40.65,-73.98,-73.96], grid_res=800, logscale=False, q=0.99)
def dimensional_likelihood(maxdim, dists):

    dims = np.arange(1,maxdim+1)

    l = (np.expand_dims(np.log(dists), axis=-1)*(dims-1) + np.log(dims)).sum(axis=-2)

    return np.exp(l-np.expand_dims(l.max(axis=-1), axis=-1))



def plot_dimension_estimate(coords, k, extent, grid_res=300):

    from matplotlib.colors import LogNorm

    from scipy.spatial import cKDTree as KDTree

    tree = KDTree(to_cartesian(coords))

    x0,x1,y0,y1 = extent

    dx = (x1-x0)/grid_res

    dy = (y1-y0)/grid_res

    g = np.moveaxis(np.mgrid[x0:x1:dx,y0:y1:dy], 0, -1)

    r = tree.query(to_cartesian(g.reshape(-1,2)), k, n_jobs=-1)[0]

    rel = r[:,:-1]/r[:,[-1]]

    d = dimensional_likelihood(3, rel).argmax(axis=1).reshape(*g.shape[:-1])

    

    plt.figure(figsize=[12,12])

    plt.imshow(d, origin='lower', extent=(y0,y1,x0,x1), vmax=2, aspect='auto', cmap='gray')

    plt.title('Dimensionality estimate of trees from NYC tree census (black = 1, grey = 2, white = 3).')

    plt.grid(False)
plot_dimension_estimate(tree_df[['latitude', 'longitude']].dropna().values, 10, [40.48,40.92,-74.3,-73.65], grid_res=800)
plot_dimension_estimate(tree_df[['latitude', 'longitude']].dropna().values, 10, [40.6,40.7,-74.0,-73.9], grid_res=800)
plot_dimension_estimate(tree_df[['latitude', 'longitude']].dropna().values, 10, [40.62,40.65,-73.98,-73.96], grid_res=800)