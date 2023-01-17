%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



points = pd.DataFrame(np.random.randn(100000,2), columns=['x','y'])

points['dist_std'] = np.max(np.abs(points-1), axis=1)

fig, ax = plt.subplots(2, 2, figsize=[20,20])

sns.scatterplot(x='x', y='y', ax=ax[0,0], data=points.query('dist_std < 8')[:300]);

sns.scatterplot(x='x', y='y', ax=ax[0,1], data=points.query('dist_std < 2')[:300]);

sns.scatterplot(x='x', y='y', ax=ax[1,0], data=points.query('dist_std < 0.5')[:300]);

sns.scatterplot(x='x', y='y', ax=ax[1,1], data=points.query('dist_std < 0.125')[:300]);
def vol_ball(n,r=1):

    from scipy.special import gammaln

    return np.exp(np.log(np.pi)*n/2 - gammaln(n/2+1) + np.log(r)*n)



def vol_cube(n,r=1):

    return (2*r)**n



def distance_stats(dim=1, k=2, buffer=5, samples=1000):

    p = vol_ball(dim)/vol_cube(dim)

    n = k/p

    m = int(np.ceil(n+5*np.sqrt(n)))

    pop = np.random.uniform(-1, 1, size=(samples,m,dim))

    dist = np.sort(np.linalg.norm(pop, axis=2), axis=1)

    return dist[:,:k-1]/dist[:,[k-1]]
def plot_dim(dim):

    d = distance_stats(dim=dim, k=5, samples=2000).ravel()

    plt.plot(np.sort(d), np.linspace(0,1,len(d)+1)[1:], label=str(dim))

    plt.legend(title='Dimension')

    plt.ylabel('CDF')

    plt.xlabel('Relative Distance')



plt.figure(figsize=[10,10])

plt.title('CDF of relative distances of $k$-NN in a uniform distribution of different dimensions.')

plot_dim(1)

plot_dim(2)

plot_dim(3)

plot_dim(4)
def dimensional_likelihood(maxdim, dists):

    dims = np.arange(1,maxdim+1)

    l = (np.expand_dims(np.log(dists), axis=-1)*(dims-1) + np.log(dims)).sum(axis=-2)

    return np.exp(l-np.expand_dims(l.max(axis=-1), axis=-1))
ks = [2,5,10,20,30]

max_dim = 20



def plot_dim(*dims):

    fig, axes = plt.subplots(len(dims), len(ks), figsize=[20,3*len(dims)], sharex=True, sharey=True)

    axes[-1,0].set_xlabel('Dimension')

    axes[-1,0].set_ylabel('Relative Likelihood')

    

    for k, axrow in zip(ks, axes):

        for dim, ax in zip(dims, axrow):

            ax.bar(np.arange(1,max_dim+1), dimensional_likelihood(max_dim, distance_stats(dim=dim, k=k, samples=1)[0]));

            ax.set_title('$d$ = {}, $k$ = {}'.format(dim,k))
plot_dim(1,2,5,10,15)
def test_mixed(k, samples=1000):

    square = np.random.uniform(-2, 0, size=(samples, 2))

    line = np.random.uniform(0, 1, size=(samples,1))*[1,1]

    x = np.vstack([square,line])

    

    from scipy.spatial import cKDTree as KDTree

    tree = KDTree(x)

    dists = tree.query(x, k=k+1)[0][:,1:]

    dists = dists/dists[:,[-1]]

    

    import pandas as pd

    

    df = pd.DataFrame(x, columns=['x','y'])

    dim = dimensional_likelihood(2, dists).argmax(axis=1)

    plt.figure(figsize=[15,15]);

    for i in range(np.max(dim)+1):

        sns.scatterplot(x=x[dim==i,0], y=x[dim==i,1])

    plt.legend(np.arange(1,2+np.max(dim)).astype(str), title='Estimated Dimension')
test_mixed(30, samples=100);
def density_estimate(loc, x, k):

    d = x.shape[1]

    v = vol_ball(d)

    

    from scipy.spatial import cKDTree as KDTree

    tree = KDTree(x)

    r = tree.query(loc.reshape(-1,d),k, n_jobs=-1)[0][:,-1].reshape(*loc.shape[:-1])

    

    return (k/v)*r**-d
def test_for_size(size):

    k = int(np.sqrt(size))

    x = np.random.uniform(-0.5,0.5, size=(size,2))

    g = np.moveaxis(np.mgrid[-1:1:0.005,-1:1:0.005], 0,-1)

    d = density_estimate(g, x, k)/len(x)



    fig, ax = plt.subplots(1, 2, figsize=[20,10])

    sns.scatterplot(ax=ax[0], x=x[:,0], y=x[:,1])

    ax[0].set_xlim([-1,1])

    ax[0].set_ylim([-1,1])

    ax[1].imshow(d, origin='lower', vmin=0, vmax=4.0, extent=(-1,1,-1,1));

    ax[1].set_title('k={}'.format(k))

    ax[1].grid(False)
test_for_size(100)
test_for_size(1000)
test_for_size(10000)
test_for_size(1000000)
def relative_error(ks, size):

    k = int(np.sqrt(size))

    x = np.random.uniform(-0.5,0.5, size=(size,2))

    g = np.moveaxis(np.mgrid[-0.25:0.25:0.005,-0.25:0.25:0.005], 0,-1)

    devs = [(density_estimate(g, x, k)/len(x)).std() for k in ks]

    

    plt.figure(figsize=[15,10])

    sns.lineplot(ks, devs)

    plt.yscale('log')

    plt.xscale('log')

    plt.ylabel('Relative Standard Deviation')

    plt.xlabel('k')



relative_error(2**np.arange(1,11), 100000)
def test_multi_scale(size):

    k = int(np.sqrt(size))

    x = np.random.uniform(-0.5,0.5, size=(size,2))

    n1 = size//3

    n2 = size*2//3

    x = np.vstack([x[:n1], 0.1*x[n1:n2], 0.01*x[n2:]])

    g = np.moveaxis(np.mgrid[-1:1:0.005,-1:1:0.005], 0,-1)

    d = density_estimate(g, x, k)/len(x)

    

    fig, ax = plt.subplots(1, 2, figsize=[20,10])

    sns.scatterplot(ax=ax[0], x=x[:n1,0], y=x[:n1,1])

    sns.scatterplot(ax=ax[0], x=x[n1:n2,0], y=x[n1:n2,1])

    sns.scatterplot(ax=ax[0], x=x[n2:,0], y=x[n2:,1])

    ax[0].set_xlim([-1,1])

    ax[0].set_ylim([-1,1])

    ax[1].imshow(np.log(d), origin='lower', extent=(-1,1,-1,1));

    ax[1].set_title('k={}'.format(k))

    ax[1].grid(False)
test_multi_scale(100000)