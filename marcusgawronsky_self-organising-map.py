from typing import Optional, Tuple



import pandas as pd

import numpy as np

import networkx as nx



from sklearn.base import BaseEstimator, ClusterMixin

from sklearn.metrics import pairwise_distances

from sklearn.datasets import make_blobs, make_regression

import holoviews as hv



hv.extension('bokeh')
class SOM(BaseEstimator, ClusterMixin):

    def __init__(self, shape: Tuple[int] = (3, 3), 

                 lattice: str = 'hexagonal_lattice_graph', 

                 n_iter: int = 100, 

                 learning_rate: float = 0.25, 

                 cooperative_learning_rate: float = 0.125, 

                 n_jobs: Optional[int] = None):

        

        self.lattice = lattice

        self.shape = shape

        self.learning_rate = learning_rate

        self.cooperative_learning_rate = cooperative_learning_rate

        self.n_jobs = n_jobs

        self.n_iter = n_iter

        

    def fit(self, X: np.ndarray):

        # initialize graph

        self.graph_ = getattr(nx.generators.lattice, self.lattice)(*self.shape)

        

        # initialize weights

        mean, var = X.mean(0), X.var(0)

        for key in self.graph_.nodes:

            self.graph_.nodes[key]['weight'] = np.random.normal(mean, var)

            

        for schedule in range(1, self.n_iter):

            for x in X:

                weights_dict = nx.get_node_attributes(self.graph_, 'weight')

                self.weights_ = np.vstack(list(weights_dict.values()))



                # competition

                argmin = int(pairwise_distances(X = x.reshape(1, -1), Y = self.weights_,

                                                n_jobs=self.n_jobs)

                             .argmin())

                min_node_key = list(weights_dict.keys())[argmin]



                self.graph_.nodes[key]['weight'] -= self.learning_rate * (self.graph_.nodes[key]['weight'] - x) / schedule



                # cooperation

                for key in self.graph_.neighbors(min_node_key):

                    self.graph_.nodes[key]['weight'] -= self.cooperative_learning_rate * (self.graph_.nodes[key]['weight'] - x) / schedule

                    

    def predict(self, X):

        return (pairwise_distances(X = X,

                                   Y = self.weights_,

                                   n_jobs=self.n_jobs)

                .argmin(1))

        

            

            
def get_blobs_on_scurve(n_samples = 2500,

                        noise = 0.01,

                        centers=2):

    x_gaussian_blobs, y_gaussian_blobs = make_blobs(n_samples,  n_features=1, centers=centers)

    x_gaussian_blobs = x_gaussian_blobs.flatten()

    

    clipped_ = (x_gaussian_blobs - x_gaussian_blobs.min())/x_gaussian_blobs.max()



    t = 3 * np.pi * ( clipped_ - clipped_.mean() )

    x = np.sin(t)

    y = 2.0 * (np.random.rand(1, n_samples) - 0.5)

    z = np.sign(t) * (np.cos(t) - 1)



    X = np.column_stack((x.reshape(-1,1),

                         y.reshape(-1,1),

                         z.reshape(-1,1)))

    X += noise * np.random.randn(1, n_samples).reshape(-1,1)

    t = np.squeeze(t)

    

    return X, y_gaussian_blobs



X, y = get_blobs_on_scurve()
hv.extension('matplotlib')

(hv.Scatter3D(pd.DataFrame(X, columns=['x','y','z'])

              .assign(cluster=y), kdims=['x','y'], vdims=['z', 'cluster'])

 .opts(color='cluster', title='Blobs along S-curve Manifold'))
som = SOM((15, 15), 'grid_2d_graph', learning_rate= 0.2, cooperative_learning_rate=0.1, n_iter=25, n_jobs=None)

som.fit(X)
hv.extension('bokeh')

grid = np.zeros(som.shape)

for key in som.graph_.nodes:

    row, column = list(key)

    for neighbour in som.graph_.neighbors(key):

        grid[row, column] += pairwise_distances(X = som.graph_.nodes[key]['weight'].reshape(1, -1),

                                                Y = som.graph_.nodes[neighbour]['weight'].reshape(1, -1))

        

hv.Image(grid).opts(title='Density along Lattices')