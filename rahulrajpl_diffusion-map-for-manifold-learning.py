import matplotlib.pyplot as plt

import numpy as np

import random

import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances

from scipy.linalg import eigh

import plotly.graph_objs as go

from plotly.subplots import make_subplots

import warnings

warnings.filterwarnings('ignore')
def find_diffusion_matrix(X=None, alpha=0.15):

    """Function to find the diffusion matrix P

        

        >Parameters:

        alpha - to be used for gaussian kernel function

        X - feature matrix as numpy array

        

        >Returns:

        P_prime, P, Di, K, D_left

    """

    alpha = alpha

        

    dists = euclidean_distances(X, X)

    K = np.exp(-dists**2 / alpha)

    

    r = np.sum(K, axis=0)

    Di = np.diag(1/r)

    P = np.matmul(Di, K)

    

    D_right = np.diag((r)**0.5)

    D_left = np.diag((r)**-0.5)

    P_prime = np.matmul(D_right, np.matmul(P,D_left))



    return P_prime, P, Di, K, D_left
def find_diffusion_map(P_prime, D_left, n_eign=3):

    """Function to find the diffusion coordinates in the diffusion space

        

        >Parameters:

        P_prime - Symmetrized version of Diffusion Matrix P

        D_left - D^{-1/2} matrix

        n_eigen - Number of eigen vectors to return. This is effectively 

                    the dimensions to keep in diffusion space.

        

        >Returns:

        Diffusion_map as np.array object

    """   

    n_eign = n_eign

    

    eigenValues, eigenVectors = eigh(P_prime)

    idx = eigenValues.argsort()[::-1]

    eigenValues = eigenValues[idx]

    eigenVectors = eigenVectors[:,idx]

    

    diffusion_coordinates = np.matmul(D_left, eigenVectors)

    

    return diffusion_coordinates[:,:n_eign]
# Reference: https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#

def make_plot(N=1000, rseed=42):

    # Make a plot with "HELLO" text; save as PNG

    fig, ax = plt.subplots(figsize=(4, 1))

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ax.axis('off')

    ax.text(0.5, 0.4, 'S', va='center', ha='center', size=85)

    fig.savefig('plot.png')

    plt.close(fig)

    fig.show()

    # Open this PNG and draw random points from it

    from matplotlib.image import imread

    data = imread('plot.png')[::-1, :, 0].T

    rng = np.random.RandomState(rseed)

    X = rng.rand(5 * N, 2)

    i, j = (X * data.shape).astype(int).T

    mask = (data[i, j] < 1)

    X = X[mask]

    X[:, 0] *= (data.shape[0] / data.shape[1])

    X = X[:N]

    return X[np.argsort(X[:, 0])]
# Generate shape

X = make_plot(3000)

data=go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(

        size=3,color=X[:,1],opacity=1,colorscale='Viridis'))

layout=go.Layout(title_text="Original shape", title_x=0.5, title_y=0.8,title_font_size=12)

fig = go.Figure(data=[data], layout=layout)

fig.update_layout(height=400, width=400,showlegend=False)

fig.update_xaxes(showticklabels=False)

fig.update_yaxes(showticklabels=False)

fig.show()
# Adding new dimension to the original geometric structure

newX = pd.DataFrame(X)

newX['dimension1'] = [random.uniform(0.1,0.5) for _ in range(len(X))]

newX = np.asarray(newX)
def plot_3Dfigure(newX, title='Datapoints'):

    data = go.Scatter3d(x=newX[:,0], y=newX[:,1], z=newX[:,2], mode='markers', marker=dict(

            size=2,color=newX[:,1],opacity=0.7,colorscale='Viridis'))

    layout = go.Layout(title_text=title,title_x=0.5,title_y=0.8,title_font_size=12)

    fig = go.Figure(data=[data], layout=layout)

    fig.update_layout(showlegend=False)

    fig.update_xaxes(showticklabels=False)

    fig.update_yaxes(showticklabels=False)

    fig.update_layout(scene = dict(

                    xaxis = dict(title= '', ticks= '', showticklabels= False,),

                    yaxis = dict(title= '', ticks= '', showticklabels= False,),

                    zaxis = dict(title= '', ticks= '', showticklabels= False,),

                    ))

                  

    fig.show()
plot_3Dfigure(newX, title='Synthetic 3D Datapoints')
def plot_2Dsub_figures(d_map, alpha_values, title='Diffused points'):

    subplot_titles=[f'α={round(a,4)}' for a in alpha_values]

    fig = make_subplots(rows=2, cols=5,subplot_titles=subplot_titles)

    for i in range(1,3):

        for j in range(1,6):

            dmap_idx = i+j-1

            fig.add_trace(

                go.Scatter(x=d_map[dmap_idx][:,0], y=d_map[dmap_idx][:,1], mode='markers', marker=dict(

                size=3,color=d_map[dmap_idx][:,1],opacity=0.8,colorscale='Viridis')),row=i, col=j)



    fig.update_layout(title_text=title, title_x=0.5)

    fig.update_xaxes(showticklabels=False)

    fig.update_yaxes(showticklabels=False)

    fig.update_layout(height=500, width=1000, showlegend=False)

    fig.show()
def apply_diffusions(alpha_start=0.001, alpha_end= 0.009, title='Diffused points'):

    d_maps = []

    alpha_values = np.linspace(alpha_start, alpha_end, 10)

    for alpha in alpha_values:

        P_prime, P, Di, K, D_left = find_diffusion_matrix(newX, alpha=alpha)

        d_maps.append(find_diffusion_map(P_prime, D_left, n_eign=2))

    return d_maps, alpha_values

d_maps, alpha_values = apply_diffusions(0.01, 0.09)

plot_2Dsub_figures(d_maps,alpha_values)
d_maps, alpha_values = apply_diffusions(0.1, 0.9)

plot_2Dsub_figures(d_maps,alpha_values)