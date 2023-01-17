from sys import argv

import numpy as np
from timeit import default_timer as timer

import pickle
import torch
import os
import pandas as pd

os.listdir('../input')
def normalize(points):
    """
    Return the normalized version of a given vector of points.

    For a given array of n-dimensions, normalize each dimension by removing the
    initial offset and normalizing the points in a proportional interval: [0,1]
    on y, maintining the original ratio on x.
    """
    ratio = (points.x.max() - points.x.min()) / (points.y.max() - points.y.min()), 1
    ratio = np.array(ratio) / max(ratio)
    norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    return norm.apply(lambda p: ratio * p, axis=1)

def generate_network(size):
    """
    Generate a neuron network of a given size.

    Return a vector of two dimensional points in the interval [0,1].
    """
    return torch.from_numpy(np.random.rand(size, 2))


def select_closest(candidates, origin):
    """Return the index of the closest candidate to a given point."""
    _, ind = torch.min(euclidean_distance(candidates, origin),0)
    return ind

def euclidean_distance(a, b):
    """Return the array of distances of two numpy arrays of points."""
    return torch.norm(a - b, p=2, dim=1)

def route_distance(cities):
    """Return the cost of traversing a route of cities in a certain order."""
    points = cities[['x', 'y']]
    distances = euclidean_distance(points, np.roll(points, 1, axis=0))
    return np.sum(distances)

def get_neighborhood(center, radix, domain):
    """Get the range gaussian of given radix around a center index."""

    # Impose an upper bound on the radix to prevent NaN and blocks
    if radix < 1:
        radix = 1

    # Compute the circular network distance to the center
    #deltas = np.absolute(center - np.arange(domain))
    deltas = torch.abs(center.float().cuda() - torch.arange(domain).float().cuda())
    #distances = np.minimum(deltas, domain - deltas)
    deltas = deltas.cpu().numpy()
    #print(deltas)
    distances = torch.from_numpy(np.minimum(deltas, domain - deltas)).cuda()
    # Compute Gaussian distribution around the given center
    #return np.exp(-(distances*distances) / (2*(radix*radix)))
    return torch.exp(-(distances*distances) / (2*(radix*radix)))
def read_tsp(filename):
    """
    Read a file in .tsp format into a pandas DataFrame

    The .tsp files can be found in the TSPLIB project. Currently, the library
    only considers the possibility of a 2D map.
    """
    with open(filename) as f:
        node_coord_start = None
        dimension = None
        lines = f.readlines()

        # Obtain the information about the .tsp
        i = 0
        while not dimension or not node_coord_start:
            line = lines[i]
            if line.startswith('DIMENSION :'):
                dimension = int(line.split()[-1])
            if line.startswith('NODE_COORD_SECTION'):
                node_coord_start = i
            i = i+1

        print('Problem with {} cities read.'.format(dimension))

        f.seek(0)

        # Read a data frame out of the file descriptor
        cities = pd.read_csv(
            f,
            skiprows=node_coord_start + 1,
            sep=' ',
            names=['city', 'y', 'x'],
            dtype={'city': str, 'x': np.float32, 'y': np.float32},
            header=None,
            nrows=dimension
        )

        # cities.set_index('city', inplace=True)

        return cities
    
def normalize(points):
    """
    Return the normalized version of a given vector of points.

    For a given array of n-dimensions, normalize each dimension by removing the
    initial offset and normalizing the points in a proportional interval: [0,1]
    on y, maintining the original ratio on x.
    """
    ratio = (points.x.max() - points.x.min()) / (points.y.max() - points.y.min()), 1
    ratio = np.array(ratio) / max(ratio)
    norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    return norm.apply(lambda p: ratio * p, axis=1)


problem = read_tsp("../input/santa/santa_test.tsp")
cities=torch.from_numpy(normalize(problem[['x','y']]).values).cuda()
n = cities.size()[0] * 8
network = generate_network(n).cuda()
iterations=1000
learning_rate=0.8
for i in range(iterations):
    if not i % 100:
        print('\t> Iteration {}/{}'.format(i, iterations), end="\r")
    # Choose a random city
    #print("=============")
    start = timer()
    city=cities[np.random.randint(0,cities.size()[0]),:]
    winner_idx = select_closest(network, city)
    gaussian = get_neighborhood(winner_idx, n//10, network.shape[0])
    network += gaussian[:,np.newaxis] * learning_rate * (city - network)
    end = timer()
    print(end - start)
    #print("=============")

    # Decay the variables
    learning_rate = learning_rate * 0.99997
    n = n * 0.9997

    # Check for plotting interval
    #if not i % 1000:
    #    plot_network(cities, network, name='diagrams/{:05d}.png'.format(i))

    # Check if any parameter has completely decayed.
    if n < 1:
        print('Radius has completely decayed, finishing execution',
        'at {} iterations'.format(i))
        break
    if learning_rate < 0.001:
        print('Learning rate has completely decayed, finishing execution',
        'at {} iterations'.format(i))
        break
else:
    print('Completed {} iterations.'.format(iterations))
def select_closest_np(candidates, origin):
    """Return the index of the closest candidate to a given point."""
    return euclidean_distance_np(candidates, origin).argmin()

def euclidean_distance_np(a, b):
    """Return the array of distances of two numpy arrays of points."""
    return np.linalg.norm(a - b, axis=1)

def route_distance_np(cities):
    """Return the cost of traversing a route of cities in a certain order."""
    points = cities[['x', 'y']]
    distances = euclidean_distance_np(points, np.roll(points, 1, axis=0))
    return np.sum(distances)

def get_route(problem,cities, network):
    """Return the route computed by a network."""
    problem[['x', 'y']] = cities
    problem['winner'] = problem[['x', 'y']].apply(
        lambda c: select_closest(network, torch.from_numpy(c)),
        axis=1, raw=True)

    return cities.sort_values('winner').index
route = get_route(problem,cities.cpu().numpy(), network.cpu().numpy())

