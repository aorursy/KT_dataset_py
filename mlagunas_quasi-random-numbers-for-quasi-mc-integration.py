import math
import numpy as np

from bisect import bisect_left
from functools import reduce

from matplotlib import pyplot as plt
%matplotlib inline
def plotter(samples, grid=None, title='',):
    """
    Plots a Square with width and height one and draws the samples inside.
    If grid is provided it draws the given grid on the square.
    """

    if grid is not None:
        grid_i = grid[0]
        grid_j = grid[1]
        all_axis = np.zeros(samples.shape)
        all_axis[:, 1] = 1

    plt.scatter(samples[:, 0], samples[:, 1])
    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([-0.1, 1.1])
    axes.set_ylim([-0.1, 1.1])
    axes.set_aspect('equal')
    if grid is not None:
        for idx in range(len(grid_i)):
            plt.plot([grid_i[idx], grid_i[idx]], all_axis[idx], alpha=0.2, color='b', dashes=[6, 2])
        for idx in range(len(grid_j)):
            plt.plot(all_axis[idx], [grid_j[idx], grid_j[idx]], alpha=0.2, color='b', dashes=[6, 2])

        # add grid lines of the last elements of the square
        plt.plot([all_axis[1], all_axis[1]], all_axis[idx], alpha=0.2, color='b', dashes=[6, 2])
        plt.plot(all_axis[idx], [all_axis[1], all_axis[1]], alpha=0.2, color='b', dashes=[6, 2])

    plt.show()


def get_factors(n):
    """
    get factors and make it a set to remove duplicates
    then make it a lit to sort in in increasing order
    """
    
    factors = list(set(reduce(list.__add__, ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0))))
    factors.sort()

    # return all factors execept the original number (n)
    return factors[:-1]

def take_closest(my_list, number):
    """
    Assumes my_list is sorted. Returns closest value to number.
    If two numbers are equally close, return the smallest number.
    """

    pos = bisect_left(my_list, number)
    if pos == 0:
        return my_list[0]
    if pos == len(my_list):
        return my_list[-1]
    before = my_list[pos - 1]
    after = my_list[pos]
    if after - number < number - before:
        return after
    else:
        return before


def two_factors_to_n(n):
    """
    n is an integer.
    The function returns two integers that multiplied together give n -> int1 * int2 = n
    """

    factors = get_factors(n)
    int1 = take_closest(factors, math.sqrt(n))
    int2 = n // int1
    return int1, int2
N = 16 # number of random samples we will generate
uniform = np.random.uniform # just make this call a bit cleaner
def random_sampling(N, get_grid=False):
    return uniform(0, 1, (N, 2))
    
plotter(random_sampling(N=N), title='random sampling')
def regular_sampling(N, get_grid=False):
        samples = np.zeros((N, 2))

        # split N into two factors in such a way N = nx * ny
        ny, nx = two_factors_to_n(N)

        # get all the combinations of nx and ny
        j, i = np.meshgrid(range(nx), range(ny))

        # get grid values
        grid_i = i[:, 0] / nx
        grid_j = j[0] / ny

        j, i = j.flatten(), i.flatten()

        # get samples
        samples[:, 0] = (i + 0.5) / nx
        samples[:, 1] = (j + 0.5) / ny

        if get_grid:
            return samples, [grid_i, grid_j]
        return samples

plotter(*regular_sampling(N=N, get_grid=True), title='regular sampling')
def jittered_sampling(N, get_grid=False):
        samples = np.zeros((N, 2))

        # split N into two factors in such a way N = nx * ny
        ny, nx = two_factors_to_n(N)

        # get all the combinations of nx and ny
        j, i = np.meshgrid(range(ny), range(nx))

        # get grid values
        grid_i = i[:, 0] / nx
        grid_j = j[0] / ny

        # flatten idxs to get samples
        j, i = j.flatten(), i.flatten()

        # get samples
        samples[:, 0] = uniform(i / nx, (i + 1) / nx)
        samples[:, 1] = uniform(j / ny, (j + 1) / ny)

        if get_grid:
            return samples, [grid_i, grid_j]
        return samples

plotter(*jittered_sampling(N, get_grid=True), title='jittered sampling')
def half_jittered_sampling(N, get_grid=False):
    samples = np.zeros((N, 2))

    # split N into two factors in such a way N = nx * ny
    ny, nx = two_factors_to_n(N)

    # get all the combinations of nx and ny
    j, i = np.meshgrid(range(ny), range(nx))

    # get grid values
    grid_i = i[:, 0] / nx
    grid_j = j[0] / ny

    # flatten idxs to get samples
    j, i = j.flatten(), i.flatten()

    # get samples
    samples[:, 0] = uniform((i + 0.25) / nx, (i + 0.75) / nx)
    samples[:, 1] = uniform((j + 0.25) / ny, (j + 0.75) / ny)

    if get_grid:
        return samples, [grid_i, grid_j]
    return samples
    
plotter(*half_jittered_sampling(N, get_grid=True), title='half-jittered sampling')
def poisson_disk_sampling(N, d, get_grid=False):
    def get_mask(samples):
        j, i = np.meshgrid(range(N), range(N))
        j, i = j.flatten(), i.flatten()

        dist = ((samples[i, 0] - samples[j, 0]) ** 2 + (samples[i, 1] - samples[j, 1]) ** 2).reshape(
            (N, N))

        upper_diag_idx = np.triu_indices(N)
        dist[upper_diag_idx] = float('inf')
        mask = np.any(dist < d ** 2, axis=1)
        return mask

    samples = uniform(0, 1, (N, 2))
    mask = get_mask(samples)

    # run the sampling on the elements that are closer than 'd' until
    # the distances between samples are higher than 'd'
    while np.any(mask):
        samples[mask] = uniform(0, 1, (N, 2))[mask]
        mask = get_mask(samples)

    return samples
    
plotter(poisson_disk_sampling(N, 0.2), title='poisson sampling')
def nrooks_sampling(N, get_grid=False):
    samples = np.zeros((N, 2))
    i = np.arange(0, N)
    samples_x = uniform(i / N, (i + 1) / N)
    samples_y = uniform(i / N, (i + 1) / N)

    # get grid values
    grid_i = grid_j = i / N

    # randomly shuffle samples over X 
    np.random.shuffle(samples_x)
    # np.random.shuffle(samples_y)

    samples[:, 0] = samples_x
    samples[:, 1] = samples_y

    if get_grid:
        return samples, [grid_i, grid_j]
    return samples
    
plotter(*nrooks_sampling(N, get_grid=True), title='n-rooks sampling')
# todo
# todo