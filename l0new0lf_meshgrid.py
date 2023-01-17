import numpy as np

import matplotlib.pyplot as plt
# Refactored Code

# ===================================

def get_mesh_grid_for_plt(plt, n_pts, subplot=False):

    """

    Returns 2-D grid of x, y points corresponding

    to meshgrid (scale of gdrid must be predefiend)

    

    :param plt: matplotlib pyplot

    :param n_pts: tuple of (n_xpts, n_ypts) reprsenting 2-D grid

    

    TIME: O(N^2)

    """

    # get lims (need to be set by user)

    if not subplot:

        xl, xr = plt.xlim()

        yb, yt = plt.ylim()

    else: # deal w/ axesplots

        xl, xr, yb, yt = plt.axis()

    

    # gen data using lims

    # -------------------------

    num_xpts, num_ypts = n_pts

    # -------------------------

    grid_xs = []

    grid_ys = []

    for x in np.linspace(xl, xr, num_xpts):

        # for every x, gen ys

        for y in np.linspace(yb, yt, num_ypts):

            grid_xs.append(x)

            grid_ys.append(y)    

            

    return (np.array(grid_xs), np.array(grid_ys))
fig = plt.figure(figsize=(10,6))



n_xpts, n_ypts = 30, 30

grid_xs, grid_ys = get_mesh_grid_for_plt(plt, (n_xpts, n_ypts))



# plot data

plt.scatter(grid_xs, grid_ys)

plt.show()
fig = plt.figure(figsize=(10,6))



n_xpts, n_ypts = 100, 100

grid_xs, grid_ys = get_mesh_grid_for_plt(plt, (n_xpts, n_ypts))



# plot data

plt.scatter(grid_xs, grid_ys)

plt.show()
def get_random_wts(size):

    """ uniform [-1, 1]"""

    return np.random.uniform(-1, 1, size=size)
fig = plt.figure(figsize=(10,6))



# predefine scale limits

plt.xlim(left=-1, right=1)

plt.ylim(bottom=-1, top=1)



# setup meshgrid: get coordinates of 2D grid

n_xpts, n_ypts = 300, 300

grid_xs, grid_ys = get_mesh_grid_for_plt(plt, (n_xpts, n_ypts))



# ========================================================

# computed color using direction of W•X 

# (no intercept i.e origin-centered)

w0, w1 = get_random_wts(2)

WX = grid_xs*w0 + grid_ys*w1 # W•X

colors = np.where(WX>=0, 'purple', 'grey')

# =========================================================



# plot data ---------------------------

plt.scatter(grid_xs, grid_ys, c=colors)

# -------------------------------------

# sanity check check equation

xs = np.linspace(-1, 1, 10)

ys = -(w0/w1) * xs

plt.plot(xs, ys, c='yellow')



plt.show()
fig, axarr = plt.subplots(4, 4, figsize=(20, 10))

axarr = axarr.reshape(-1)



for ax in axarr:



    # predefine scale limits

    ax.axis(xmin=-1, xmax=1, ymin=-1, ymax=1)    



    # setup meshgrid: get coordinates of 2D grid

    n_xpts, n_ypts = 30, 30

    grid_xs, grid_ys = get_mesh_grid_for_plt(ax, (n_xpts, n_ypts), subplot=True)



    # ========================================================

    # computed color using direction of W•X 

    # (no intercept i.e origin-centered)

    w0, w1 = get_random_wts(2)

    WX = grid_xs*w0 + grid_ys*w1 # W•X

    colors = np.where(WX>=0, 'purple', 'grey')

    # =========================================================



    # plot data ---------------------------

    ax.scatter(grid_xs, grid_ys, c=colors)

    # -------------------------------------

    # sanity check check equation

    xs = np.linspace(-1, 1, 10)

    ys = -(w0/w1) * xs

    ax.plot(xs, ys, c='yellow')



plt.show()
NUM_RANDOM_LINES = 100



fig = plt.figure(figsize=(10,6))



# predefine scale limits

plt.xlim(left=-1, right=1)

plt.ylim(bottom=-1, top=1)



xs = np.linspace(-1, 1, 10)

for _ in range(NUM_RANDOM_LINES):

    w0, w1 = get_random_wts(2)

    ys = -(w0/w1) * xs

    plt.plot(xs, ys, c='yellow')



plt.show()