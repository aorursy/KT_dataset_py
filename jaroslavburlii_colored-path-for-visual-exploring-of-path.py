import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
import matplotlib.collections as mcoll
import matplotlib.path as mpath
cities  = pd.read_csv('../input/cities.csv')
cities.head()
# Greedy algorithm without prime numbers
def greedy_whp(verbose=True, k_iter=10000):
    ID = cities.CityId.values
    coord = cities[['X', 'Y']].values
    pos = coord[0]
    path = [0]
    
    ID = np.delete(ID, 0)
    coord = np.delete(coord, 0, axis=0)
    
    it = 0
    
    while len(path) != cities.shape[0]:
        # Compute the distance matrix
        dist_matrix = np.linalg.norm(coord - pos, axis=1)
        
        # Find the nearest city
        i_min = dist_matrix.argmin()
        
        path.append(ID[i_min])
        pos = coord[i_min]
        
        # Delete it
        coord = np.delete(coord, i_min, axis=0)
        ID = np.delete(ID, i_min)
        
        it += 1
        
        if verbose and it%k_iter == 0:
            print('{} iterations, {} remaining cities.'.format(it, len(ID)))
    
    # Don't forget to add the north pole at the end!
    path.append(0)
    
    return path
%time path_greedy_whp = greedy_whp(verbose=True)
# Draw colored path with diagram
# Base on: 
# https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib/25941474#25941474
# https://matplotlib.org/1.2.1/examples/pylab_examples/multicolored_line.html
def drawColoredPath(x, y, z=None, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    # draw path
    fig, ax = plt.subplots(nrows=1, figsize=(20,15))
    ax.add_collection(lc)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    north_pole = cities[cities.CityId==0]
    plt.scatter(north_pole.X, north_pole.Y, marker='*', c='red', s=1000)    
    
    # draw diagram color - index of step/city
    fig, ax = plt.subplots(nrows=1, figsize=(15,0.5))
    norm = mpl.colors.Normalize(vmin=0, vmax=len(x))
    cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    ax.set_xlabel('Colors and steps mapping')
    ax.xaxis.set_label_position('top')
    
    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

# Draw colored path, details:
# https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib/25941474#25941474
# https://matplotlib.org/1.2.1/examples/pylab_examples/multicolored_line.html
def plot_colored_path(path):
    coords = cities[['X', 'Y']].values
    ordered_coords = coords[np.array(path)]
    codes = [Path.MOVETO] * len(ordered_coords)
    path = Path(ordered_coords, codes)

    x, y = ordered_coords[:, 0], ordered_coords[:, 1]
    z = np.linspace(0, 1, len(x))
    drawColoredPath(x, y, z, cmap=plt.get_cmap('jet'), 
                    linewidth=1)
    
show_path = path_greedy_whp[0:100] # print only first 100 steps
plot_colored_path(show_path)
show_path = path_greedy_whp
plot_colored_path(show_path) 
