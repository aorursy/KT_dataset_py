import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.datasets import make_blobs

import numpy as np

import matplotlib as mpl

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors





def plot_colortable(colors, title, sort_colors=True, emptycols=0, filter=False):



    cell_width = 212

    cell_height = 22

    swatch_width = 48

    margin = 12

    topmargin = 40

    if filter:

        cols_old = colors.items()

        colors = {}

        for name, color in cols_old:

            color_ = mcolors.to_rgb(color)

            if min(color_) < 0.35 and 'grey' not in name:

                colors[name] = color

    print(len(colors.keys()))

    # Sort colors by hue, saturation, value and name.

    if sort_colors is True:

        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),

                         name)

                        for name, color in colors.items())

        names = [name for hsv, name in by_hsv]

    else:

        names = list(colors)



    n = len(names)

    ncols = 4 - emptycols

    nrows = n // ncols + int(n % ncols > 0)



    width = cell_width * 4 + 2 * margin

    height = cell_height * nrows + margin + topmargin

    dpi = 72



    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    fig.subplots_adjust(margin/width, margin/height,

                        (width-margin)/width, (height-topmargin)/height)

    ax.set_xlim(0, cell_width * 4)

    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)

    ax.yaxis.set_visible(False)

    ax.xaxis.set_visible(False)

    ax.set_axis_off()

    ax.set_title(title, fontsize=24, loc="left", pad=10)



    for i, name in enumerate(names):

        row = i % nrows

        col = i // nrows

        y = row * cell_height



        swatch_start_x = cell_width * col

        swatch_end_x = cell_width * col + swatch_width

        text_pos_x = cell_width * col + swatch_width + 7



        ax.text(text_pos_x, y, name, fontsize=14,

                horizontalalignment='left',

                verticalalignment='center')



        ax.hlines(y, swatch_start_x, swatch_end_x,

                  color=colors[name], linewidth=18)



    return fig



#plot_colortable(mcolors.BASE_COLORS, "Base Colors",

#                sort_colors=False, emptycols=1)

#plot_colortable(mcolors.TABLEAU_COLORS, "Tableau Palette",

#                sort_colors=False, emptycols=2)



#sphinx_gallery_thumbnail_number = 3

plot_colortable(mcolors.CSS4_COLORS, "CSS Colors")



# Optionally plot the XKCD colors (Caution: will produce large figure)

#xkcd_fig = plot_colortable(mcolors.XKCD_COLORS, "XKCD Colors")

#xkcd_fig.savefig("XKCD_Colors.png")

plt.show()
colors = {}

c_list = []

tmp = []

for name, color in mcolors.CSS4_COLORS.items():

    color_ = mcolors.to_rgb(color)

    if min(color_) < 0.35 and 'grey' not in name:

        tmp.append([color_, name])

        c_list.append(color_)

tmp.sort(key=lambda x: x[0].index(max(x[0])))

c_list.sort(key=lambda x: x.index(max(x)))

for i in range(len(tmp)):

    colors[tmp[i][1]] = tmp[i][0]

del colors['black']
l1 = ['wheat', 'lightgray', 'mistyrose', 'peachpuff', 'ivory', 'burlywood']

for light_color in l1:

    color = mcolors.to_rgb(mcolors.CSS4_COLORS[light_color])

    print(color, sum(color), min(color))
def generate_random_colors(n_colors):

    import random

    section_size = len(c_list) / n_colors

    if section_size >= 1.0:

        block_size = int(section_size)

        ret_colors = {}

        for i in range(n_colors):

            start = i*block_size

            end = (i+1)*block_size - 1

            ri = random.randint(start, end)

            name = list(colors.keys())[ri]

            color = list(colors.values())[ri]

            ret_colors[name] = color

            

        return ret_colors

    else:

        pass
c_d1 = generate_random_colors(66)

plot_colortable(c_d1, "CSS Colors", filter=False)



# Optionally plot the XKCD colors (Caution: will produce large figure)

#xkcd_fig = plot_colortable(mcolors.XKCD_COLORS, "XKCD Colors")

#xkcd_fig.savefig("XKCD_Colors.png")

plt.show()
from collections import Iterable

def bright(cmaps, alpha=0.05):

    cmps_bright = None

    if isinstance(alpha, Iterable):

        cmps_bright = list(cmaps)

        for i in range(len(cmps_bright)):

            cmps_bright[i] += (alpha[i],)

    else:

        cmps_bright = list(cmaps)

        for i in range(len(cmps_bright)):

            cmps_bright[i] += (alpha,)

    return cmps_bright

    
def visualize2D(X_set, y_set, classifier, alpha=0.5, accuracy=0.1, n_classes=None, cmps="auto", \

              title="SVM Model Visualization", x_label="Feature1", y_label="Feature2", **fig_kwargs):

    f, ax = plt.subplots(1, 1, **fig_kwargs)

    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = accuracy),

                        np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = accuracy))

    pred_mask_graph = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)

    from matplotlib.colors import ListedColormap

    if cmps == "auto":

        cmps = list(generate_random_colors(n_classes).values())

        cmps_bg = bright(list(cmps), alpha)

    else:

        cmps_bg = bright(list(cmps), alpha)

    tot_classes = np.unique(pred_mask_graph)

    y_uniques = np.unique(y_set)

    if len(y_uniques) > len(tot_classes):

        y_set[np.isin(y_set, tot_classes, invert=True)] = -1

        cmps.insert(0, mcolors.to_rgb(mcolors.CSS4_COLORS['black']))

        

    

    ax.contourf(X1, X2, pred_mask_graph.reshape(X1.shape), #alpha=0.1,

                 cmap=ListedColormap(cmps_bg))

    ax.set_xlim(X1.min(), X1.max())

    ax.set_ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):

        ax.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                    c = ListedColormap(cmps)(i), label = j)

    ax.set_title(title)

    ax.set_ylabel(y_label)

    ax.set_xlabel(x_label)

    ax.xaxis.set_label_coords(1.05, -0.025)

    ax.legend()

    plt.show()

#colors = [(1, 0, 0, 0.2), (0, 1, 0, 0.3), (0, 0, 1, 0.2), (1, 1, 0, 0.2)]

X, y = make_blobs(n_samples = 50, n_features = 2, centers = [[0, 0], [10, 10], [10, 0], [0, 10]], cluster_std = 2, random_state = 43)

classifier = SVC(gamma='auto')

classifier.fit(X, y)

#y[-1] = 22

#y[-2] = 1023

visualize2D(X, y, classifier, alpha=0.5, figsize=(12, 12), n_classes=4, accuracy=0.05)


def visualize3D(X_set, y_set, classifier, alpha, n_classes, cmps='auto', title='SVM Model Visualization', \

                x_label="Feature 1", y_label="Feature 2", z_label="Feature 3", accuracy=0.2, **fig_kwargs):

    from matplotlib.colors import ListedColormap

    from mpl_toolkits.mplot3d import Axes3D

    if cmps == "auto":

        cmps = list(generate_random_colors(n_classes).values())

        cmps_bg = bright(list(cmps), alpha)

    else:

        cmps_bg = bright(list(cmps), alpha)

    fig = plt.figure(**fig_kwargs)

    ax = fig.add_subplot(111, projection='3d')

    grid = np.array(np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = accuracy),

                    np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = accuracy), 

                    np.arange(start = X_set[:, 2].min() - 1, stop = X_set[:, 2].max() + 1, step = accuracy))).T.reshape(-1,3)

    print("New grid is created.")

    preds = classifier.predict(grid)

    print("classification of each point is done.")

    ax.set_xlim(X_set[:, 0].min(), X_set[:, 0].max())

    ax.set_ylim(X_set[:, 1].min(), X_set[:, 1].max())

    ax.set_zlim(X_set[:, 2].min(), X_set[:, 2].max())

    print(len(grid))

    for i, j in enumerate(np.unique(y_set)):

        print(i)

        ax.scatter(grid[preds == j, 0], grid[preds == j, 1], grid[preds == j, 2], c = ListedColormap(cmps_bg)(i))

        ax.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], X_set[y_set == j, 2],

                    c = ListedColormap(cmps)(i), label = j)

    ax.set_title(title)

    ax.set_ylabel(y_label)

    ax.set_xlabel(x_label)

    ax.set_zlabel(z_label)

    ax.xaxis.set_label_coords(1.05, -0.025)

    ax.legend()

    ax.legend(loc='upper right', bbox_to_anchor=(1, -0.1)) 

    plt.show()

colors_ = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 185/255, 0)]

X, y = make_blobs(n_samples = 50, n_features = 3, centers = [[5, 0, 0], [5, 5, 5], [0, -5, 5], [0, -5, 0]], cluster_std = 1, random_state = 43)

classifier = SVC(gamma='auto')

classifier.fit(X, y)

visualize3D(X, y, classifier, alpha=[0.01, 0.01, 0.005, 0.01], figsize=(6, 6), n_classes=4, cmps=colors_)
class Visualizer:

    

    def __init__(self, n_features, model):

        if n_features in [2, 3]:

            self.n_features = n_features

        else:

            raise("Invalid input", ValueError)

        self.model = model

    

    def visualize(self, X_set, y_set, alpha, n_classes, cmps='auto', title='SVM Model Visualization', \

                x_label="Feature 1", y_label="Feature 2", z_label="Feature 3", accuracy=0.1, **fig_kwargs):

        if self.n_features == 2:

            visualize2D(X_set=X_set, y_set=y_set, classifier=self.model, alpha=alpha, n_classes=n_classes, cmps=cmps, \

                     title=title, x_label=x_label, y_label=y_label, accuracy=accuracy, **fig_kwargs)

        else:

            visualize3D(X_set=X_set, y_set=y_set, classifier=self.model, alpha=alpha, n_classes=n_classes, cmps=cmps, \

                        title=title, x_label=x_label, y_label=y_label, z_label=z_label, accuracy=accuracy, **fig_kwargs)

    
X, y = make_blobs(n_samples = 50, n_features = 2, centers = [[0, 0], [10, 10], [10, 0], [0, 10]], cluster_std = 2, random_state = 43)

classifier = SVC(gamma='auto')

classifier.fit(X, y)

#y[-1] = 22

#y[-2] = 1023

vis_cls = Visualizer(2, classifier)

vis_cls.visualize(X, y, alpha=0.5, figsize=(12, 12), n_classes=4, accuracy=0.05)
import time
colors_ = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 185/255, 0)]

X, y = make_blobs(n_samples = 50, n_features = 3, centers = [[6, 0, 0], [5, 5, 6], [0, -5, 5], [0, -5, 0]], cluster_std = 1, random_state = 43)

classifier = SVC(gamma='auto')

classifier.fit(X, y)

v_c = Visualizer(3, classifier)

tic = time.time()

v_c.visualize(X, y, alpha=[0.005, 0.005, 0.003, 0.005], figsize=(6, 6), n_classes=4, cmps=colors_, accuracy=0.15)

tac = time.time()

print(tac - tic)
colors_ = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

X, y = make_blobs(n_samples = 50, n_features = 3, centers = [[5, 0, 0], [5, 5, 6], [0, -5, 5]], cluster_std = 1, random_state = 43)

classifier = SVC(gamma='auto')

classifier.fit(X, y)

v_c = Visualizer(3, classifier)

tic = time.time()

v_c.visualize(X, y, alpha=[0.005, 0.005, 0.005], figsize=(6, 6), n_classes=4, cmps=colors_, accuracy=0.2)

tac = time.time()

print(tac - tic)
colors_ = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.90, 180/255, 0)]

X, y = make_blobs(n_samples = 50, n_features = 3, centers = [[5, 0, 0], [5, 5, 6], [0, -5, 4], [-1, -5, 7]], cluster_std = 1, random_state = 43)

classifier = SVC(gamma='auto')

classifier.fit(X, y)

v_c = Visualizer(3, classifier)

tic = time.time()

v_c.visualize(X, y, alpha=[0.005, 0.005, 0.004, 0.005], figsize=(6, 6), n_classes=4, cmps=colors_, accuracy=0.2)

tac = time.time()

print(tac - tic)
def visualizerrgr(X, y, classifier, step=0.1, xlabel="Features", ylabel="Predictions"):

    _x = np.arange(np.min(X), np.max(X), step)

    _y = classifier.predict(_x.reshape(-1, 1))

    plt.scatter(X, y, label="Data Points")

    plt.plot(_x, _y, c='r', label="Regression line")

    plt.legend()

    plt.title("Regression model visualizer")

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.grid()

    plt.show()

from sklearn.svm import SVR

_x = np.arange(1, 10)

err = np.random.random((len(_x,)))

y = _x**2 + 5*err**(1 + err)

_x = _x.reshape(-1, 1)

classifier = SVR(kernel="poly", degree=2, gamma="auto")

classifier.fit(_x, y)

visualizerrgr(_x, y, classifier)
def visualizerrgr2D(X, y, classifier, step=0.1, xlabel="Feature1", ylabel="Feature2", zlabel="Predictions"):

    X1, X2 = np.meshgrid(np.arange(start = X[:, 0].min() - 1, stop = X[:, 0].max() + 1, step = step),

                        np.arange(start = X[:, 1].min() - 1, stop = X[:, 1].max() + 1, step = step))

    _z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X1.ravel(), X2.ravel(), _z, c=(0, 1, 0, 0.05))

    ax.scatter(X[:, 0], X[:, 1], y, c='r', label="Data points")

    ax.legend(loc='upper right', bbox_to_anchor=(1, -0.1))

    ax.set_ylabel(ylabel)

    ax.set_xlabel(xlabel)

    ax.set_zlabel(zlabel)

    ax.set_title("Regression model visualizer")

    plt.grid()

    plt.show()

from sklearn.svm import SVR

_x = np.arange(1, 10, 0.2) - 5

_y = np.arange(1, 10, 0.2) - 5

np.random.shuffle(_x)

np.random.shuffle(_y)

err = np.random.random((len(_x,)))

z = _x**2 + _y**2 + 5*err**(1 + err)**3 + 1

z = -z

_x = _x.reshape(-1, 1)

_y = _y.reshape(-1, 1)

X = np.concatenate([_x, _y], axis=1)

classifier = SVR(kernel="poly", degree=2, gamma="auto")

classifier.fit(X, z)

visualizerrgr2D(X, z, classifier)