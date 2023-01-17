%matplotlib inline
COLOUR_FIGURE = False


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
features = data['data']
feature_names = data['feature_names']
target = data['target']
species = data['target_names'][data['target']]

pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
for i,(p0,p1) in enumerate(pairs):
    plt.subplot(2,3,i+1)
    for t,marker,c in zip(range(3),">ox","rgb"):
        plt.scatter(features[target == t,p0], features[target == t,p1], marker=marker, c=c)
    plt.xlabel(feature_names[p0])
    plt.ylabel(feature_names[p1])
    plt.xticks([])
    plt.yticks([])
plt.show()

# a closer look at the first subplot above
for t,marker,c in zip(np.arange(3),">ox","rgb"):
    plt.scatter(features[target == t,0], 
                features[target == t,1],
                marker=marker,
                c=c)
    plt.title('Iris Dataset')
    plt.xlabel('septal length (cm)')
    plt.ylabel('septal width (cm)')
labels = data['target_names'][data['target']]

plength = features[:,2]
is_setosa = (labels == 'setosa')
print('Maximum of setosa: {0}.'.format(plength[is_setosa].max()))
print('Minimum of others: {0}.'.format(plength[~is_setosa].min()))

# looking at the graph above, the differences b/w setosa v. others below - we can make a 
# simple model. If the petal length is smaller than two, this is an Iris Setosa;
# otherwise, it is either an Iris Virginica or Iris Versicolor.

if features[:,2].all() < 2: print('Iris Setosa')
else: print('Iris Virginica or Iris Versicolour')

import matplotlib.patches as mpatches

# Since we have a simple model to differentiate sesota's, we should now determine how 
# to differentiate the other types of iris's. The graph below depicts a possible decison boundary

setosa = (species == 'setosa')
features = features[~setosa]
species = species[~setosa]
virginica = species == 'virginica'

t = 1.75
p0,p1 = 3,2

if COLOUR_FIGURE:
    area1c = (1.,.8,.8)
    area2c = (.8,.8,1.)
else:
    area1c = (1.,1,1)
    area2c = (.7,.7,.7)

x0,x1 =[features[:,p0].min()*.9,features[:,p0].max()*1.1]
y0,y1 =[features[:,p1].min()*.9,features[:,p1].max()*1.1]

plt.fill_between([t,x1],[y0,y0],[y1,y1],color=area2c)
plt.fill_between([x0,t],[y0,y0],[y1,y1],color=area1c)
plt.plot([t,t],[y0,y1],'k--',lw=2)
plt.plot([t-.1,t-.1],[y0,y1],'k:',lw=2)
plt.scatter(features[virginica,p0], features[virginica,p1], c='b', marker='o')
plt.scatter(features[~virginica,p0], features[~virginica,p1], c='r', marker='x')
plt.ylim(y0,y1)
plt.xlim(x0,x1)
plt.xlabel(feature_names[p0])
plt.ylabel(feature_names[p1])
#plt.legend(loc="lower left")
#plt.legend([('o'), ('x')], ["Attr A", "Attr A+B"])
virginica_ir = mpatches.Patch(color='white', label='Virginica')
versicolor_ir = mpatches.Patch(color='gray', label='Versicolor')
plt.legend(handles=[virginica_ir,versicolor_ir])
plt.show()
# below we id all possible thresholds for this feature
# the inner for loop tests all thresholds
best_acc = -1.0
for fi in range(features.shape[1]):
  thresh = features[:,fi].copy()
  thresh.sort()
  for t in thresh:
    pred = (features[:,fi] > t)
    acc = (pred == virginica).mean()
    if acc > best_acc:
      best_acc = acc
      best_fi = fi
      best_t = t
# The next few lines are somewhat redunant, but I was having difficulty using the .csv data to make
# these images. Any advice on how to avoid this repetitive step, and display different kernel 
# classifications using the .csv? Thanks for your time, any advice would be greatly appreciated!
from sklearn import datasets, svm
iris = datasets.load_iris()
X = iris.data
y = iris.target

Septal_length = X[:,0]
Septal_width = X[:,1]
Petal_length = X[:,2]
Petal_width = X[:,3]


X = X[y != 0, :2]
y = y[y != 0]

n_sample = len(X)

np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
y = y[order].astype(np.float)

X_train = X[:.9 * n_sample]
y_train = y[:.9 * n_sample]
X_test = X[.9 * n_sample:]
y_test = y[.9 * n_sample:]

for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)

    plt.figure(fig_num)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.title(kernel)
plt.show()

# Below is a tutorial that is available in the link below. It looks at the decision 
# surfaces of forests of randomized trees trained on pairs of features of the dataset.
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html
# All of this interesting vizualization was generated using the scikit tutorial listed below:
# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html
from sklearn import clone
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.externals.six.moves import xrange
from sklearn.tree import DecisionTreeClassifier

# Parameters - only 3 types of iris flowers in the dataset
n_classes = 3
n_estimators = 30
plot_colors = "ryb"
cmap = plt.cm.RdYlBu
plot_step = 0.02  # fine step width for decision surface contours - highlights differences b/w
# decision boundaries
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
RANDOM_SEED = 13  # fix the seed on each iteration

plot_idx = 1

# Below is a list containing the different types of classifiers that we will be looking at;
# the for loops and conditional staements below will illustrate the how the decision boundaries
# of these types of classifiers differs when examining different features of the dataset. 
models = [DecisionTreeClassifier(max_depth=None),
          RandomForestClassifier(n_estimators=n_estimators),
          ExtraTreesClassifier(n_estimators=n_estimators),
          AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                             n_estimators=n_estimators)]

for pair in ([0, 1], [0, 2], [2, 3]):
    for model in models:
        # Only lookin at two corresponding features below
        X = iris.data[:, pair]
        y = iris.target

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        # Train
        clf = clone(model)
        clf = model.fit(X, y)

        scores = clf.score(X, y)
        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(".")[-1][:-2][:-len("Classifier")]
        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(len(model.estimators_))
        print( model_details + " with features", pair, "has a score of", scores )

        plt.subplot(3, 4, plot_idx)
        if plot_idx <= len(models):
            # Add a title at the top of each column
            plt.title(model_title)

        # Now plot the decision boundary using a fine mesh as input to a
        # filled contour plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        # Plot either a single DecisionTreeClassifier or alpha blend the
        # decision surfaces of the ensemble of classifiers
        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            # Choose alpha blend level with respect to the number of estimators
            # that are in use (noting that AdaBoost can use fewer estimators
            # than its maximum if it achieves a good enough fit early on)
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a black outline
        xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser),
                                             np.arange(y_min, y_max, plot_step_coarser))
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15, c=Z_points_coarser, cmap=cmap, edgecolors="none")

        # Plot the training points, these are clustered together and have a
        # black outline
        for i, c in zip(xrange(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=c, label=iris.target_names[i],
                        cmap=cmap)

        plot_idx += 1  # move on to the next plot in sequence

plt.suptitle("Classifiers on feature subsets of the Iris dataset")
plt.axis("tight")

plt.show()
# Again, ran into some trouble here so the next few lines are redundant b/c I had to 
# import the iris dataset again. The next few boxes produce some histograms of the iris
# dataset features. 

from scipy.stats import norm 
iris = datasets.load_iris()
X = iris.data
y = iris.target

Septal_length = X[:,0]
Septal_width = X[:,1]
Petal_length = X[:,2]
Petal_width = X[:,3]

S_L= np.array(Septal_length)
print('mean septal length', S_L.mean())
S_L_mean= S_L.mean()
print("max septal length", S_L.max())
print("min septal length", S_L.min())
print("standard dev", S_L.std())
S_L_std =S_L.std()
S_L_mean, S_L_std = norm.fit(S_L)

plt.figure()
h= plt.hist(S_L)
plt.legend(['Septal length cm'])
plt.xlabel('Septal length cm')
plt.ylabel('occurances')
plt.grid()
S_W= np.array(Septal_width)
print('mean septal width', S_W.mean())
S_W_mean= S_W.mean()
print("max septal width", S_W.max())
print("min septal width", S_W.min())
print("standard dev", S_W.std())
S_W_std =S_W.std()
S_W_mean, S_W_std = norm.fit(S_W)

plt.figure()
h= plt.hist(S_W)
plt.legend(['Septal width cm'])
plt.xlabel('Septal width cm')
plt.ylabel('occurances')
plt.grid()
P_L= np.array(Petal_length)
print('mean petal length', P_L.mean())
P_L_mean= P_L.mean()
print("max petal length", P_L.max())
print("min petal length", P_L.min())
print("standard dev", P_L.std())
P_L_std =P_L.std()
P_L_mean, P_L_std = norm.fit(P_L)

plt.figure()
h= plt.hist(P_L)
plt.legend(['Petal length cm'])
plt.xlabel('Petal length cm')
plt.ylabel('occurances')
plt.grid()
P_W= np.array(Petal_width)
print('mean Petal width', P_W.mean())
P_W_mean= P_W.mean()
print("max Petal width", P_W.max())
print("min Petal width", P_W.min())
print("standard dev", P_W.std())
P_W_std =P_W.std()
P_W_mean, P_W_std = norm.fit(P_W)

plt.figure()
h= plt.hist(P_W)
plt.legend(['Petal width cm'])
plt.xlabel('Petal width cm')
plt.ylabel('occurances')
plt.grid()
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.boxplot(data=iris.data, orient="h");
plt.xlabel("Cm")
plt.ylabel("data features")
plt.xticks(), feature_names
plt.legend(feature_names,loc="lower right")
plt.title('Iris categorical plot')
