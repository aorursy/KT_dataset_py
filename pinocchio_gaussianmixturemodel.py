import numpy as np
import pandas as pd
from sklearn import mixture
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('../input/train.csv',
                        usecols=['x','y','time','place_id','accuracy'], 
                        index_col = 'place_id')
place_id = pd.DataFrame(train.index.value_counts())
place_id.columns = ['count']
place_id.index.name = 'place_id'
place_id = place_id.iloc[:40]
train = train[train.index.isin(place_id.index)]
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def fit_gmm(X):
    score = np.zeros(X.shape[0])
    for i in range(10,0,-1):
        gmm = mixture.GMM(n_components=i, covariance_type='tied', min_covar=0.0001)
        gmm.fit(X[score>-7].values)
        if (gmm.weights_<0.003).sum() == 0:
            #print(i)
            break
        score = gmm.score(X)
    return gmm

g = train[['x','y']].groupby(level=0)
place_id['gmm'] = g.apply(fit_gmm)
# adapted from http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html
from scipy import linalg
import matplotlib as mpl
import itertools

plt.rcParams['figure.figsize'] = (16,32)

color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y','gray','brown'])

#splot = plt.subplot(4, 1, 1)

p = [1,6, 10, 16, 21, 26, 31, 33, 35]

for j in range(len(p)):
    clf = place_id.gmm.iloc[p[j]]
    X = train.loc[place_id.index[p[j]], ['x','y']].values
    Y_ = clf.predict(X)
    splot = plt.subplot(9, 1, 1 +j)
        
    for i, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        #if not np.any(Y_ == i):
        #    continue
        splot.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .5, color=color, alpha=0.5)
        
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        #print(j, color)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(0,10)
    plt.title(str(p[j])+', '+str(clf.n_components))