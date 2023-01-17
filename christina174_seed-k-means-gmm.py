import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from yellowbrick.cluster import KElbowVisualizer

from scipy.stats import mode
data = pd.read_csv('../input/seed-from-uci/Seed_Data.csv')
y = data['target']
data = data.drop(columns = 'target')

data.rename(columns= {'A':'area', 'P':'perimeter', 'C':'compactness', 'LK':'length of kernel', 'WK':'width of kernel', 'LKG':'length of kernel groove'}, inplace=True)

data.head()
sns.heatmap(data.corr(), square=True, annot=True, cbar=False, cmap='BuGn');
data_drop = data.drop(columns=['perimeter','width of kernel'])

data_drop
kmeans = KMeans(random_state=42)
visualizer = KElbowVisualizer(kmeans, k=(7))

visualizer.fit(data_drop)        # Fit the data to the visualizer

visualizer.show();        # Finalize and render the figure
kmeans = KMeans(n_clusters=3, random_state=1)

clusters = kmeans.fit_predict(data_drop)
def make_labels(y, clusters):

    labels = np.zeros_like(clusters)

    for i in range(3):

        mask = (clusters == i)

        labels[mask] = mode(y[mask])[0]

    return labels
labels = make_labels(y, clusters)
accuracy_score(y, labels)
matrix = confusion_matrix(y, labels)

sns.heatmap(matrix.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label');
from sklearn.mixture import GaussianMixture as GMM

model_gmm = GMM(n_components=3, random_state=42)#.fit(data_drop)

model_gmm.fit(data_drop)

labels = model_gmm.predict(data_drop)

plt.scatter(data['area'], data['A_Coef'], c=y, s=40, cmap='viridis'); # true
labels = make_labels(y, labels)
from matplotlib.patches import Ellipse



def draw_ellipse(position, covariance, ax=None, **kwargs):

    ax = ax or plt.gca()

    

    if covariance.shape == (2, 2): #Convert covariance to principal axes

        U, s, Vt = np.linalg.svd(covariance)

        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))

        width, height = 2 * np.sqrt(s)

    

    for nsig in range(1, 4): # Draw the Ellipse

        ax.add_patch(Ellipse(position, nsig * width, nsig * height,

                             angle, **kwargs)) 

    

def plot_gmm(gmm, X, labels, label=True, ax=None):

    ax = ax or plt.gca()

#     labels = gmm.fit(X).predict(X)

    if label:

        ax.scatter(X['area'], X['A_Coef'], c=labels, s=40, cmap='viridis', zorder=2)

    else:

        ax.scatter(X['area'], X['A_Coef'], s=40, zorder=2)

    ax.axis('equal')

    

    w_factor = 0.2 / gmm.weights_.max()

    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):

        

        covar = covar[np.ix_([0,3],[0,3])]

        pos = pos[np.ix_([0,3])]

        draw_ellipse(pos, covar, alpha=w * w_factor)



plot_gmm(model_gmm, data_drop, labels)
accuracy_score(y, labels)
matrix = confusion_matrix(y, labels)

sns.heatmap(matrix.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label');
pd.options.display.max_rows = (len(y))

df = pd.DataFrame ({'Actual': y, 'Predicted': labels})

df