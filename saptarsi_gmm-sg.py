# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
colors = ['royalblue','red','deeppink', 'maroon', 'mediumorchid', 'tan', 'forestgreen', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=200, random_state=8,cluster_std=[1.0, 2.5, 0.5])
plt.scatter(X[:,0], X[:,1],c=vectorizer(y))
from sklearn import mixture
gmm = mixture.GaussianMixture(n_components=3, covariance_type='tied',max_iter=100,init_params='random')
gmm.fit(X)
yclust=gmm.predict(X)
plt.scatter(X[:,0], X[:,1],c=vectorizer(yclust))
from sklearn import mixture
gmm = mixture.GaussianMixture(n_components=3, covariance_type='tied',max_iter=100,init_params='random')
gmm.fit(X)
yclust=gmm.predict(X)
plt.scatter(X[:,0], X[:,1],c=vectorizer(yclust))
from sklearn import metrics
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
from sklearn import datasets
myiris = datasets.load_iris()
x = myiris.data
y = myiris.target
lab=list()
pure=list()
gmm = mixture.GaussianMixture(n_components=3, covariance_type='tied',max_iter=10,init_params='random')
gmm.fit(x)
yclust=gmm.predict(x)
lab.append('Random 10 tied')
pure.append(purity_score(y,yclust))
pure
gmm = mixture.GaussianMixture(n_components=3, covariance_type='tied',max_iter=100,init_params='random')
gmm.fit(x)
yclust=gmm.predict(x)
lab.append('Random 100 tied')
pure.append(purity_score(y,yclust))

gmm = mixture.GaussianMixture(n_components=3, covariance_type='full',max_iter=10,init_params='random')
gmm.fit(x)
yclust=gmm.predict(x)
lab.append('Random 10 full')
pure.append(purity_score(y,yclust))
gmm = mixture.GaussianMixture(n_components=3, covariance_type='full',max_iter=100,init_params='random')
gmm.fit(x)
yclust=gmm.predict(x)
lab.append('Random 100 full')
pure.append(purity_score(y,yclust))

gmm = mixture.GaussianMixture(n_components=3, covariance_type='full',max_iter=100,init_params='kmeans')
gmm.fit(x)
yclust=gmm.predict(x)
lab.append('Init kmeans 100 full')
pure.append(purity_score(y,yclust))
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'random', max_iter = 300, n_init = 10, random_state = 0)
yclust = kmeans.fit_predict(x)
lab.append('kmeans')
pure.append(purity_score(y,yclust))
yclust.shape
d = {'Method':lab,'Purity':pure}
import pandas as pd
df = pd.DataFrame(d)
df.plot.barh(x='Method',y='Purity',title='EM with various paramters', color=tuple(["g", "b","r","y","k"]))
n_components = np.arange(1, 15,2)
models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(x)
          for n in n_components]
plt.plot(n_components, [m.bic(x) for m in models], label='BIC')
plt.xlabel('n_components');