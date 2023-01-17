import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import os



from sklearn import preprocessing

from sklearn.decomposition import PCA
os.listdir('../input/lish-moa')
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')
GENES = [col for col in train_features.columns if col.startswith('g-')]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])



data_rescaled = scaler.fit_transform(data)
pca = PCA().fit(data_rescaled)



import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12,6)



fig, ax = plt.subplots()

xi = np.arange(1, 773, step=1)

y = np.cumsum(pca.explained_variance_ratio_)



plt.ylim(0.0,1.1)

plt.plot(xi, y, marker='o', linestyle='--', color='b')



plt.xlabel('Number of Components')

plt.xticks(np.arange(0, 720, step=40)) #change from 0-based array index to 1-based human-readable label

plt.ylabel('Cumulative variance (%)')

plt.title('The number of components needed to explain variance')



plt.axhline(y=0.95, color='r', linestyle='-')

plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)



ax.grid(axis='x')

plt.show()