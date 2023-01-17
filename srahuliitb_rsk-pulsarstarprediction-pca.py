# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
stars_df = pd.read_csv('/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv')

stars_df.head()
stars_df.shape
stars_df.info()
stars_df.isnull().sum()
stars_df.isna().sum()
stars_df['target_class'].value_counts()
X = stars_df.iloc[:, :len(stars_df.columns) - 1].values

X
y = stars_df['target_class'].values

y
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X_scaled = scaler.fit_transform(X)

X_scaled
X_scaled.shape
np.mean(X_scaled), np.std(X_scaled)
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)



pri_comps = pca.fit_transform(X_scaled)

pri_comps
pri_comps_df = pd.DataFrame(data = pri_comps, columns = ['PC-1', 'PC-2'])

pri_comps_df.head()
pca.explained_variance_ratio_
final_df = pd.concat([pri_comps_df, stars_df['target_class']], axis = 1)

final_df.head()
import matplotlib.pyplot as plt



fig = plt.figure(figsize = (10, 10))

ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('PC-1', fontsize = 15)

ax.set_ylabel('PC-2', fontsize = 15)

ax.set_title('Scatter Plot Principal Components', fontsize = 20)



targets = [0, 1]

colors = ['r', 'g']

for target, color in zip(targets, colors):

    indicesToKeep = final_df['target_class'] == target

    ax.scatter(final_df.loc[indicesToKeep, 'PC-1']

               , final_df.loc[indicesToKeep, 'PC-2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()