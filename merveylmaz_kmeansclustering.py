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
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()  # plot un şekli için

import numpy as np



from sklearn.datasets.samples_generator import make_blobs  # Dataseti kulanmak için

X, y_true = make_blobs(n_samples=350, centers=3,

                       cluster_std=0.70, random_state=0)   # Datasetimiz oluşturuldu. 350 tane, 3 merkezli olacak şekilde, merkeze yakınlık oranı 0,70, rastgele veriler oluşturuldu



plt.scatter(X[:, 0], X[:, 1], s=50);                       # İki sütundan oluşan veri seti plt üzerinde gösterildi
from sklearn.cluster import KMeans        # Kmeans için 



kmeans = KMeans(n_clusters=3)             # k=3 şeçildi

kmeans.fit(X)                             

y_kmeans = kmeans.predict(X)              # Tahmin işlemi yapıldı



plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')



centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.5);  # Kümeleme işleminin merkezleri kırmızı daireler ile belirtildi