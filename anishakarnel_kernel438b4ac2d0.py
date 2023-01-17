# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
country_dataset=pd.read_csv("/kaggle/input/unsupervised-learning-on-country-data/Country-data.csv")
country_datadict=pd.read_csv("/kaggle/input/unsupervised-learning-on-country-data/Country-data.csv")
sn.boxplot(x=country_dataset['income'])
country_dataset.shape
country_dataset = country_dataset.drop(['country'], axis=1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
cols=country_dataset.columns
country_dataset = scaler.fit_transform(country_dataset)
country_dataset=pd.DataFrame(data=country_dataset,columns=cols)
from sklearn.cluster import KMeans
ssd = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(country_dataset)
    ssd.append(km.inertia_)
plt.figure(figsize=(10,6))
plt.plot(K, ssd, 'bx-')
plt.xlabel('k')
plt.ylabel('ssd')
plt.title('Elbow Method For Optimal k')
plt.show()
# Create and fit model
kmeans = KMeans(n_clusters=3)
model = kmeans.fit(country_dataset)
pred = model.labels_
country_dataset['cluster'] = pred
country_dataset.head(10)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_model = pca.fit_transform(country_dataset)
data_transform = pd.DataFrame(data = pca_model, columns = ['PCA1', 'PCA2','PCA3'])
data_transform['Cluster'] = pred
data_transform.head()
