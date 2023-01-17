# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import cufflinks as cf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
init_notebook_mode()
cf.go_offline()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/wholesale-customers-data-set/Wholesale customers data.csv')
df.head()
fig = go.Figure(data=[go.Surface(z=df[df.Region == 1].corr().values), 
                      go.Surface(z=df[df.Region == 2].corr().values + 3, showscale=False, opacity=0.9), 
                      go.Surface(z=df[df.Region == 3].corr().values + 6, showscale=False, opacity=0.9)])

fig.update_layout(title='Region Correlation', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))


fig.show()
fig = go.Figure(data=[go.Surface(z=df[df.Channel == 1].corr().values), 
                      go.Surface(z=df[df.Channel == 2].corr().values + 3, showscale=False, opacity=0.9)])

fig.update_layout(title='Channel Correlation', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

fig.show()
targets = df.Channel
df.drop(['Channel', 'Region'], inplace=True, axis=1)
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(df)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X)
# principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
pca.explained_variance_.round(2)
pca.explained_variance_ratio_
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
y_pred = kmeans.predict(X) + 1
kmeans_pca = KMeans(n_clusters=2, random_state=0).fit(principalDf.values)
y_pred_pca = kmeans_pca.predict(principalDf.values) + 1
ax = sns.scatterplot(x=X[:,0], y=X[:,1], hue=y_pred)
ax = sns.scatterplot(x=X[:,0], y=X[:,1], hue=y_pred_pca)
