#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.structured import *
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
PATH = "../input/"
# Any results you write to the current directory are saved as output.
df_train = pd.read_csv(f'{PATH}train.csv', parse_dates=['Open Date'])
df_test = pd.read_csv(f'{PATH}test.csv', parse_dates=['Open Date'])

df_joined = pd.concat([df_train.drop('revenue', axis=1), df_test], axis=0)
sns.distplot(df_train['revenue'])
sns.distplot(df_train['revenue'].apply(np.sqrt))
sns.distplot(df_train['revenue'].apply(np.sqrt).apply(np.log));
train_cats(df_train)
df_train = df_train.drop('Open Date', axis=1)
X, _ , _ = proc_df(df_train)
X.hist(figsize=(15,10));
#correlation matrix
corrmat = X.corr()
f, ax = plt.subplots(figsize=(16, 16))
sns.heatmap(corrmat, vmax=1.2, square=True);
from scipy.cluster import hierarchy as hc

corr = np.round(scipy.stats.spearmanr(X).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=X.columns, orientation='left', leaf_font_size=16)
plt.show()
clf = LocalOutlierFactor(n_neighbors=10, contamination=0.1)

a = [i for i in range(len(df_train))]
X = df_train.revenue.values.reshape(-1,1)
y_pred = clf.fit_predict(X)
#n_errors = (y_pred != ground_truth).sum()
X_scores = clf.negative_outlier_factor_

plt.title("Local Outlier Factor (LOF)")
plt.scatter(a, X[:, 0], color='k', s=3., label='Data points')
# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
plt.scatter(a, X[:, 0], s=1000 * radius, edgecolors='r',
            facecolors='none', label='Outlier scores')
plt.axis('tight')
#plt.xlim((-5, 5))
#plt.ylim((-5, 5))
plt.xlabel("Id")
legend = plt.legend(loc='upper right')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()
df_train[df_train.revenue > 1e7]
sns.categorical.countplot(df_train['City'])
from sklearn import cluster
# K Means treatment for city (mentioned in the paper)
def adjust_cities(data, train, k):
    
    # As found by box plot of each city's mean over each p-var
    relevant_pvars =  ["P1", "P2", "P11", "P19", "P20", "P23", "P30"]
    train = train.loc[:, relevant_pvars]
    
    # Optimal k is 20 as found by DB-Index plot    
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(train)
    
    # Get the cluster centers and classify city of each data instance to one of the centers
    data['City Cluster'] = kmeans.predict(data.loc[:, relevant_pvars])
    del data["City"]
    
    return data

# Convert unknown cities in test data to clusters based on known cities using KMeans
data = adjust_cities(df_test, df_train, 20)
sns.categorical.countplot(data['City Cluster'])
