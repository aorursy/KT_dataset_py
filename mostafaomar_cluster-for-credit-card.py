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
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
data = pd.read_csv('../input/ccdata/CC GENERAL.csv')
data.head()
data.info
data.describe()
#correlation matrix
corrmat = data.corr()
f, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(corrmat, vmax=.8, annot=True);
data = data.drop('CUST_ID', axis = 1) 

for col in data:
    data[[col]].hist()

data.isna().sum()
data.loc[(data['MINIMUM_PAYMENTS'].isnull()==True),'MINIMUM_PAYMENTS']=data['MINIMUM_PAYMENTS'].mean()
data.loc[(data['CREDIT_LIMIT'].isnull()==True),'CREDIT_LIMIT']=data['CREDIT_LIMIT'].mean()
data.isna().sum()
data.boxplot(rot=100, figsize=(40,20))

cols = list(data)
irq_score = {}

for c in cols:
    q1 = data[c].quantile(0.25)
    q3 = data[c].quantile(0.75)
    score = q3 - q1
    outliers = data[(data[c] < q1 - 1.5 * score) | (data[c] > q3 + 1.5 * score)][c]
    values = data[(data[c] >= q1 - 1.5 * score) | (data[c] <= q3 + 1.5 * score)][c]
    
    irq_score[c] = {
        "Q1": q1,
        "Q3": q3,
        "IRQ": score,
        "n_outliers": outliers.count(),
        "outliers_avg": outliers.mean(),
        "outliers_stdev": outliers.std(),
        "outliers_median": outliers.median(),
        "values_avg:": values.mean(),
        "values_stdev": values.std(),
        "values_median": values.median(),
    }
    
irq_score = pd.DataFrame.from_dict(irq_score, orient='index')

irq_score

data.shape

from scipy import stats
import numpy as np
z = np.abs(stats.zscore(data))
print(z)

threshold = 3
print(np.where(z > 3))
data = data[(z < 3).all(axis=1)]

data.boxplot(rot=90, figsize=(30,10))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data = sc.fit_transform(data)

data = pd.DataFrame(data)
data.head()
data.shape
# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range (1,12):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=40)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,12),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_km = kmeans.fit_predict(data)
print(y_km)
labels = kmeans.labels_
labels
from sklearn.decomposition import PCA
pca = PCA(2)
principalComponents = pca.fit_transform(data)
x, y = principalComponents[:, 0], principalComponents[:, 1]
print(principalComponents.shape)

colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'purple'}
final_data = pd.DataFrame({'x': x, 'y':y, 'label':labels}) 
groups = final_data.groupby(labels)
fig, ax = plt.subplots(figsize=(15, 10)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, color=colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.set_title("Customer Segmentation based on Credit Card usage")
plt.show()


