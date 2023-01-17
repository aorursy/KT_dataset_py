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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = '../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv'
data = pd.read_csv (data)
data.head ()
round (data [['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']].describe (),2)
data.columns
plt.figure (figsize = (12,4))
plt.subplot (1,3,1)
sns.distplot (data ['Annual Income (k$)'], hist = False)

plt.subplot (1,3,2)
sns.distplot (data ['Age'], hist = False)

plt.subplot (1,3,3)
sns.distplot (data ['Spending Score (1-100)'], hist = False)

plt.tight_layout ()
plt.figure (figsize = (16,4))
plt.subplot (1,4,1)
data.groupby ('Gender')['Age'].mean ().plot.bar (title = 'Age')

plt.subplot (1,4,2)
data.groupby ('Gender')['Annual Income (k$)'].mean ().plot.bar (title = 'Income')

plt.subplot (1,4,3)
data.groupby ('Gender')['Spending Score (1-100)'].mean ().plot.bar (title = 'Spending_score')

plt.subplot (1,4,4)
data ['Gender'].value_counts (normalize = True).plot.bar (title = 'Gender_count')

plt.tight_layout ()
data ['Gender'] = data ['Gender'].replace ({'Male': 0, 'Female' : 1})
plt.figure (figsize = (16,5))

plt.subplot (1,3,1)
sns.scatterplot (data = data, x = 'Age', y = 'Annual Income (k$)', hue = 'Gender')
plt.title (data [['Age','Annual Income (k$)']].corr ().iloc [0,1])

plt.subplot (1,3,2)
sns.scatterplot (data = data, x = 'Age', y = 'Spending Score (1-100)', hue = 'Gender')
plt.title (data [['Age','Spending Score (1-100)']].corr ().iloc [0,1])

plt.subplot (1,3,3)
sns.scatterplot (data = data, x = 'Annual Income (k$)', y = 'Spending Score (1-100)', hue = 'Gender')
plt.title (data [['Annual Income (k$)','Spending Score (1-100)']].corr ().iloc [0,1])

plt.tight_layout ()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
X = StandardScaler ().fit_transform (data [['Age', 'Gender','Annual Income (k$)', 'Spending Score (1-100)']].values)
inertia = {}
for k in range (2,30):
    KMean = KMeans(n_clusters=k, init='k-means++', n_init=10, verbose=0, random_state=1, algorithm='elkan').fit (X)
    inertia.update ({k : KMean.inertia_})
inertia_df = pd.DataFrame (inertia, index = [0]).transpose ()
inertia_df.columns = ['inertia']

inertia_df.plot ()
round (inertia_df ['inertia'].pct_change ()*100)
K_Means = KMeans(n_clusters=3, init='k-means++', n_init=16, verbose=0, random_state=1, algorithm='elkan').fit (X)
labels = K_Means.labels_
K_Means.labels_
centers = K_Means.cluster_centers_
K_Means.cluster_centers_
plt.figure (figsize = (8,6))
area = np.pi * ( data.iloc [:, 1])**20
plt.scatter(data.iloc[:, 2], data.iloc[:, 3], s=30*area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(4, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=44, azim=70)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Income')
ax.set_ylabel('Age')
ax.set_zlabel('Score')

ax.scatter(data.iloc[:, 2], data.iloc[:, 0], data.iloc[:, 3], c= labels.astype(np.float))

data ['labels'] = pd.Series (labels)
data.head ()
clustering = round (data [['Gender', 'Age', 'Annual Income (k$)',
       'Spending Score (1-100)', 'labels']].groupby (['labels']).mean (),2).reset_index ()
clustering
bins = [0,40,70,100]
labels = ['low_score','mid_score', 'high_score' ]
clustering ['score_cluster'] = pd.cut (clustering ['Spending Score (1-100)'], bins, labels = labels, include_lowest = True)

bins = [0,40,80,140]
labels = ['low_income','mid_income', 'high_income' ]
clustering ['income_cluster'] = pd.cut (clustering ['Annual Income (k$)'], bins, labels = labels, include_lowest = True)

bins = [0,35,60,70]
labels = ['young', 'mature', 'aged']
clustering ['age_cluster'] = pd.cut (clustering ['Age'], bins, labels = labels, include_lowest = True)

clustering ['gender_cluster'] = clustering ['Gender'].apply (lambda x : 'male' if x < 0.5 else 'female')
clustering [['score_cluster', 'income_cluster',
       'age_cluster', 'gender_cluster']]
