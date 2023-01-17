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
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
%matplotlib inline
df = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
df.head()
df.rename(columns = {'Annual Income (k$)' :'Income','Spending Score (1-100)':'Score'},inplace = True)
df.hist()
plt.show()
plt.rcParams['figure.figsize'] = (15, 8)
sns.countplot(df['Age'], palette = 'hsv')
plt.title('Distribution of Age', fontsize = 20)
plt.show()
sns.pairplot(df)
plt.rcParams['figure.figsize'] = (15, 8)
sns.heatmap(df.corr(), cmap = 'Wistia', annot = True)
plt.title('Heatmap for the Data', fontsize = 20)
plt.show()
Gend = {'Male' : 0,'Female' : 1}
df['Gender'] = df['Gender'].map(Gend)
df
x = df.iloc[:, [2, 4]].values
x
df.drop(['CustomerID','Gender'],axis = 1,inplace = True)
df.columns
km = KMeans(n_clusters = 4)
y_pred = km.fit_predict(df)
y_pred
df['cluster'] = y_pred
df.head(4)
df.cluster.unique()
#Elbow plot
from sklearn.cluster import KMeans
sse = []
for k in range(1,11):
    km = KMeans(n_clusters = k)
    km.fit(df)
    sse.append(km.inertia_)
plt.rcParams['figure.figsize'] = (4,3)
plt.grid()
plt.xlabel('k')
plt.ylabel("sum of square error")
plt.plot(range(1,11),sse)
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
ymeans = kmeans.fit_predict(df)

plt.rcParams['figure.figsize'] = (10, 10)
plt.title('Cluster of Ages', fontsize = 30)
df = np.array(df)
plt.scatter(df[ymeans == 0, 0], df[ymeans == 0, 1], s = 100, c = 'pink', label = 'Usual Customers' )
plt.scatter(df[ymeans == 1, 0], df[ymeans == 1, 1], s = 100, c = 'orange', label = 'Priority Customers')
plt.scatter(df[ymeans == 2, 0], df[ymeans == 2, 1], s = 100, c = 'lightgreen', label = 'Target Customers(Young)')
plt.scatter(df[ymeans == 3, 0], df[ymeans == 3, 1], s = 100, c = 'red', label = 'Target Customers(Old)')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'black')

plt.style.use('fivethirtyeight')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid()
plt.show()