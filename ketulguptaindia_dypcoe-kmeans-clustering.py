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
import pandas as pd

CLV = pd.read_csv("../input/CLV.csv")
CLV.describe()

import matplotlib.pyplot as plt

plt.scatter(CLV.INCOME,CLV.SPEND)

plt.xlabel("Ïncome")

plt.ylabel("Spends")

plt.title("Visualizing Data")

plt.show()
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

    km=KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)

    km.fit(CLV)

    wcss.append(km.inertia_)

plt.plot(range(1,11),wcss)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('wcss')

plt.show()
#Plot styling

import seaborn as sns; sns.set()  # for plot styling

%matplotlib inline

plt.rcParams['figure.figsize'] = (16, 9)

plt.style.use('ggplot')
##Fitting kmeans to the dataset with k=4

X=CLV

km4=KMeans(n_clusters=4,init='k-means++', max_iter=300, n_init=10, random_state=0)

y_means = km4.fit_predict(X)



plt.scatter(X.INCOME, X.SPEND, c=y_means, s=50, cmap='viridis')



centers = km4.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5);

plt.title('Customer segments')

plt.xlabel('Annual income of customer')

plt.ylabel('Annual spend from customer on site')

plt.legend()

plt.show()
km4=KMeans(n_clusters=6,init='k-means++', max_iter=300, n_init=10, random_state=0)

y_means = km4.fit_predict(X)



plt.scatter(X.INCOME, X.SPEND, c=y_means, s=50, cmap='viridis')



centers = km4.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5);

plt.title('Customer segments')

plt.xlabel('Annual income of customer')

plt.ylabel('Annual spend from customer on site')

plt.legend()

plt.show()
X