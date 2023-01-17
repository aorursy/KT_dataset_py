# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# %matplotlib inline
plt.style.use("ggplot")

import sklearn
from sklearn.decomposition import TruncatedSVD

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
amazon_rate = pd.read_csv('../input/amazon-ratings/ratings_Beauty.csv')
amazon_rate= amazon_rate.dropna()
amazon_rate.head()
#print(amazon_rate.shape)
popular = pd.DataFrame(amazon_rate.groupby('ProductId')['Rating'].count())
Best = popular.sort_values('Rating',ascending=False)
Best.head(10)
Best.head(30).plot(kind='bar')
amazon_rating = amazon_rate.head(10000)
utlity_matrix = amazon_rating.pivot_table(values='Rating',index='UserId',columns= 'ProductId',fill_value=0)
utlity_matrix.head()
utlity_matrix.shape
X = utlity_matrix.T
X.head()
X.shape
SVD = TruncatedSVD(n_components=10)
decomposed = SVD.fit_transform(X)
decomposed.shape
correlation = np.corrcoef(decomposed)
correlation
X.sample(3)
i = X.index[400]
product = list(X.index)
product_ID = product.index(i)
product_ID

corr_prodID = correlation[product_ID]
corr_prodID
Recommend = list(X.index[corr_prodID > 0.9])
Recommend.remove(i)
Recommend[0:8]
Product_descrip = pd.read_csv('../input/product-description-amazon/product_descriptions.csv')
Product_descrip = Product_descrip.dropna()
Product_descrip.head()
Prod_descrip = Product_descrip.head(700)
Prod_descrip['product_description'].head(10)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
vectorizer = TfidfVectorizer(stop_words='english')
Data = vectorizer.fit_transform(Prod_descrip["product_description"])
Data
X=Data

kmeans = KMeans(n_clusters = 10, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)
plt.plot(y_kmeans, ".")
plt.show()

# Cluster 

True_k = 10
model = KMeans(n_clusters=True_k, init='k-means++', max_iter=100, n_init=1)
model.fit(Data)
print("Top terms/ Clusters :" )
order = model.cluster_centers_.argsort()[:,::-1]
terms = vectorizer.get_feature_names()
for a in range(True_k):
    print("Cluster %d:" % a),
    for i in order[a, :10]:
        print(' %s' % terms[i]),
    print
print("Cluster ID:")
M = vectorizer.transform(["privacy"])
Prediction = model.predict(M)
print(Prediction)