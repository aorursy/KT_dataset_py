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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# %matplotlib inline

plt.style.use("ggplot")



import sklearn

from sklearn.decomposition import TruncatedSVD
df_ratings = pd.read_excel('/kaggle/input/Ratings.xlsx', header=None)

df_ratings = df_ratings.dropna()

df_ratings.columns = ["UserId","ProductId","Rating","Timestamp"]

df_ratings.head()
df_ratings['Rating'].hist()
print("Count of Unique Users is "+str(len(df_ratings['UserId'].unique())))

print("Count of Unique Products is "+str(len(df_ratings['ProductId'].unique())))
from collections import Counter

 # equals to list(set(words))

val = list(Counter(df_ratings['Rating']).values()) # counts the elements' frequency

labels = list(Counter(df_ratings['Rating']).keys())

import matplotlib.pyplot as plt

my_colors = ['lightblue','lightsteelblue','silver','darkblue','grey']

plt.pie(val, labels=labels, autopct='%1.1f%%', startangle=15, shadow = True, colors=my_colors, explode=None)

plt.title('Ratings')

plt.axis('equal')

plt.show()
df = df_ratings[df_ratings['Rating']==1]

df_withLowRating = pd.DataFrame(df.groupby('ProductId')['Rating'].count())

df_withLowRating = df_withLowRating.sort_values('Rating', ascending=False)

Products_with_Rating1 = df_withLowRating[df_withLowRating['Rating']>50]

Products_with_LowestRating = list(Products_with_Rating1.index)

print("There are "+str(len(Products_with_LowestRating)) + " products which got Rating 1 more than 50 times.")

Products_with_LowestRating
Regular_users = pd.DataFrame(df_ratings.groupby('UserId')['Rating'].count())

Regular_users = Regular_users.sort_values('Rating', ascending=False)

Regular_users_1 = Regular_users[Regular_users['Rating']>50]

Regular_users_list = list(Regular_users_1.index)

less_sparse_denser_df = df_ratings[df_ratings.UserId.isin(Regular_users_list)]

less_sparse_denser_df = less_sparse_denser_df.reset_index(drop=True)

less_sparse_denser_df.info()
popular_products = pd.DataFrame(df_ratings.groupby('ProductId')['Rating'].count())

most_popular = popular_products.sort_values('Rating', ascending=False)

most_popular.head(10)
most_popular.head(30).plot(kind = "bar")
ratings_utility_matrix = less_sparse_denser_df.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)

ratings_utility_matrix.head()
ratings_utility_matrix.shape
X = ratings_utility_matrix.T

X.head()
X.shape
X1 = X
SVD = TruncatedSVD(n_components=10)

decomposed_matrix = SVD.fit_transform(X)

decomposed_matrix.shape
correlation_matrix = np.corrcoef(decomposed_matrix)

correlation_matrix.shape
X.index[99]
i = 'B00003006R'



product_names = list(X.index)

product_ID = product_names.index(i)

product_ID
correlation_product_ID = correlation_matrix[product_ID]

correlation_product_ID.shape
Recommend = list(X.index[correlation_product_ID > 0.90])

# Removes the item already bought by the customer

Recommend.remove(i) 

Recommend
Recommend = list(X.index[correlation_product_ID > 0.90])

# Removes the item already bought by the customer

Recommend.remove(i) 

Recommend[0:5]