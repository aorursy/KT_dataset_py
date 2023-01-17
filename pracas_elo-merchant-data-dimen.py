# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 12) ### Setting the size of the Plots

from keras.layers import Input, Dense
from keras.models import Model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#### Lets load the file
merchant_df = pd.read_csv("../input/merchants.csv")
#### Lets take a quick look at the basic info. some of the columns have missing values
merchant_df.info()
#### Next lets look at the numeric values and see if everything is ok. 
merchant_df.describe()
#### We can see that 3 columns have the maximum value as inf, Lets see how many rows are affected
print(merchant_df[merchant_df['avg_purchases_lag3']==float('Inf')].merchant_id.count())
print(merchant_df[merchant_df['avg_purchases_lag6']==float('Inf')].merchant_id.count())
print(merchant_df[merchant_df['avg_purchases_lag12']==float('Inf')].merchant_id.count())
merchant_df[merchant_df['avg_purchases_lag3']==float('Inf')]
#### We can see that only 3 rows are affected. we can drop these rows
merchant_df = merchant_df.drop(merchant_df.index[merchant_df.avg_purchases_lag3 ==float('Inf')],0)
merchant_df.describe()
#### Lets look at handling some of the outliers in the sales lag / purchase lag fields
merchant_df.plot.scatter(x='avg_sales_lag3', y='avg_purchases_lag3')
plt.show()
### we can see only 16 records are getting affected. Lets drop these
### We can repeat the scatter plot for avg_sales_lag6, avg_sales_lag12 as well
print(merchant_df[merchant_df['avg_sales_lag3']>20000].merchant_id.count())
merchant_df = merchant_df.drop(merchant_df.index[merchant_df.avg_sales_lag3>20000],0)
merchant_df.plot.scatter(x='avg_sales_lag6', y='avg_purchases_lag6')
plt.show()
### we can see only 16 records are getting affected. Lets drop these
### We can repeat the scatter plot for avg_sales_lag6, avg_sales_lag12 as well
print(merchant_df[merchant_df['avg_sales_lag6']>20000].merchant_id.count())
merchant_df = merchant_df.drop(merchant_df.index[merchant_df.avg_sales_lag6>20000],0)
merchant_df.plot.scatter(x='avg_sales_lag12', y='avg_purchases_lag12')
plt.show()
### we can see only 16 records are getting affected. Lets drop these
### We can repeat the scatter plot for avg_sales_lag6, avg_sales_lag12 as well
print(merchant_df[merchant_df['avg_sales_lag12']>20000].merchant_id.count())
merchant_df = merchant_df.drop(merchant_df.index[merchant_df.avg_sales_lag12>20000],0)
merchant_df.describe()
##### Next Lets handle the 4 categorical values
### category_1, category_4  have Y or N we can replace this with 0 / 1
### most_recent_sales_range, most_recent_purchases_range are kind of ranking of some sort with A>B>C>D>E, we can replace these with 5>4>3>2>1
clean_up_categoricals = {'category_1':{'Y':1, 'N':0},
                         'category_4' :{'Y':1, 'N':0},
                        'most_recent_sales_range' : {'A':5,'B':4,'C':3,'D':2,'E':1},
                        'most_recent_purchases_range' : {'A':5,'B':4,'C':3,'D':2,'E':1}}
merchant_df.replace(clean_up_categoricals, inplace=True)
merchant_df.head(5)
##### Finally lets replace the null values with the respective means of the columns
merchant_df = merchant_df.fillna(merchant_df.mean())
merchant_df.head(5)
#### Next we can extract our X component out of this - All fields excluding the merchant_id
X = merchant_df
X = X.drop('merchant_id',1)
X.head(2)
X.info()
#### Lets use the standard scaler to scale all our columns
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size = 0.2, random_state = 0)

# this is the size of our encoded representations
encoding_dim = 6  # we need our 21 columns to be encoded as 6
# this is our input placeholder
input_shape = Input(shape=(21,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_shape)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(21, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_shape, decoded)
# this model maps an input to its encoded representation
encoder = Model(input_shape, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
#### Compile the autoencoder
autoencoder.compile(optimizer='adadelta', loss='MSE')
#### Train the autoencoder
autoencoder.fit(X_train, X_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))
#### Get our encoded data
encoded_data = encoder.predict(X)
encoded_data.shape
#### Lets find the optimal number of clusters using the elbow method by plotting wcss against the number of clusters
from sklearn.cluster import KMeans
wcss = []

for i in range (1,11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300,
                    n_init = 10, random_state = 0)
    kmeans.fit(encoded_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

##### 4 seems to be a good number of clusters

kmeans = KMeans(n_clusters = 4, init = "k-means++", max_iter = 300,
                    n_init = 10, random_state = 0)
ykmeans = kmeans.fit_predict(encoded_data)
plt.scatter(encoded_data[ykmeans==0,0], encoded_data[ykmeans==0,1], s = 10, c='red', label = "1")
plt.scatter(encoded_data[ykmeans==1,0], encoded_data[ykmeans==1,1], s = 10, c='blue', label = "2")
plt.scatter(encoded_data[ykmeans==2,0], encoded_data[ykmeans==2,1], s = 10, c='green', label = "3")
plt.scatter(encoded_data[ykmeans==3,0], encoded_data[ykmeans==3,1], s = 10, c='cyan', label = "4")

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s= 100, c="yellow", label = "Centroids")
plt.title("Clusters of Clients")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

##### Finally Lets Create a new dataframe that will have just 2 columns - the merchant_id and the corresponding cluster id
merchant_id = merchant_df['merchant_id']
print(merchant_id.shape)
print(ykmeans.shape)

new_df = pd.DataFrame()
new_df['merchant_id'] = merchant_df['merchant_id']
new_df['cluster_id'] = ykmeans
print(new_df.info())
new_df.to_csv('merchant_id_clusters.csv', index=False)