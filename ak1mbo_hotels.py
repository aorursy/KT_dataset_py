# !pip install tensorflow
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle

import sklearn

import sklearn.metrics

import tensorflow as tf



from sklearn.decomposition import PCA

from sklearn.preprocessing import label_binarize

from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier as KNN







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# reads chunks of data from test.csv and makes a DataFrame for the test set



test_df = pd.DataFrame()

chunks = [test_df]



for chunk in pd.read_csv('../input/test.csv', sep=',', chunksize=1e6):

    chunks += [chunk]



test_df = pd.concat(chunks)

test_df.head()
# summary statistics for test data set



test_df.describe()
train_df = pd.DataFrame()

chunks = [train_df]



sample_size = int(1e5)

bin_num = 38

bin_sample_size = int(sample_size // bin_num)



for chunk in pd.read_csv('../input/train.csv', sep=',', chunksize=1e6):

    if sample_size - (2 * bin_sample_size) < 0:

#         temp = chunk.loc[chunk['is_booking'] == 1].sample(sample_size)

        temp = chunk.sample(sample_size)

    else:

#         temp = chunk.loc[chunk['is_booking'] == 1].sample(bin_sample_size)

        temp = chunk.sample(bin_sample_size)

        

    print(len(chunks))

    

    sample_size -= bin_sample_size

    chunks += [temp]





train_df = pd.concat(chunks)

train_df.head()
train_df.describe()
dest_df = pd.DataFrame()



for chunk in pd.read_csv('../input/destinations.csv', sep=',', chunksize=1e6):

    dest_df = pd.concat([dest_df, chunk])



dest_df.head()
dest_df.describe()
# frequency count of 5 most popular hotel clusters from training sample



freq_df = pd.DataFrame(train_df['hotel_cluster'].copy())

freq_df.columns = ['actual']





freq_predictions = train_df['hotel_cluster'].value_counts().head().index.tolist()

freq_df['freq_predict'] = [freq_predictions[0] for i in range(freq_df.shape[0])]

temp = [freq_predictions[0] for i in range(freq_df.shape[0])]

freq_df.head()
y = freq_df['actual'].values

y = label_binarize(y, classes=list(range(0,100)))

x = label_binarize(temp, classes=list(range(0,100)))



sklearn.metrics.average_precision_score(y, x)
cats = train_df[['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'user_id', 'is_mobile', 'is_package', 'channel', 'srch_destination_id', 'srch_destination_type_id', 'hotel_continent', 'hotel_country', 'hotel_market', 'is_booking', 'cnt', 'hotel_cluster']]

cats.corr(method='spearman')
# check Pearson's Correlation Coefficient (r) values for every search feature against each other



rcorrs = train_df.corr()



for i, r in rcorrs.iterrows():

    for j in range(rcorrs.shape[1]):

        if np.abs(r[j]) < 0.4 or np.abs(r[j]) == 1:

            pass

#             r[j] = np.NaN

    

rcorrs
pca = PCA(n_components=5)

# dest_pca = dest_pca.fit_transform(dest_df.iloc[:,1:])

dest_pca = pca.fit_transform(dest_df.iloc[:,1:])

dest_pca = pd.DataFrame(dest_pca)

dest_pca.head()
enc = OneHotEncoder()



temp = train_df[['hotel_continent', 'site_name', 'srch_rm_cnt', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'cnt', 'hotel_cluster']].copy().dropna()

onehot = pd.get_dummies(temp[['hotel_continent', 'site_name', 'srch_rm_cnt', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'cnt']])

temp = temp.drop(['hotel_continent', 'site_name', 'srch_rm_cnt', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'cnt'], axis=1)

temp = temp.join(onehot)

# dist = enc.fit([temp['orig_destination_distance']])
# print(dist)

# print(temp['hotel_cluster'].shape)

print(temp.head())

print(onehot.head())

print(temp.columns)

print(temp.shape)

# ['site_name', 'user_location_country', 'orig_destination_distance', 'srch_adults_cnt', 'srch_rm_cnt', 'srch_destination_id', 'hotel_continent', 'hotel_country']

neighbors = KNN()

neighbors.fit(temp[['hotel_continent', 'site_name', 'srch_rm_cnt', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'cnt']], train_df['hotel_cluster'])

# neighbors.fit(train_df.iloc[:,:24], train_df.iloc[:,23:])
# # dest_df.iloc[:,1:].shape

# # dest_df[["d{0}".format(i + 1) for i in range(149)]].shape





# # onehot = pd.get_dummies(train_df['hotel_cluster'])

# temp[['hotel_continent', 'site_name', 'srch_rm_cnt', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'cnt']].head()

# neighbors.predict([[0, 0, 0, 0, 0, 1]])



temp_test = train_df[['hotel_continent', 'site_name', 'srch_rm_cnt', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'cnt']].sample(temp.shape[0])

print(temp_test.shape)

print(temp.shape)



predictions = neighbors.predict(temp_test)
# from sklearn.multioutput import MultiOutputClassifier

# from sklearn.utils import shuffle
print(predictions[:5])

temp_test.head()
temp = train_df[['hotel_continent', 'hotel_cluster']]

tempshape = temp.shape

print(temp.shape)

# print(temp['hotel_continent'].dtype)

temp.head()
import tensorflow as tf
temp = temp.as_matrix()

countries_tensor = tf.constant(temp, dtype=tf.int64, shape=tempshape)
country = tf.feature_column.categorical_column_with_vocabulary_list('hotel_continent', vocabulary_list=[0, 1, 2, 3, 4, 5, 6])



temp_tensor = tf.estimator.LinearClassifier(feature_columns=['hotel_continent'])

# temp_tensor.train(input_fn=train_df[['hotel_continent', 'hotel_cluster']], steps=200)
train_df.head()
import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np



tf.set_random_seed(1)

np.random.seed(1)



# fake data

n_data = np.ones((100, 2))

x0 = np.random.normal(2*n_data, 1)      # class0 x shape=(100, 2)

y0 = np.zeros(100)                      # class0 y shape=(100, 1)

x1 = np.random.normal(-2*n_data, 1)     # class1 x shape=(100, 2)

y1 = np.ones(100)                       # class1 y shape=(100, 1)

x = np.vstack((x0, x1))  # shape (200, 2) + some noise

y = np.hstack((y0, y1))  # shape (200, )



# plot data

plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=0, cmap='RdYlGn')

plt.show()



tf_x = tf.placeholder(tf.float32, x.shape)     # input x

tf_y = tf.placeholder(tf.int32, y.shape)     # input y



# neural network layers

l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)          # hidden layer

output = tf.layers.dense(l1, 2)                     # output layer



loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)           # compute cost

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables

    labels=tf.squeeze(tf_y), predictions=tf.argmax(output, axis=1),)[1]

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)

train_op = optimizer.minimize(loss)



sess = tf.Session()                                                                 # control training and others

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

sess.run(init_op)     # initialize var in graph



plt.ion()   # something about plotting

for step in range(100):

    # train and net output

    _, acc, pred = sess.run([train_op, accuracy, output], {tf_x: x, tf_y: y})

    if step % 2 == 0:

        # plot and show learning process

        plt.cla()

        plt.scatter(x[:, 0], x[:, 1], c=pred.argmax(1), s=100, lw=0, cmap='RdYlGn')

        plt.text(1.5, -4, 'Accuracy=%.2f' % acc, fontdict={'size': 20, 'color': 'red'})

        plt.pause(0.1)



plt.ioff()

plt.show()