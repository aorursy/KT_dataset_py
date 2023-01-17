from pandas import *

f = read_csv(r'../input/train.csv')

#read the the first few lines of the data

print(f.head())

#check if there are any duplicates

f.duplicated().value_counts()
#check how many rows contain NaN

print('Rows that contains NaN')

print(f.isnull().any(axis=1).value_counts())

#check how many columns contain NaN

print('Columns that contains Nan')

print(f.isnull().any(axis=0).value_counts())
f['timestamp'] = f['timestamp'].map(to_datetime)#convert to datetime objects

f['timestamp'].describe()
#get a rough view of relevence between timestamp and price_doc

from matplotlib.pyplot import *

pcm_bytime = f[['timestamp', 'price_doc']].groupby('timestamp').mean()

pcm_bytime.plot()

show()
#calculate the covariance matrix

pc_cov = f.cov()['price_doc']

pc_cov = pc_cov.copy()
#sorting the covariance matrix, and take a look

pc_cov.sort_values(ascending=False, inplace=True)

del pc_cov['price_doc']

pc_cov.head(10)
#we choose first 8 of the covariance matrix 

index_sel = ['price_doc'] + pc_cov[0:8].index.values.tolist()

fnew = f[index_sel]
fnew.head()
fnew['price_doc'].describe()
#writing the transformed data to csv file

fnew.to_csv(r'./data.csv')
import pandas as pd

import numpy as np

import tensorflow as tf



sess = tf.Session()

data = pd.read_csv(r'./data.csv')

y_vals = data['price_doc']

del data['price_doc']

del data[data.columns[0]] #discard first column



print(data.head())

print(y_vals.head())



x_vals = data.as_matrix() #convert to numpy ndarray

y_vals = y_vals.values #conver to numpy ndarray



seed = 3

tf.set_random_seed(3)

np.random.seed(seed=seed)

batch_size = 100



#split into train and test sets

train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.9), replace=False)

test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]

y_vals_train = y_vals[train_indices]

x_vals_test = x_vals[test_indices]

y_vals_test = y_vals[test_indices]



def normalize_cols(mat):

    col_max = mat.max(axis=0)

    col_min = mat.min(axis=0)

    return (mat - col_min) / (col_max - col_min)



x_vals_train = normalize_cols(x_vals_train)

x_vals_test = normalize_cols(x_vals_test)



def init_weight_uniform(shape, low, high):

    return tf.Variable(tf.random_uniform(shape=shape, minval=low, maxval=high))



def init_weight_normal(shape, st_dev):

    return tf.Variable(tf.random_normal(shape=shape, stddev=st_dev))



def init_bias(shape, st_dev, mean=0.0):

    return tf.Variable(tf.random_normal(shape=shape, stddev=st_dev))



x_data = tf.placeholder(shape=[None, 8], dtype=tf.float32)

y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)



def fully_connected_layer(input_layer, weights, bias):

    layer = tf.add(tf.matmul(input_layer, weights), bias)

    return tf.nn.relu(layer)



#building network graph

#weight_1 = init_weight_uniform(shape=[8, 40], low=-1/np.sqrt(8), high=1/np.sqrt(8))

weight_1 = init_weight_normal(shape=[8, 40], st_dev=10.0)

bias_1 = init_bias(shape=[40], st_dev=10.0)

layer_1 = fully_connected_layer(x_data, weight_1, bias_1)



#weight_2 = init_weight_uniform(shape=[40, 15], low=-1/np.sqrt(40), high=1/np.sqrt(40))

weight_2 = init_weight_normal(shape=[40, 15], st_dev=10.0)

bias_2 = init_bias(shape=[15], st_dev=10.0)

layer_2 = fully_connected_layer(layer_1, weight_2, bias_2)



#weight_3 = init_weight_uniform(shape=[15, 1], low=-1/np.sqrt(20), high=1/np.sqrt(20))

weight_3 = init_weight_normal(shape=[15, 1], st_dev=10.0)

bias_3 = init_bias(shape=[15], st_dev=10.0)

final_output = fully_connected_layer(layer_2, weight_3, bias_3)



#loss function, root mean squared log error

loss = tf.sqrt(tf.reduce_mean(tf.square(tf.log(final_output + 1) - tf.log(y_target + 1))))

my_opt = tf.train.AdamOptimizer(0.05)

train_step = my_opt.minimize(loss)



#start training

saver = tf.train.Saver()

init = tf.initialize_all_variables()

sess.run(init)



train_loss_vec = []

test_loss_vec = []

for i in range(4000):

    rand_index = np.random.choice(len(x_vals_train), size=batch_size)

    rand_x = x_vals_train[rand_index]

    rand_y = np.transpose([y_vals_train[rand_index]])

    sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})

    tmp_loss = sess.run(loss, feed_dict={x_data:rand_x, y_target:rand_y})

    train_loss_vec.append(tmp_loss)

    tmp_loss = sess.run(loss, feed_dict={x_data:x_vals_test, y_target:np.transpose([y_vals_test])})

    test_loss_vec.append(tmp_loss)

    if i%100 == 0:

        '''

        #save model

        if i%400 == 0:

            saver.save(sess, r'../output/model', global_step=i)

        '''

        print('Generation: ' + str(i) + '. Loss = ' + str(tmp_loss))





#plotting model

from matplotlib import pyplot as plt

plt.plot(train_loss_vec, 'k--', label='Train Loss')

plt.plot(test_loss_vec, 'r--', label='Test Loss')

plt.title('Loss per Generation')

plt.xlabel('Generation')

plt.ylabel('Loss')

plt.legend(loc='upper right')

plt.show()