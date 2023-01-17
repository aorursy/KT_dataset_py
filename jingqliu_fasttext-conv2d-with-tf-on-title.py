import pandas as pd
import numpy as np
import re
import csv
import tensorflow as tf
import nltk
import gc
from gensim.models import Word2Vec
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from collections import Counter
import math
df_train = pd.read_csv('../input/avito-demand-prediction/train.csv') 
train_input = df_train['title']
y_train = df_train['deal_probability']
df_test = pd.read_csv('../input/avito-demand-prediction/test.csv')
test_input = df_test['title']
del df_train
del df_test
gc.collect()
def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open('../input/fasttext-russian-2m/wiki.ru.vec'))
len(embeddings_index) 
del embeddings_index['1888423']
def clean(string):
    string = re.sub(r'\n', ' ', string)
    string = re.sub(r'\t', ' ', string)
    string = re.sub('[\W]', ' ', string)
    string = re.sub('[0-9]', ' ', string)
    string = re.sub(r'\s{2,}', ' ', string.lower())
    return string
x_train = train_input.apply(clean)
x_test = test_input.apply(clean)
x_train = x_train.fillna('fillna')
x_test = x_test.fillna('fillna')
lst = []
for line in x_train:
    lst += line.split()
    
count = Counter(lst)
for k in list(count.keys()):
    if k not in embeddings_index:
        del count[k]
len(count)
count = dict(sorted(count.items(), key=lambda x: -x[1]))
count = {k:v for (k,v) in count.items() if v >= 3}
len(count)
count = dict(zip(list(count.keys()),range(1,43547 + 1)))
embedding_matrix = {}
for key in count:
    embedding_matrix[key] = embeddings_index[key]
lst = []
for line in x_test:
    lst += line.split()
    
count_test = Counter(lst)
for k in list(count_test.keys()):
    if k not in embedding_matrix:
        del count_test[k]
    else:
        count_test[k] = count[k]
len(count_test)
W = np.zeros((1,300))
W = np.append(W, np.array(list(embedding_matrix.values())),axis=0)
W = W.astype(np.float32, copy=False)
del lst
gc.collect()
for i in range(len(x_train)):
    temp = x_train[i].split()
    for word in temp[:]:
        if word not in count:
            temp.remove(word)
    for j in range(len(temp)):
        temp[j] = count[temp[j]]
    x_train[i] = temp
    
for i in range(len(x_test)):
    temp = x_test[i].split()
    for word in temp[:]:
        if word not in count_test:
            temp.remove(word)
    for j in range(len(temp)):
        temp[j] = count_test[temp[j]]
    x_test[i] = temp
x_train = sequence.pad_sequences(list(x_train), maxlen = 15)
x_test = sequence.pad_sequences(list(x_test), maxlen = 15)
filter_sizes = [1,2,3,4,5]
num_filters = 32
batch_size = 256
num_filters_total = num_filters * len(filter_sizes)
embedding_size = 300
sequence_length = 15
num_epochs = 3
dropout_keep_prob = 0.8
input_x = tf.placeholder(tf.int32, [None, sequence_length], name = "input_x")
input_y = tf.placeholder(tf.float32, [None,2], name = "input_y")
embedded_chars = tf.nn.embedding_lookup(W, input_x)
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
def CNN(data):
    pooled_outputs = []
    
    for i, filter_size in enumerate(filter_sizes):
        
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        
        w = tf.Variable(tf.truncated_normal(filter_shape,stddev = 0.05), name = "w")
        b = tf.Variable(tf.truncated_normal([num_filters], stddev = 0.05), name = "b")
            
        conv = tf.nn.conv2d(
            data,
            w,
            strides = [1,1,1,1],
            padding = "VALID",
            name = "conv"
        )
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name = "relu")
        pooled = tf.nn.max_pool(
            h,
            ksize = [1, sequence_length - filter_size + 1, 1, 1],
            strides = [1,1,1,1],
            padding = "VALID",
            name = "pool"
        )
        
        pooled_outputs.append(pooled)
    
    #return pooled_outputs
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    return h_pool_flat
h_pool_flat = CNN(embedded_chars_expanded)
h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
wd1 = tf.Variable(tf.truncated_normal([num_filters_total, int(num_filters_total/2)], stddev=0.05), name = "wd1")
bd1 = tf.Variable(tf.truncated_normal([int(num_filters_total/2)], stddev = 0.05), name = "bd1")
layer1 = tf.nn.xw_plus_b(h_drop, wd1, bd1, name = 'layer1')
layer1 = tf.nn.relu(layer1)
wd2 = tf.Variable(tf.truncated_normal([int(num_filters_total/2),2], stddev = 0.05), name = 'wd2')
bd2 = tf.Variable(tf.truncated_normal([2], stddev = 0.05), name = "bd2")
layer2 = tf.nn.xw_plus_b(layer1, wd2, bd2, name = 'layer2') 
prediction = tf.nn.softmax(layer2)
rmse = tf.reduce_mean(tf.losses.mean_squared_error(predictions = prediction, labels = input_y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0007).minimize(rmse)
def generate_batch(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    l = 0
    for epoch in range(num_epochs):
        l += 1
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
def blocks(data, block_size):
    data = np.array(data)
    data_size = len(data)
    nums = int((data_size-1)/block_size) + 1
    for block_num in range(nums):
        if block_num == 0:
            print("prediction start!")
        start_index = block_num * block_size
        end_index = min((block_num + 1) * block_size, data_size)
        yield data[start_index:end_index]
batch1 = generate_batch(list(zip(np.array(x_train), y_train)), batch_size, 1)
batch2 = generate_batch(list(zip(np.array(x_train), y_train)), batch_size, 1)
batch3 = generate_batch(list(zip(np.array(x_train), y_train)), batch_size, 1)
batch_bag = [batch1,batch2,batch3]
test_blocks = blocks(list(np.array(x_test)), 2000)
int((len(x_train)-1)/256) + 1
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init_op)
    i = 0
    for batches in batch_bag:
        i += 1
        print('Epoch: ' + str(i) + ' start!')
        avg_loss = 0
        avg_rmse = 0
        for batch in batches:
            batch = pd.DataFrame(batch, columns = ['a','1'])
            x_batch = pd.DataFrame(list(batch['a']))
            y_batch = batch.loc[:, batch.columns != 'a']
            y_batch['0'] = 1 - y_batch['1']
            _,m = sess.run([optimizer, rmse],feed_dict = {input_x: x_batch, input_y: y_batch})
            avg_rmse += m
        avg_rmse = math.sqrt(avg_rmse/5873)
        print('Epoch:' + str(i) + ' Rmse is ' + str(avg_rmse))
        
    print('Prediction Start!')
    
    df = pd.DataFrame()
    for block in test_blocks:
        block = pd.DataFrame(block)
        pred = sess.run(prediction, feed_dict = {input_x: block})
        df = df.append(pd.DataFrame(pred))
    
    print('Finish!') 
df.round().mean()
df.columns = ['0','1']
submission = pd.read_csv("../input/avito-demand-prediction/sample_submission.csv")
submission['deal_probability'] = np.array(df['0'])
submission.to_csv("submission.csv", index=False)