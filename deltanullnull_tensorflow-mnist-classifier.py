# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
print(test_data.columns)
label_col = ['label']

train_x = train_data.drop(columns=label_col)
train_y = train_data[label_col]

test_x = test_data[:]
from sklearn import model_selection

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_x, train_y)
print(train_x.shape)
print(train_y.shape)

print('Valid X', valid_x.shape)
print('Valid Y', valid_y.shape)
import numpy as np
train_x_np = np.reshape(np.array(train_x), [len(train_x), 28 * 28])
valid_x_np = np.reshape(np.array(valid_x), [len(valid_x), 28 * 28])

test_x_np = np.array(test_x).reshape([len(test_x), 28 * 28])
print(train_x_np.shape)
print(valid_x_np.shape)

print(test_x_np.shape)
n_classes = 10

train_y_np = np.zeros([len(train_y), n_classes])
valid_y_np = np.zeros([len(valid_y), n_classes])

t_y = np.array(train_y).reshape([len(train_y)])
v_y = np.array(valid_y).reshape([len(valid_y)])

#print(train_y)
for i in range(len(train_y)):
    idx = t_y[i]
    train_y_np[i,idx] = 1
    

#print(train_y)
for i in range(len(valid_y)):
    idx = v_y[i]
    valid_y_np[i,idx] = 1
print(train_y_np.shape)
print(valid_y_np.shape)
import tensorflow as tf
learning_rate = 0.001
dropout_rate = 0.8
def extract_batch(x, y, batch_size, curr):
    batch_x = np.zeros([batch_size, x.shape[1]])
    batch_y = np.zeros([batch_size, y.shape[1]])
    
    for i in range(batch_size):
        batch_x[i] = x[curr]
        batch_y[i] = y[curr]
        curr += 1
        if (curr >= x.shape[0]):
            curr = 0
            
    return batch_x, batch_y, curr
def create_model(x, y):
    x_resh = tf.reshape(x, shape=[-1, 28, 28, 1])
    
    w_0 = tf.Variable(tf.random_normal([3, 3, 1, 16]))
    b_0 = tf.Variable(tf.random_normal([16]))
    
    conv_0 = tf.nn.conv2d(input=x_resh, filter=w_0, strides=[1, 1, 1, 1], padding='SAME')
    conv_0 = tf.nn.bias_add(conv_0, b_0)
    
    pool_0 = tf.nn.max_pool(value=conv_0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    w_1 = tf.Variable(tf.random_normal([3, 3, 16, 32]))
    b_1 = tf.Variable(tf.random_normal([32]))
    
    conv_1 = tf.nn.conv2d(input=pool_0, filter=w_1, strides=[1, 1, 1, 1], padding='SAME')
    conv_1 = tf.nn.bias_add(conv_1, b_1)
    
    pool_1 = tf.nn.max_pool(value=conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    drop_1 = tf.nn.dropout(pool_1, dropout_rate)
    
    w_2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
    b_2 = tf.Variable(tf.random_normal([64]))
    
    conv_2 = tf.nn.conv2d(input=drop_1, filter=w_2, strides=[1, 1, 1, 1], padding='SAME')
    conv_2 = tf.nn.bias_add(conv_2, b_2)
    
    drop_2 = tf.nn.dropout(conv_2, dropout_rate)
    #pool_2 = tf.nn.max_pool(value=conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    w_fc = tf.Variable(tf.random_normal([7 * 7 * 64, 1024]))
    b_fc = tf.Variable(tf.random_normal([1024]))
    
    flat_0 = tf.reshape(drop_2, [-1, 7 * 7 * 64])
    
    fc_0 = tf.nn.xw_plus_b(flat_0, w_fc, b_fc)
    fc_0 = tf.nn.relu(fc_0)
    
    w_out = tf.Variable(tf.random_normal([1024, n_classes]), name='w_out')
    b_out = tf.Variable(tf.random_normal([n_classes]), name='b_out')
    
    out = tf.nn.xw_plus_b(fc_0, w_out, b_out)
    return out
x = tf.placeholder(shape=[None, 28 * 28], dtype=tf.float32)
y = tf.placeholder(shape=[None, 10], dtype=tf.float32)

logits = create_model(x, y)

preds = tf.nn.softmax(logits=logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
opt_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(x=tf.argmax(preds, 1), y=tf.argmax(y, 1))
acc_op = tf.reduce_mean(tf.cast(x=correct_pred, dtype=tf.float32))

init = tf.global_variables_initializer()
n_steps = 6000
batch_size = 200

sess = tf.Session()
#with tf.Session() as sess:
sess.run(init)
curr = 0
for step in range(n_steps):
    # TODO extract batch

    train_x_batch, train_y_batch, curr = extract_batch(x=train_x_np, y=train_y_np, batch_size=batch_size, curr = curr)
    feed_dict = {x: train_x_batch, y: train_y_batch}

    sess.run(opt_op, feed_dict=feed_dict)

    if (step % 10 == 0):
        loss, acc = sess.run([loss_op, acc_op], feed_dict=feed_dict)

        #print(loss)
        #print(acc)

        print("Step: %.2d" % step, " Loss: ", "{:.2f}".format(loss), " Accuracy: ", "{:.2f}".format(acc))

    # validation

print("Optimization finished")

loss, acc = sess.run([loss_op, acc_op], feed_dict={x: valid_x_np, y: valid_y_np})

print("Validation Loss: ", "{:.2f}".format(loss), " Accuracy: ", "{:.2f}".format(acc))
predictions = sess.run(preds, feed_dict={x: test_x})
result = np.zeros([len(predictions)], dtype=np.int64)

for idx, pred in enumerate(predictions):
    # pred = 1d array
    result[idx] = np.argmax(pred)
df = pd.DataFrame(data={'ImageId': range(1,len(predictions)+1), 'Label': result})
df.to_csv('result.csv', index=False)