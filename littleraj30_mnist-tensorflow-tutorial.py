import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
# Importing CSV files as this step you all knows  

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
# Checking dataframe shape

train.shape , test.shape
train.head()
# Extracting label from train dataframe

train_label=train.iloc[:,0]
#converting dataframe to category for hot encode

train_label=train_label.astype('category')
# Converted into Hot encode

train_label=pd.get_dummies(train_label)

train_label.shape
del train['label']
# importing Tensorflow

import tensorflow as tf
def variable(x,weight_shape,bias_shape):

    weight_init=tf.truncated_normal_initializer(stddev=0.1)

    bias_init=tf.constant_initializer(0.1)

    weight=tf.get_variable(shape=weight_shape,name='weight',initializer=weight_init)

    bias=tf.get_variable(shape=bias_shape,name='bias',initializer=bias_init)

    output= tf.add(tf.matmul(x,weight),bias)

    return output
x=tf.placeholder(tf.float32,name='x',shape=[None,784])

y=tf.placeholder(tf.float32,name='y',shape=[None,10])

drop=tf.placeholder(tf.float32)
with tf.variable_scope('layer_1'):

    hidden_1=variable(x,[784,512],[512])

with tf.variable_scope('layer_2'):

    hidden_2=variable(hidden_1,[512,256],[256])

with tf.variable_scope('layer_3'):

    hidden_3=variable(hidden_2,[256,128],[128])

    out1=tf.nn.dropout(hidden_3,drop)   # To prevent from Overfitting

with tf.variable_scope('outputlayer'):

    output=variable(out1,[128,10],[10])
# Defining cost function which will be used  by gradient descent

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output))   
# Graident Descent for minimize cross entropy

optimize=tf.train.AdamOptimizer(learning_rate=0.001)

step=optimize.minimize(cross_entropy)
correct_pred=tf.equal(tf.argmax(output,1),tf.argmax(y,1))

accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
# Initialization of all variables

init=tf.initialize_all_variables()
train.shape, test.shape
# Creating Session for all computation 

sess=tf.Session()
#Initialize variables

sess.run(init)
train=train.values

train_label=train_label.values
# Some useful parameters for minibatch creation and gradient descent optimization 

iteration=2000

batch_size=256
for i in range(iteration):

    choice=np.random.choice(42000,size=batch_size)

    x_train=train[choice]

    y_train=train_label[choice]

    sess.run(step,feed_dict={x:x_train,y:y_train,drop:0.4})



    if (i%100==0):

        loss,accu=sess.run([cross_entropy,accuracy],feed_dict={x:x_train,y:y_train,drop:1})

        print ('loss is',str(loss),'accuracy is',str(accu),'iteration',i+1)
# Test dataset shape

test.shape
# Convert to arrays

test_array=test.values
result=sess.run(output,feed_dict={x:test_array,drop:1})
result=np.argmax(result,axis=1)
final=pd.DataFrame({'Predicted':result})

final.head()
import matplotlib.pyplot as plt
arr=test_array[0:5]

i=0

plt.imshow(arr[i].reshape([28,28]))

plt.title(final.iloc[0,0],size=20)
arr=test_array[0:5]

i=1

plt.imshow(arr[i].reshape([28,28]))

plt.title(final.iloc[1,0],size=20)
arr=test_array[0:5]

i=2

plt.imshow(arr[i].reshape([28,28]))

plt.title(final.iloc[2,0],size=20)