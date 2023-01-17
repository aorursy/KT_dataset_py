# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import StratifiedShuffleSplit

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
#load test data
with open("../input/test.csv","rb") as f:
    temp_data = pd.read_csv(f)
#test_labels = np.asarray(temp_data["label"])
test_data = np.asarray(temp_data)

#load train data and labels
with open("../input/train.csv","rb") as f:
    temp_data = pd.read_csv(f)
train_labels = np.asarray(temp_data["label"])
train_data = np.asarray(temp_data.drop(columns="label"))

#val_count = np.floor_divide(30*len(train_labels),100)
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(train_data,train_labels):
    strat_train_data, strat_val_data = train_data[train_index], train_data[test_index]
    strat_train_labels, strat_val_labels = train_labels[train_index], train_labels[test_index]
#define image dimensions which is 28x28x1 for MNIST dataset
height = 28
width = 28
channel = 1
proj_dir = "../working/character_classification_akj/"
checkpoint_path = proj_dir + "intermediate_checkpoint.ckpt"
checkpoint_epoch_path = checkpoint_path+".epoch"
final_model_path = proj_dir + "my_model"
def fetch_batch(feature_set, labels, batch_size):
    p = np.random.permutation(len(feature_set))
    return (feature_set[p][:batch_size,:], labels[p][:batch_size])
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None,height*width*channel], name="X")
y = tf.placeholder(tf.int32, [None], name="y")
training = tf.placeholder_with_default(False,shape=(), name = "training_variable")
learning_rate = tf.placeholder_with_default(0.01,shape=(), name= "learning_rate")
global_step_tensor = tf.Variable(0,trainable=False, name= "global_step")


input_layer = tf.reshape(X, [-1,height,width,channel], name="Input_layer")

conv1 = tf.layers.conv2d(inputs= input_layer,
                         filters= 32,
                         kernel_size= [5,5],
                         padding= "same",
                         activation= tf.nn.relu,
                         name= "conv_1")

pool1 = tf.layers.max_pooling2d(inputs= conv1,
                                pool_size= [2,2],
                                strides= 2,
                                name= "pool_1")

conv2 = tf.layers.conv2d(inputs= pool1,
                         filters= 64,
                         kernel_size= [5,5],
                         padding= "same",
                         activation= tf.nn.relu,
                         name= "conv_2")
    
pool2 = tf.layers.max_pooling2d(inputs= conv2,
                                pool_size= [2,2],
                                strides= 2,
                                name= "pool_2")
    
pool2_flat = tf.reshape(pool2, [-1,7*7*64])
    
dense = tf.layers.dense(inputs= pool2_flat,
                        units= 1024,
                        activation= tf.nn.relu)

dropout = tf.layers.dropout(dense,
                            rate=0.3,
                            training=training,
                            name="Dropout")

logits = tf.layers.dense(inputs= dropout, units= 10)
#classes = tf.argmax(logits,axis=1)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=y,
                                                          name="xentropy")
loss = tf.reduce_mean(xentropy,
                      name="loss")

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss,global_step=global_step_tensor)

correct = tf.nn.in_top_k(logits, y,1)
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

learning_rate_array = 0.0001*np.exp(-20000*np.linspace(0.000001,0.001,10))
train_loss_list = []
train_accuracy_list = []
val_loss_list = []
val_accuracy_list = []
old_loss = np.inf
new_loss = 0
early_stop_count = 0

n_epoch = 10
n_iter = 1000
batch_size = 200
with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        with open(checkpoint_epoch_path) as f:
            start_epoch = int(f.read())
        print("Training was interupted resuming from epoch ", start_epoch)
        saver.restore(sess,checkpoint_path)
    else:
        start_epoch = 0
        init.run()
        os.mkdir("../working/character_classification_akj")
    for epoch in range(start_epoch,n_epoch):
        learning_rate_epoch = learning_rate_array[epoch]
        for iteration in range(n_iter):
            X_batch, y_batch = fetch_batch(strat_train_data, strat_train_labels, batch_size)
            sess.run(training_op, feed_dict={X:X_batch,
                                             y:y_batch,
                                             training:True,
                                             learning_rate:learning_rate_epoch})
            new_loss = sess.run(loss,feed_dict={X:strat_val_data,y:strat_val_labels})
            if new_loss<old_loss:
                saver.save(sess,final_model_path)
                old_loss = new_loss
                best_epoch = epoch
                best_iteration = iteration
                early_stop_count=0
            else:
                early_stop_count+=1
                if early_stop_count>500:
                    print("Early stopping satisfied with best epoch ",best_epoch,
                          " and best iteration ",iteration)
                    current_train_loss, current_train_accu = sess.run([loss,accuracy],
                                                                      feed_dict={X:strat_train_data,
                                                                                y:strat_train_labels})
                    current_val_loss, current_val_accu = sess.run([loss,accuracy],
                                                                      feed_dict={X:strat_val_data,
                                                                                y:strat_val_labels})
                    print("Current train loss: ",current_train_loss,
                          " Current train accuracy: ", current_train_accu)
                    print()
                    print("Current val loss: ",current_val_loss,
                          " Current val accuracy: ", current_val_accu)
                    print()
                    print("Best model loss: ", old_loss)
                    break
        if early_stop_count>50:
            break
        if epoch%1==0:
            print("Saving checkpoint for epoch ",epoch)
            saver.save(sess,checkpoint_path)
            with open(checkpoint_epoch_path,"wb") as f:
                f.write(b'%d' % (epoch+1))
            train_loss, train_accuracy = sess.run([loss, accuracy],
                                                    feed_dict={X:X_batch, y:y_batch})
            train_loss_list.append(train_loss)
            train_accuracy_list.append(train_accuracy)
            print("Epoch: ",epoch)    
            print("Training loss: ",train_loss," Training accuracy: ",train_accuracy)
            print()
    #saver.save(sess,final_model_path)
with tf.Session() as sess:
    saver.restore(sess,final_model_path)
    predict = sess.run(logits,feed_dict={X:test_data})
    predict_class = sess.run(tf.argmax(predict,axis=1))
output_file = "submission.csv"
with open(output_file, 'w') as f :
    f.write('ImageId,Label\n')
    for i in range(len(predict_class)) :
        f.write("".join([str(i+1),',',str(predict_class[i]),'\n']))
os.listdir("./")
plt.plot(0.01*np.exp(-20000*np.linspace(0.000001,0.0001,10)))
0.0001*np.exp(-20000*np.linspace(0.000001,0.001,10))
