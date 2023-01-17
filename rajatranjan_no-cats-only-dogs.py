# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline
import random
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# print(os.listdir("../input/cat-vs-dogs-arrays"))
TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

ROWS = 64
COLS = 64
CHANNELS = 3


train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]

train_images = train_dogs[:3000] + train_cats[:3000]
random.shuffle(train_images)
#test_images =  test_images[:1000]
from PIL import ImageFilter
from sklearn import preprocessing
def read_image(file_path):
    img = Image.open(file_path)
    img=img.resize((ROWS, COLS), Image.ANTIALIAS)
    #img = img.filter(ImageFilter.BLUR)
    #img = img.filter(ImageFilter.FIND_EDGES)
    
    return np.array(img)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS,CHANNELS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
        if i%1000 == 0:
            print('Processed {} of {}'.format(i, count))
            #print(image.shape)
            #plt.imshow(image)
            #plt.show()
    #print(data.shape)
    return data

train = prep_data(train_images)
test = prep_data(test_images)

# train=np.load('../input/cat-vs-dogs-arrays/train.npz')
# train.shape
import seaborn as sns
from matplotlib import ticker
train_labels = []
for i in train_images:
    if 'dog' in i:
        train_labels.append([1,0])
    else:
        train_labels.append([0,1])
train_labels=np.array(train_labels)
#sns.countplot(train_labels)
#train_labels=np.array(train_labels)
train_labels.shape
print(train)
train=train/train.max()
print(train)
test=test/test.max()
for i in range(0,len(train_images),500):
    print(train_images[i],train_labels[i])
from sklearn.model_selection import train_test_split
test_size = 0.25
X_train, X_test, Y_train, Y_test = train_test_split(train,train_labels, test_size=test_size, random_state=101)

img_size = 64
channel_size = 1
print("Training Size:", X_train.shape)
print(X_train.shape[0],"samples - ", X_train.shape[1],"x",X_train.shape[2],"rgb image")

print("\n")

print("Test Size:",X_test.shape)
print(X_test.shape[0],"samples - ", X_test.shape[1],"x",X_test.shape[2],"rgb image")
class SignClass():
    
    def __init__(self):
        self.i = 0
        
        self.training_images = X_train
        self.training_labels = Y_train
        
        self.test_images = X_test
        self.test_labels = Y_test
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
        self._num_examples = X_train.shape[0]
    


        
    def next_batch(self, batch_size,fake_data=False, shuffle=True):
#         x = self.training_images[self.i:self.i+batch_size].reshape(-1,64,64,1)
#         y = self.training_labels[self.i:self.i+batch_size]
#         self.i = (self.i + batch_size) % len(self.training_images)
#         return x, y
        x = self.training_images[self.i:self.i+batch_size]
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y
        """if fake_data:
            fake_image = [1] * 4096
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self.training_images = self.training_images[perm0]
            self.training_labels = self.training_labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
        # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self.training_images[start:self._num_examples]
            labels_rest_part = self.training_labels[start:self._num_examples]
          # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self.training_images = self.training_images[perm]
                self.training_labels = self.training_labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self.training_images[start:end]
            labels_new_part = self.training_labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.training_images[start:end], self.training_labels[start:end]"""
ch = SignClass()
import tensorflow as tf
x = tf.placeholder(tf.float32,shape=[None,64,64,3])
y_true = tf.placeholder(tf.float32,shape=[None,2])
hold_prob = tf.placeholder(tf.float32)
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape):
    #with tf.name_scope(name):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    c=conv2d(input_x, W)
    act=tf.nn.relu(c + b)
    #tf.summary.histogram("weights",W)
    #tf.summary.histogram("biases",b)
    #tf.summary.histogram("activations",act)
    return act

def normal_full_layer(input_layer, size):
    #with tf.name_scope(name):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b
convo_1 = convolutional_layer(x,shape=[5,5,3,64])
convo_1_pooling = max_pool_2by2(convo_1)
convo_2 = convolutional_layer(convo_1_pooling,shape=[5,5,64,128])
convo_2_pooling = max_pool_2by2(convo_2)

convo_2_flat = tf.reshape(convo_2_pooling,[-1,16*16*128])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
y_pred = normal_full_layer(full_one_dropout,2)
# with tf.name_scope("crossentropy"):
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
#     tf.summary.scalar('cross_entropy', cross_entropy)
# with tf.name_scope("optimizer"):
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
trainop = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()
steps = 1000
saver = tf.train.Saver()
with tf.Session() as sess:
    
    sess.run(init)
    #merged_Summary=tf.summary.merge_all()
    #writer=tf.summary.FileWriter('dir3/')
    #writer.add_graph(sess.graph)
    for i in range(steps):
        batch = ch.next_batch(100)
        
        sess.run(trainop, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:
            #s=sess.run(merged_Summary,feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})
            #writer.add_summary(s,i)
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            #with tf.name_scope("accuracy"):
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            #tf.summary.scalar('accuracy', acc)
            print(sess.run(acc,feed_dict={x:ch.test_images,y_true:ch.test_labels,hold_prob:1.0}))
            print('\n')
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)
#import tensorflow as tf
predictions=[]
with tf.Session() as sess:
  # Restore variables from disk.
    saver.restore(sess, "/tmp/model.ckpt")
    print('predicting')
    #predicta=tf.argmax(y_pred,1)
    predicts=tf.nn.softmax(y_pred)
    #print(1)
#     print(sess.run(tf.Print(y_pred,[y_pred])))
#     print(sess.run(tf.Print(predicta,[predicta])))
#     print(sess.run(tf.Print(predicts,[predicts])))
    
    predictions1=sess.run(predicts,feed_dict={x:test[0:1500],hold_prob:1.0})
    predictions2=sess.run(predicts,feed_dict={x:test[1500:3000],hold_prob:1.0})
    predictions3=sess.run(predicts,feed_dict={x:test[3000:4500],hold_prob:1.0})
    predictions4=sess.run(predicts,feed_dict={x:test[4500:6000],hold_prob:1.0})
    predictions5=sess.run(predicts,feed_dict={x:test[6000:7500],hold_prob:1.0})
    predictions6=sess.run(predicts,feed_dict={x:test[7500:9000],hold_prob:1.0})
    predictions7=sess.run(predicts,feed_dict={x:test[9000:11500],hold_prob:1.0})
    predictions8=sess.run(predicts,feed_dict={x:test[11500:],hold_prob:1.0})

# for j in [predictions1,predictions2,predictions3,predictions4,predictions5,predictions6]:
    
#     predictions.append(j)
#predictions=predictions1+predictions2+predictions3+predictions4+predictions5+predictions6
# for j in predictions1:
#     predictions.append([format(float(x), '.16f') for x in j])
# for j in predictions2:
#     predictions.append([format(float(x), '.16f') for x in j])
# for j in predictions3:
#     predictions.append([format(float(x), '.16f') for x in j])
# for j in predictions4:
#     predictions.append([format(float(x), '.16f') for x in j])
# for j in predictions5:
#     predictions.append([format(float(x), '.16f') for x in j])
# for j in predictions6:
#     predictions.append([format(float(x), '.16f') for x in j])

for j in predictions1:
    predictions.append(j)
for j in predictions2:
    predictions.append(j)
for j in predictions3:
    predictions.append(j)
for j in predictions4:
    predictions.append(j)
for j in predictions5:
    predictions.append(j)
for j in predictions6:
    predictions.append(j)
for j in predictions7:
    predictions.append(j)
for j in predictions8:
    predictions.append(j)
predictions=np.array(predictions)
print('done >',predictions.shape)
pr=pd.DataFrame(data=predictions,columns=['label','cat'])

# for i in range(1,len(predictions)+1):
    
prl=pd.DataFrame(data=pr['label'],columns=['label'])
prl.head()
d=np.array(list(range(1,len(predictions)+1)))
ids=pd.DataFrame(data=d,columns=['id'])
ids
mmm=pd.concat([ids,prl],axis=1)
mmm.to_csv('s1.csv',index=False)
mmm
