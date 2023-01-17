# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import PIL.Image



daisy_path = "/kaggle/input/AIAxTainanDL/flower/flower_classification/train/daisy/"

dandelion_path = "/kaggle/input/AIAxTainanDL/flower/flower_classification/train/dandelion/"

rose_path = "/kaggle/input/AIAxTainanDL/flower/flower_classification/train/rose/"

sunflower_path = "/kaggle/input/AIAxTainanDL/flower/flower_classification/train/sunflower/"

tulip_path = "/kaggle/input/AIAxTainanDL/flower/flower_classification/train/tulip/"

test_path="/kaggle/input/AIAxTainanDL/flower/flower_classification/test/"

submission = pd.read_csv('/kaggle/input/AIAxTainanDL/submission.csv')
from os import listdir

import cv2







img_data = []

labels = []



size = 64,64

def iter_images(images,directory,size,label):

    try:

        for i in range(len(images)):

            img = cv2.imread(directory + images[i])

            img = cv2.resize(img,size)

            img_data.append(img)

            labels.append(label)

    except:

        pass



iter_images(listdir(daisy_path),daisy_path,size,0)

iter_images(listdir(dandelion_path),dandelion_path,size,1)

iter_images(listdir(rose_path),rose_path,size,2)

iter_images(listdir(sunflower_path),sunflower_path,size,3)

iter_images(listdir(tulip_path),tulip_path,size,4)
len(img_data),len(labels)
test_data = []



size = 64,64

def test_images(images,directory,size):

    try:

        for i in range(len(images)):

            img = cv2.imread(directory + submission['id'][i]+".jpg")

            img = cv2.resize(img,size)

            test_data.append(img)

    except:

        pass





test_images(listdir(test_path),test_path,size)
len(test_data)
import numpy as np

data = np.asarray(img_data).reshape(len(img_data), 64*64*3)

testData=np.asarray(test_data).reshape(len(test_data), 64*64*3)



#div by 255

data = data / 255.0

testData=testData/255.0



labels = np.asarray(labels)

data.shape,labels.shape
from sklearn.model_selection import train_test_split



# Split the data

X_train, X_validation, Y_train, Y_validation = train_test_split(data, labels, test_size=0.02, shuffle= True)

print("Length of X_train:", len(X_train), "Length of Y_train:", len(Y_train))

print("Length of X_validation:",len(X_validation), "Length of Y_validation:", len(Y_validation))
from keras.utils import np_utils

Y_train_one_hot = np_utils.to_categorical(Y_train, 5)

Y_validation_one_hot = np_utils.to_categorical(Y_validation, 5)
# TensorFlow

import tensorflow as tf

from tqdm import tqdm

from sklearn.utils import shuffle



tf.reset_default_graph()



with tf.name_scope('placeholder'):

    input_data = tf.placeholder(tf.float32, shape=[None, 64*64*3], name='X')

    y_true = tf.placeholder(tf.float32, shape=[None, 5], name='y')

    

with tf.variable_scope('network'):

    h1 = tf.layers.dense(input_data, 256, activation=tf.nn.sigmoid, name='hidden1')  

    h2 = tf.layers.dense(h1, 128, activation=tf.nn.relu, name='hidden2') 

    h3 = tf.layers.dense(h2, 64, activation=tf.nn.relu, name='hidden3')

    out = tf.layers.dense(h3, 5, name='output')

    

with tf.name_scope('loss'):

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=out), name='loss')

    

with tf.name_scope('accuracy'):

    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(out), 1), tf.argmax(y_true, 1))

    compute_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    

with tf.name_scope('opt'):

    update = tf.train.GradientDescentOptimizer(learning_rate=0.4).minimize(loss) 



init = tf.global_variables_initializer()
tf.global_variables() # 查看所有tf變數
# 計算總共有幾組weight

var_sum = 0

for tensor in tf.global_variables(scope='network'):

    if 'Adam' not in tensor.name:

        var_sum += np.product(tensor.shape.as_list())

    

print('the total number of weights is', var_sum) 
bs = 64



train_loss_epoch, train_acc_epoch = [], []

test_loss_epoch, test_acc_epoch = [], []



sess = tf.Session()

sess.run(init)



for i in tqdm(range(101)):

    

#     training part

    train_loss_batch, train_acc_batch = [], []

    

    total_batch = len(X_train) // bs

    

    for j in range(total_batch):

        

        X_batch = X_train[j*bs : (j+1)*bs]

        y_batch = Y_train_one_hot[j*bs : (j+1)*bs]

        batch_loss, batch_acc, _ = sess.run([loss, compute_acc, update], 

                                            feed_dict={input_data: X_batch, y_true: y_batch})

        

        train_loss_batch.append(batch_loss)

        train_acc_batch.append(batch_acc)

        

    train_loss_epoch.append(np.mean(train_loss_batch))

    train_acc_epoch.append(np.mean(train_acc_batch))

    

#     testing part

    batch_loss, batch_acc = sess.run([loss, compute_acc], 

                                    feed_dict={input_data: X_validation, y_true: Y_validation_one_hot})



    test_loss_epoch.append(batch_loss)

    test_acc_epoch.append(batch_acc)

    

    X_train, Y_train_one_hot = shuffle(X_train, Y_train_one_hot)

    

    if i%10 == 0:

        print('step: {:2d}, train loss: {:.3f}, train acc: {:.3f}, test loss: {:.3f}, test acc: {:.3f}'

             .format(i, train_loss_epoch[i], train_acc_epoch[i], test_loss_epoch[i], test_acc_epoch[i]))
import matplotlib.pyplot as plt

plt.plot(train_loss_epoch, 'b', label='train')

plt.plot(test_loss_epoch, 'r', label='test')

plt.legend()

plt.title("Loss")

plt.show()



plt.plot(train_acc_epoch, 'b', label='train')

plt.plot(test_acc_epoch, 'r', label='test')

plt.legend()

plt.title("Accuracy")

plt.show()
pred =  np.argmax(sess.run(out,feed_dict={input_data: testData}), axis=1)

newsSbmission=submission

newsSbmission["class"]=pred

newsSbmission.to_csv("submission.csv", index=False)
pred