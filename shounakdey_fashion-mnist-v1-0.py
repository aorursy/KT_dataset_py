# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#check python version
!python --version
train_data = pd.read_csv('../input/fashion-mnist_train.csv')
test_data = pd.read_csv('../input/fashion-mnist_test.csv')
print(train_data.shape,test_data.shape)
train_data.head()
# define the parameters
IMG_SIZE = 28
VAL_SIZE = 500
CHANNELS = 1
BATCH = 128
LABELS = 10
HIDDEN = 100
STEPS = 20000
LEARNING_RATE = 0.01
DECAY_FACTOR = 0.95
TRAIN = 59500
VAL = 500
KEEP_PROB = 0.5
#reshape the training data and encode the labels into one-hot encoding format
x_train = np.array(train_data.iloc[:,1:],dtype=np.float32)
y_train = np.array(train_data.iloc[:,0],dtype=np.float32)
y_train = (np.arange(10) == y_train[:,None]).astype(np.float32)
x_train = x_train.reshape(-1,IMG_SIZE,IMG_SIZE,1)
print(x_train.shape,y_train.shape)
#reshape the test data and encode the lables into one-hot encoding format
x_test = np.array(test_data.iloc[:,1:],dtype=np.float32)
y_test = np.array(test_data.iloc[:,0],dtype=np.float32)
y_test = (np.arange(10) == y_test[:,None]).astype(np.float32)
x_test = x_test.reshape(-1,IMG_SIZE,IMG_SIZE,1)
print(x_test.shape,y_test.shape)
#Split the training into folds for obtaining the validation set.
x_train,x_val = x_train[:x_train.shape[0]-VAL_SIZE],x_train[x_train.shape[0]-VAL_SIZE:]
y_train,y_val = y_train[:y_train.shape[0]-VAL_SIZE],y_train[y_train.shape[0]-VAL_SIZE:]
print(x_train.shape,x_val.shape)
tf_train_data = tf.placeholder(tf.float32,(BATCH,IMG_SIZE,IMG_SIZE,CHANNELS))
tf_test_data = tf.constant(x_test)
tf_val_data = tf.constant(x_val)
tf_train_labels = tf.placeholder(tf.float32,(BATCH,LABELS))
rate_decay = tf.Variable(1e-3)
filter1 = tf.Variable(tf.truncated_normal([3,3,1,8],stddev=0.01))
print("Filter 1:",filter1.shape)
bias1 = tf.Variable(tf.zeros([8]))
print("Bias 1:",bias1.shape)
filter2 = tf.Variable(tf.truncated_normal([3,3,8,16],stddev=0.01))
print("Filter 2:",filter2.shape)
bias2 = tf.Variable(tf.zeros([16]))
print("Bias 2:",bias2.shape)
fclayer = tf.Variable(tf.truncated_normal([7*7*16,HIDDEN],stddev=0.01))
print("Fully connected layer weight shape:",fclayer.shape)
bias3 = tf.Variable(tf.zeros([HIDDEN]))
print("FC Layer bias shape:",bias3.shape)
outputlayer = tf.Variable(tf.truncated_normal([HIDDEN,LABELS],stddev=0.01))
print("Output Layer Weights shape:",outputlayer.shape)
bias4 = tf.Variable(tf.zeros([LABELS]))
print("Output Layer Bias:",bias4.shape)
def logits(data,mode="train"):
    conv1 = tf.nn.conv2d(data,filter1,[1,1,1,1],padding="SAME")
    print("Conv1 shape: ",conv1.shape)
    pool1 = tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1],padding="SAME")
    print("Pool1 shape", pool1.shape)
    hidden1 = tf.nn.relu(pool1 + bias1)
    if mode=="train":
        hidden1 = tf.nn.dropout(hidden1,KEEP_PROB)
    print("ReLU layer 1 shape:",hidden1.shape)
    conv2 = tf.nn.conv2d(hidden1,filter2,[1,1,1,1],padding="SAME")
    print("Conv2 shape: ",conv2.shape)
    pool2 = tf.nn.max_pool(conv2,[1,2,2,1],[1,2,2,1],padding="SAME")
    print("Pool2 shape: ",pool2.shape)
    hidden2 = tf.nn.relu(pool2 + bias2)
    if mode=="train":
        hidden2 = tf.nn.dropout(hidden2,KEEP_PROB)
    print("ReLU layer 2 shape: ",hidden2.shape)
    fcinput = tf.reshape(hidden2,(-1,7*7*16))
    if mode=="train":
        fcinput = tf.nn.dropout(fcinput,KEEP_PROB)
    print("FC Layer input shape: ",fcinput.shape)
    fcoutput = tf.matmul(fcinput,fclayer)+bias3
    print("FC output shape",fcoutput.shape)
    rectifiedFcOutput = tf.nn.relu(fcoutput)
    print("ReLU FC layer shape: ",rectifiedFcOutput.shape)
    return tf.matmul(rectifiedFcOutput,outputlayer)+bias4
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits(tf_train_data),labels=tf_train_labels))
#defining an optimizer 
learning_rate = tf.train.exponential_decay(LEARNING_RATE,rate_decay,1000,DECAY_FACTOR,staircase=False)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=rate_decay)
predict_train = tf.nn.softmax(logits(tf_train_data,mode="train"))
predict_val = tf.nn.softmax(logits(tf_val_data,mode="val"))
predict_test = tf.nn.softmax(logits(tf_test_data,mode="predict"))
session = tf.Session()
tf.global_variables_initializer().run(session=session)
def acc(pred,labels):
    return 100.0 *  np.mean(np.float32(np.argmax(pred, axis=1) == np.argmax(labels, axis=1)), axis=0)
_step = 0
for step in np.arange(STEPS):
    _step += 1
    start = (step*BATCH)%(TRAIN-BATCH)
    stop = start + BATCH
    batch_data = x_train[start:stop]
    batch_labels = y_train[start:stop]
    feed_dict = {tf_train_data:batch_data,tf_train_labels:batch_labels}
    opt,batch_loss,batch_prediction = session.run([optimizer,loss,predict_train],feed_dict=feed_dict)
    if (step % 100 == 0):
        b_acc = acc(batch_prediction, batch_labels)
        v_acc = acc(predict_val.eval(session=session), y_val)
        print('Step %i'%step, end='\t')
        print('Loss = %.2f'%batch_loss, end='\t')
        print('Batch Acc. = %.1f'%b_acc, end='\t\t')
        print('Valid. Acc. = %.1f'%v_acc, end='\n')
#Accuracy of the test data.
print(acc(predict_test.eval(session=session),y_test))