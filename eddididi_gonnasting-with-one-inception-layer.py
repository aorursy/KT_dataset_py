# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        pass

# Any results you write to the current directory are saved as output.
metaTrain=pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")

baseDir='/kaggle/input/aerial-cactus-identification/train/train/'

metaTrain.head()
fileNames=[str(baseDir+x) for x in metaTrain['id'].tolist()]

fileLabels=metaTrain['has_cactus'].tolist()

print('There are '+str(len(fileNames)) + ' pictures.')
temp=pd.get_dummies(fileLabels)

fileLabels=[]

for x,y in zip(temp[0].tolist(),temp[1].tolist()):

    fileLabels.append([x,y])
import PIL

import matplotlib.pyplot as plt



from PIL import Image



def makeTrainingBatch(batchIdx,batchSize):

    trainImgs=[]

    trainLabels=[]

    for i in range(batchIdx*batchSize,(batchIdx+1)*batchSize):

        img = Image.open(fileNames[i])

        label=fileLabels[i]

        img=img.resize((28,28))

        img=np.asarray(img)

        trainImgs.append(img)

        trainLabels.append(label)

    return np.array(trainImgs)/255.0,np.array(trainLabels)
import tensorflow as tf



x=tf.placeholder(tf.float32,[None,28,28,3])

y=tf.placeholder(tf.int32,[None,2])

prob = tf.placeholder_with_default(1.0, shape=())



skipConnection1 = tf.layers.conv2d(inputs=x,filters=16,kernel_size=[2, 2],

                         padding="same",activation=tf.nn.relu)

skipConnection2= tf.layers.conv2d(inputs=x,filters=16,kernel_size=[3, 3],

                         padding="same",activation=tf.nn.relu)

skipConnection3= tf.layers.conv2d(inputs=x,filters=16,kernel_size=[4, 4],

                         padding="same",activation=tf.nn.relu)



conv1=tf.keras.layers.concatenate([skipConnection1,skipConnection2,skipConnection3],axis=-1)



conv2 = tf.layers.conv2d(inputs=conv1,filters=64,kernel_size=[3, 3],

      padding="same",activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# 14*14*64



conv3 = tf.layers.conv2d(inputs=pool2,filters=64,kernel_size=[3, 3],

      padding="same",activation=tf.nn.relu)

pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

# 7*7*64



conv4 = tf.layers.conv2d(inputs=pool3,filters=128,kernel_size=[3, 3],

      padding="same",activation=tf.nn.relu)

pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)



pool2_flat = tf.reshape(pool4, [-1, 3 * 3 * 128])

dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

dropout = tf.layers.dropout(dense, prob)



dense2 = tf.layers.dense(inputs=dropout, units=32, activation=tf.nn.relu)



logits = tf.layers.dense(inputs=dense2, units=2)



loss = tf.losses.softmax_cross_entropy(y, logits)



# Configure the Training Op (for TRAIN mode)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)

train_op = optimizer.minimize(loss=loss)

## Batch 200*300

init=tf.global_variables_initializer()

sess=tf.Session()



sess.run(init)

batchInt=0

for fulliters in range(200):

    batchInt+=1

    totLoss=0

    for batch in range(100):

        x_train,y_train=makeTrainingBatch(batch,175)

        _,currLoss=sess.run([train_op,loss],feed_dict={x:x_train,y:y_train,prob:0.3})

        totLoss+=currLoss

    print('Batch: '+str(batchInt) + ' Loss is: '+ str(totLoss/175))
sampleSubmit=pd.read_csv('/kaggle/input/aerial-cactus-identification/sample_submission.csv')

sampleSubmit.head()
testBaseDir='/kaggle/input/aerial-cactus-identification/test/test/'

fileNamesPred=[str(testBaseDir+x) for x in sampleSubmit['id'].tolist()]

fileLabelsPred=sampleSubmit['has_cactus'].tolist()
def makeTestingBatch(batchIdx,batchSize):

    trainImgs=[]

    trainLabels=[]

    for i in range(batchIdx*batchSize,(batchIdx+1)*batchSize):

        img = Image.open(fileNamesPred[i])

        label=fileLabelsPred[i]

        img=img.resize((28,28))

        img=np.asarray(img)

        trainImgs.append(img)

        trainLabels.append(label)

    return np.array(trainImgs)/255.0,np.array(trainLabels)
for fulliters in range(1):

    masterPred=[]

    x_train,y_train=makeTestingBatch(0,len(fileLabelsPred))

    pred=sess.run([logits],feed_dict={x:x_train,prob:1.0})

    for i in pred[0]:

        if i[1]>i[0]:

            masterPred.append(1)

        else:

            masterPred.append(0)

sampleSubmit['has_cactus']=masterPred
sampleSubmit.to_csv('toSubmit.csv',index=False)