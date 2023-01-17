# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/fashion-mnist_train.csv')
X_train=data.iloc[:,1:].values.reshape(-1,28,28,1)/255

Y_train=data.iloc[:,0].values.reshape(-1,1)
print(X_train.shape,Y_train.shape)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
Y_Train=np.zeros((60000,10))

for elem in range(60000):

    Y_Train[elem,Y_train[elem,0]]=1

Y_train=Y_Train

Xtr,Xte,Ytr,Yte=train_test_split(X_train,Y_train,test_size=0.2)

import tensorflow as tf
X=tf.placeholder(tf.float32,[None,28,28,1])

Y=tf.placeholder(tf.float32,[None,10])
c1=tf.contrib.layers.conv2d(X/255, 20, kernel_size=(3,3))

p1=tf.contrib.layers.max_pool2d(c1,(2,2))

bn1=tf.contrib.layers.batch_norm(p1,center=True, scale=True,is_training=True,scope='bn1')

a1=tf.nn.relu(bn1)



c2=tf.contrib.layers.conv2d(a1, 40, kernel_size=(3,3))

p2=tf.contrib.layers.max_pool2d(c2,(2,2))

bn2=tf.contrib.layers.batch_norm(p2,center=True, scale=True,is_training=True,scope='bn2')

a2=tf.nn.relu(bn2)



c3=tf.contrib.layers.conv2d(a2, 60, kernel_size=(3,3))

p3=tf.contrib.layers.max_pool2d(c3,(2,2))

bn3=tf.contrib.layers.batch_norm(p3,center=True, scale=True,is_training=True,scope='bn3')

a3=tf.nn.relu(bn3)

F=tf.contrib.layers.flatten(a3)

F1=tf.contrib.layers.fully_connected(F,100)

Yhat=tf.contrib.layers.fully_connected(F1,10)
loss=tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=Y,logits=Yhat))

#tf.nn.softmax_cross_entropy_with_logits()
opt=tf.train.AdamOptimizer(0.001).minimize(loss)
with tf.Session() as sess:

    tf.global_variables_initializer().run()

    for i in range(500+1):

        inds=np.random.randint(0,48000,size=128)

        X_batch,Y_batch=Xtr[inds,...],Ytr[inds,...]

        _=sess.run(opt,feed_dict={X:X_batch,Y:Y_batch})

        if i%50==0:

            print("Loss at iter ",i,' is ',sess.run(loss,feed_dict={X:X_batch,Y:Y_batch}))

    inds_te=np.random.randint(0,12000,size=128)

    inds_tr=np.random.randint(0,48000,size=128)

    xte,yte=Xte[inds_te],Yte[inds_te]

    xtr,ytr=Xtr[inds_tr],Ytr[inds_tr]

    predict_op = tf.argmax(Yhat, 1)

    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    test_accuracy = accuracy.eval({X:xte,Y:yte})

    train_accuracy = accuracy.eval({X:xtr,Y:ytr})

    print("Train Accuracy : ",train_accuracy)

    print("Test Accuracy : ",test_accuracy)