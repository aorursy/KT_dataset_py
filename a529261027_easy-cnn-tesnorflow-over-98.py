import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
train_df=pd.read_csv('../input/train.csv')
y_=train_df['label'].values.ravel()
x_=train_df.iloc[:,1:].values.astype(np.float)
x_=np.multiply(x_, 1.0 / 255.0)
y_=pd.get_dummies(y_)
print(y_.head())
x_train,x_test,y_train,y_test=train_test_split(x_,y_,test_size=0.1,random_state=1)
def Weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def Biase(shape):
    return tf.Variable(tf.zeros(shape)+0.1)
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
def pooling(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,28,28,1])
w1_cov=Weight([5,5,1,32])
b1_cov=Biase([32])
l1_cov=conv2d(x_image,w1_cov)+b1_cov
l1_pooling=pooling(l1_cov)
 
w2_cov=Weight([5,5,32,64])
b2_cov=Biase([64])
l2_cov=conv2d(l1_pooling,w2_cov)+b2_cov
l2_pooling=pooling(l2_cov)
l_flat=tf.reshape(l2_pooling,[-1,7*7*64])
 
w1_full=Weight([7*7*64,1024])
b1_full=Biase([1024])
wx1_full=tf.matmul(l_flat,w1_full)+b1_full
l1_all=tf.nn.relu(wx1_full)
 
w2_full=Weight([1024,10])
b2_full=Biase([10])
wx2_full=tf.matmul(l1_all,w2_full)+b2_full
l2_full=tf.nn.softmax(wx2_full)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=l2_full))
train=tf.train.AdamOptimizer(0.0001).minimize(loss)
correct=tf.equal(tf.argmax(y,1),tf.argmax(l2_full,1))
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
def minbatch(n):
    all_train=np.concatenate([x_train,y_train],axis=1)
    random.shuffle(all_train)
    result=[]
    num=int(x_train.shape[0]/n)
    for i in range(num):
        result.append(all_train[i*n:(i+1)*n])
    return result
saver=tf.train.Saver()
n=100
num=int(x_train.shape[0]/100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Start-Train')
    for i in range(11):
        all_train=minbatch(n)
        for j in range(num):
            x_batch,y_batch=all_train[j][:,:784],all_train[j][:,-10:]
            sess.run(train,feed_dict={x:x_batch,y:y_batch})
        acc=sess.run(accuracy,feed_dict={x:x_test,y:y_test})
        print(i,acc)
    saver.save(sess,'..\\net-cnn.ckpt')
test_df=pd.read_csv('..\\test.csv')
test_data=test_df.iloc[:,:].values.astype(np.float)
test_data=np.multiply(test_data, 1.0 / 255.0)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,'..\\net-cnn.ckpt')
    y_hat=sess.run(l2_full,feed_dict={x:test_data})
result=[]
for i in y_hat:
    result.append((np.argmax(i)))
predict=pd.DataFrame(data={'ImageId':list(range(1,28001)),'Label':result})
predict.to_csv('..\\cnn-1.csv',index=False)