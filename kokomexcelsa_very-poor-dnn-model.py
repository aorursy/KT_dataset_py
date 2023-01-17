%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import os

where_img = pd.read_csv('img-class.csv')
print(where_img.head())
img = cv2.imread('//data/examples/may_the_4_be_with_u/where_am_i/train/' + where_img.img.iloc[0], 0)

where_img = where_img[where_img.img.str.contains('image')]
where_y = pd.get_dummies(where_img['classname'], '').as_matrix()#將label做 one_hot encoding
print(where_y)
x_train_list, y_train = where_img, where_y #資料切割
## record the dim of img ##
img = cv2.resize(img, (256,256))#改變圖片大小(太大會跑太久)
img = img.flatten()

from sklearn.utils import shuffle 

def simpson_train_batch_generator(x, y, bs, shape):
    x_train = np.array([]).reshape((0, shape))
    y_train = np.array([]).reshape((0, y.shape[1]))
    while True:
        new_ind = shuffle(range(len(x)))
        x = x.take(new_ind)
        y = np.take(y, new_ind, axis=0)
        for i in range(len(x)):
            dir_img = '//data/examples/may_the_4_be_with_u/where_am_i/train/' + x.img.iloc[i]
            img = cv2.imread(dir_img, 0)
            img = cv2.resize(img, (256,256))
            x_train = np.row_stack([x_train, img.flatten()])
            y_train = np.row_stack([y_train, y[i]])
            if x_train.shape[0] == bs:
                x_batch = x_train.copy()
                x_batch /= 255.
                y_batch = y_train.copy()
                x_train = np.array([]).reshape((0 ,shape))
                y_train = np.array([]).reshape((0 ,y.shape[1]))        
                yield x_batch, y_batch
tf.reset_default_graph()
#### define placeholder ####
input_data = tf.placeholder(dtype=tf.float32, 
                           shape=[None, img.shape[0]],
                           name='input_data') #用來接 feature 資料進入 tensorflow 

y_true = tf.placeholder(dtype=tf.float32, 
                        shape=[None, y_train.shape[1]],
                        name='y_true') #用來接 label 資料進入 tensorflow 
#### define variables(weight/bias) ####
x1 = tf.layers.dense(input_data, 256, activation=tf.nn.relu, name='hidden1') #第一層hidden layer
x2 = tf.layers.dense(x1, 128, activation=tf.nn.relu, name='hidden2') #第二層hidden layer
x3 = tf.layers.dense(x2, 64, activation=tf.nn.relu, name='hidden3')#第三層hidden layer
x4 = tf.layers.dense(x3, 32, activation=tf.nn.relu, name='hidden4')#第三層hidden layer
x5 = tf.layers.dense(x4, 16, activation=tf.nn.relu, name='hidden5')#第三層hidden layer
out = tf.layers.dense(x5, y_train.shape[1], name='output')# output layer

y_pred = out
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)

update = opt.minimize(loss)
tf.global_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
from tqdm import tqdm_notebook #用來顯示進度條的套件
from sklearn.metrics import accuracy_score

epoch = 300 #要跑多少epoch
bs = 32 #設定看過幾筆資料走一次更新(batch size)
update_per_epoch = 100 #一個epoch要跑幾個batch
'''因為我們現在是用generator產生新的batch，所以我們要自行給定一個epoch要跑幾個batch,
當然也可以另外修改程式讓generator回傳適合的batch數(total number of data/batch size)'''

tr_loss = list() #準備一個空的list用來存training過程中的loss值
tr_acc = list() #準備一個空的list用來存training過程中的準確率
train_gen = simpson_train_batch_generator(x_train_list, y_train, bs, img.shape[0])

print('start modelling!')

for i in range(epoch):
    
    #### calculate training loss & update variables ####
    training_loss = 0 #用來計算epoch內所有batch loss的平均值，所以在每個epoch一開始要歸零
    training_acc = 0#用來計算epoch內所有batch acc的平均值，所以在每個epoch一開始要歸零
    bar = tqdm_notebook(range(update_per_epoch)) #外面的tqdm是進度條的function
    
    for j in bar:
        
        x_batch, y_batch = next(train_gen) #我們是用yield來寫generator,所以用next取得下一筆資料
        
        tr_pred, training_loss_batch, _ = sess.run([y_pred, loss, update], feed_dict={
            input_data:x_batch,
            y_true:y_batch
        })
        
        training_loss += training_loss_batch
        
        training_acc_batch = accuracy_score(np.argmax(y_batch, axis=1), np.argmax(tr_pred, axis=1))
        training_acc += training_acc_batch
        
        if j % 5 == 0:
            bar.set_description('loss: %.4g' % training_loss_batch) 
            #每5次batch更新顯示的batch loss(進度條前面)

    training_loss /= update_per_epoch
    training_acc /= update_per_epoch
    
    tr_loss.append(training_loss)
    tr_acc.append(training_acc)
    
    print('epoch {epochs}: training loss {training_loss}'.format(
            epochs=(i+1), 
            training_loss=training_loss))#每個epoch結束後顯示目前的的training loss

path = os.listdir('testset/')
total_img = list()
for i in range(len(path)):
    print(path[i])
#     subimg = cv2.imread('testset/' + path[i],0)
#     subimg = cv2.resize(subimg,(256,256))
#     subimg = subimg.reshape(-1)
#     total_img.append(subimg)
    
test = sess.run([y_pred], feed_dict={input_data: total_img})
total_pred = np.argmax(test, axis=2)[0]
proced_path = [v.split('.')[0] for v in path]
submission_dict = dict(zip(proced_path, total_pred))
submission_tb = pd.DataFrame(list(submission_dict.items()), columns=['id','class'])
submission_tb.to_csv('submission_test.csv',index=False)
plt.figure(1)
plt.subplot(121)
plt.plot(range(len(tr_loss)), tr_loss, label='training')
plt.title('Loss')
plt.legend(loc='best')
plt.subplot(122)
plt.plot(range(len(tr_acc)), tr_acc, label='training')
plt.title('Accuracy')
print(sess.run(loss, feed_dict={
            input_data:x_batch,
            y_true:y_batch
        }))




