# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import pandas as pd

import math

import numpy as np

import tensorflow as tf

from sklearn.preprocessing import StandardScaler







def one_hot_encoding(array):

    length=len(array)

    mirror=[]

    for i in range(length):

        if array[i]==1:

            mirror.append(0)

        else:

            mirror.append(1)

    result=np.column_stack((array,np.array(mirror)))

    return result

        



def import_data():

    #raw=pd.read_csv('c:\Users\lenove\Desktop\datafile\creditcard.csv')

    #raw=pd.read_csv('C:\\Users\\lenove\\Desktop\\datafile\\creditcard.csv')

    raw=pd.read_csv('/usr/local/creditcard/creditcard.csv')

    raw['Amount'] = StandardScaler().fit_transform(raw['Amount'].reshape(-1, 1))

    raw = raw.drop(['Time'],axis=1)

    data_is_one=raw[raw['Class']==1].values

    data_is_zero=raw[raw['Class']==0].values

    trainX_len_one=math.floor(data_is_one.shape[0]*0.7)

    trainX_len_zero=math.floor(data_is_zero.shape[0]*0.7)

    train_one=data_is_one[:trainX_len_one]

    train_zero=data_is_zero[:trainX_len_zero]

    #train_zero=train_zero[:50]

    train=np.concatenate((train_one,train_zero),axis=0)

    np.random.shuffle(train)

    trainX=train[:,:-1]

    trainY=train[:,-1]

    trainY=one_hot_encoding(trainY)

    test_one=data_is_one[trainX_len_one:]

    test_zero=data_is_zero[trainX_len_zero:]

    test=np.concatenate((test_one,test_zero),axis=0)

    np.random.shuffle(test)

    testX=test[:,:-1]

    testY=test[:,-1]

    testY=one_hot_encoding(testY)

    return trainX,trainY,testX,testY



trainX,trainY,testX,testY=import_data()



def next_batch(num, data1,data2):

    idx = np.arange(0, len(data1))  

    np.random.shuffle(idx)  

    idx = idx[0:num]  

    data_shuffle_x = [data1[i] for i in idx]

    data_shuffle_y=[data2[i] for i in idx]

    data_shuffle_x = np.asarray(data_shuffle_x)  

    data_shuffle_y = np.asarray(data_shuffle_y)



    return data_shuffle_x,data_shuffle_y





feature_num=trainX.shape[1]

labels=2



steps=100000



learningrate=tf.train.exponential_decay(learning_rate=0.001,

                                        global_step=1,

                                        decay_steps=trainX.shape[0],

                                        decay_rate=0.95,

                                        staircase=True)



X=tf.placeholder(tf.float32,shape=[None,feature_num])

Y=tf.placeholder(tf.float32,shape=[None,labels])



Weights=tf.Variable(tf.random_normal([feature_num,150],

                                     mean=0,stddev=0.1),name='weight')

bias=tf.Variable(tf.random_normal([1,150],

                                  mean=0,stddev=0.1),name='bias')



wmux=tf.matmul(X,Weights)

wmux_plus_b=tf.add(wmux,bias)

layer1=tf.nn.relu(wmux_plus_b)





def add_layer_relu(x,input_size,output_size):

    Weights=tf.Variable(tf.random_normal([input_size,output_size],

                                     mean=0,stddev=0.1))

    bias=tf.Variable(tf.random_normal([1,output_size],

                                  mean=0,stddev=0.1))

    wmux=tf.matmul(x,Weights)

    wmux_plus_b=tf.add(wmux,bias)

    activation=tf.nn.relu(wmux_plus_b)

    return activation



def add_layer_tanh(x,input_size,output_size):

    Weights=tf.Variable(tf.random_normal([input_size,output_size],

                                     mean=0,stddev=0.1))

    bias=tf.Variable(tf.random_normal([1,output_size],

                                  mean=0,stddev=0.1))

    wmux=tf.matmul(x,Weights)

    wmux_plus_b=tf.add(wmux,bias)

    activation=tf.nn.tanh(wmux_plus_b)

    return activation



def add_layer_softmax(x,input_size,output_size):

    Weights=tf.Variable(tf.random_normal([input_size,output_size],

                                     mean=0,stddev=0.1))

    bias=tf.Variable(tf.random_normal([1,output_size],

                                  mean=0,stddev=0.1))

    wmux=tf.matmul(x,Weights)

    wmux_plus_b=tf.add(wmux,bias)

    activation=tf.nn.softmax(wmux_plus_b)

    return activation



layer2=add_layer_relu(layer1,150,60)

layer3=add_layer_relu(layer2,60,30)

activation=add_layer_relu(layer3,30,2)







cost=tf.nn.l2_loss(activation-Y,name="squared_error_cost")

train=tf.train.AdamOptimizer(learningrate).minimize(cost)



sess=tf.Session()

init=tf.global_variables_initializer()

sess.run(init)



correct_prediction_op=tf.equal(tf.argmax(activation,1),tf.argmax(Y,1))

acc_op=tf.reduce_mean(tf.cast(correct_prediction_op,'float'))

cost_value=0

diff=1

train_acc=0

for i in range(steps):

    if train_acc>1:

        print('diff convergence',diff)

        break

    else:

        batch_x,batch_y=next_batch(5000,trainX,trainY)

        #batch_y=batch_y.reshape(batch_y.shape[0],1)

        #trainY=trainY.reshape(trainY.shape[0],1)

        step=sess.run(train,feed_dict={X:batch_x,Y:batch_y})

    if i%10 is 0:

        train_acc,newcost=sess.run([acc_op,cost],feed_dict={X:batch_x,Y:batch_y})

        diff=abs(newcost-cost_value)

        test_acc=sess.run(acc_op,feed_dict={X:testX,Y:testY})

        cost_value=newcost

        print('step',i,'train_acc',train_acc,'cost',newcost,'test acc',test_acc)

testY=testY.reshape(testY.shape[0],1)    



test_acc=sess.run(acc_op,feed_dict={X:testX,Y:testY})



print('test_acc',test_acc)


