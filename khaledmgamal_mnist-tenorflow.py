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

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install tensorflow==1.13.1

import pandas as pd

import numpy as np

import tensorflow as tf











def read_csv(file_path):

    data=pd.read_csv(file_path)

    return data



def pre_process(data,is_test):

    if is_test==False:

       y=data['label']

       y_one_hot=pd.get_dummies(y,prefix='label')

       X_train=data.drop('label',axis=1)

       X_train_np=np.array(X_train)

       X_train_np=X_train_np.reshape([X_train_np.shape[0],28,28,1])

       X_train_np_normalized=X_train_np/255

    

       return X_train_np_normalized,y_one_hot

    if is_test==True:

       X_train=data

       y_one_hot=0

       X_train_np=np.array(X_train)

       X_train_np=X_train_np.reshape([X_train_np.shape[0],28,28,1])

       X_train_np_normalized=X_train_np/255

    

       return X_train_np_normalized,y_one_hot

   



'''

Weights={'Wc1':tf.Variable(tf.random_normal([5,5,1,32])),

         'Wc2':tf.Variable(tf.random_normal([5,5,32,64])),

         'Wc3':tf.Variable(tf.random_normal([5,5,64,128])),



         'Wd1':tf.Variable(tf.random_normal([4*4*128,128])),

         'Wd2':tf.Variable(tf.random_normal([128,10]))

         

        }   



Biases={'bc1':tf.Variable(tf.random_normal([32])),

        'bc2':tf.Variable(tf.random_normal([64])),

        'bc3':tf.Variable(tf.random_normal([128])),



        'bd1':tf.Variable(tf.random_normal([128])),

        'bd2':tf.Variable(tf.random_normal([10]))

        

        }



'''

Weights = {

    'Wc1' : tf.get_variable('W0', shape = (3, 3, 1, 32), initializer = tf.contrib.layers.xavier_initializer()),

    'Wc2' : tf.get_variable('W1', shape = (3, 3, 32, 64), initializer = tf.contrib.layers.xavier_initializer()),

    'Wc3' : tf.get_variable('W2', shape = (3, 3, 64, 128), initializer = tf.contrib.layers.xavier_initializer()),

    'Wd1' : tf.get_variable('W3', shape = (4 * 4 * 128, 128), initializer = tf.contrib.layers.xavier_initializer()),

    'Wd2' : tf.get_variable('W4', shape = (128, 10), initializer = tf.contrib.layers.xavier_initializer())

}



Biases = {

    'bc1': tf.get_variable('B0', shape = (32), initializer = tf.contrib.layers.xavier_initializer()),

    'bc2': tf.get_variable('B1', shape = (64), initializer = tf.contrib.layers.xavier_initializer()),

    'bc3': tf.get_variable('B2', shape = (128), initializer = tf.contrib.layers.xavier_initializer()),

    'bd1': tf.get_variable('B3', shape = (128), initializer = tf.contrib.layers.xavier_initializer()),

    'bd2': tf.get_variable('B4', shape = (10), initializer = tf.contrib.layers.xavier_initializer()),

}



def conv2d(input_,W,b,stride=1):

    out=tf.nn.conv2d(input_,W,[1,stride,stride,1],'SAME')

    out=tf.nn.bias_add(out, b)

    out=tf.nn.relu(out)

    

    return out

def maxPooling2d(input_,k=2):

    

    return tf.nn.max_pool(input_,ksize=[1, k, k, 1],strides=[1, k, k, 1],padding='SAME')



def conv_net(input_,W,b,stride,k,dropout):

    conv1=conv2d(input_,W['Wc1'],b['bc1'],stride)

    conv1=maxPooling2d(conv1,k)

    conv2=conv2d(conv1,W['Wc2'],b['bc2'],stride)

    conv2=maxPooling2d(conv2,k)

    conv3=conv2d(conv2,W['Wc3'],b['bc3'],stride)

    conv3=maxPooling2d(conv3,k)

    fc1_input=tf.reshape(conv3,[-1,W['Wd1'].shape.as_list()[0]])

    fc1=tf.add(tf.matmul(fc1_input,W['Wd1']),b['bd1'])

    fc1 = tf.nn.relu(fc1)

    fc1=tf.nn.dropout(fc1,dropout)

    fc2=tf.add(tf.matmul(fc1,W['Wd2']),b['bd2'])

    

    return fc2





def genrate_batches(X_data,y_data,batch_size):

    no_batches=0

    batches=[]

    X_data=np.array(X_data)

    y_data=np.array(y_data)

    if X_data.shape[0]%batch_size==0:

        no_batches=X_data.shape[0]/batch_size

    else :

        no_batches=X_data.shape[0]//batch_size+1

    for batch in range(int(no_batches)):

        

       if batch<no_batches-1:

          batches.append([X_data[batch*batch_size:(batch*batch_size)+batch_size,:,:,:],\

                              y_data[batch*batch_size:(batch*batch_size)+batch_size,:]])

       if batch==no_batches-1 :

          batches.append([X_data[batch*batch_size:,:,:,:],\

                              y_data[batch*batch_size:,:]])

    return  np.array(batches),no_batches     

    

  

# parameters

learning_rate = 0.001

epochs = 10

batch_size = 128



# number of samples to calculate validation and accuracy

# decrease this if you're running out of memory

test_valid_size = 256



# network Parameters

n_classes = 10  # MNIST total classes (0-9 digits)

dropout = 0.9 # dropout (probability to keep units)    

X=tf.placeholder(tf.float32,[None,28,28,1],name='input_') 

y=tf.placeholder(tf.float32,[None,10],name='out')

keep_prob = tf.placeholder(tf.float32,name='drop')



logits=conv_net(X,Weights,Biases,1,2,keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

y_pred = tf.nn.softmax(logits,name="y_pred")

init=tf.global_variables_initializer()





train_data=read_csv('/kaggle/input/digit-recognizer/train.csv')

X_train,y_train=pre_process(train_data,is_test=False)

batches,no_batches=genrate_batches(X_train,y_train,batch_size)



with tf.Session() as sess:

    sess.run(init)

    for e in range(epochs):

        for batch in range(no_batches):

            dict_={

                X: batches[batch][0],

                y: batches[batch][1],

                keep_prob: dropout}

            sess.run(optimizer, feed_dict=dict_)

            loss = sess.run(cost, feed_dict=dict_)

            valid_acc = sess.run(accuracy, feed_dict=dict_)



            print('Epoch {:>2}, Batch {:>3} -'

                  'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(

                e + 1,

                batch + 1,

                loss,

                valid_acc))

    saver = tf.train.Saver()   

    saver.save(sess, 'my_model')









test_data=read_csv('/kaggle/input/digit-recognizer/test.csv')

X_test,y_test=pre_process(test_data,is_test=True)

Y_=[]

sess =tf.Session() 

new_saver = tf.train.import_meta_graph('my_model.meta')

new_saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()

X = graph.get_tensor_by_name('input_:0')

y_predict=graph.get_tensor_by_name("y_pred:0")

keep_prob=graph.get_tensor_by_name('drop:0')

for batch_no in range((X_test.shape[0]+(batch_size-1))//batch_size):

    test_batch=X_test[batch_no*batch_size:min((batch_no+1)*batch_size,X_test.shape[0])]

    dict_={X:test_batch,keep_prob:dropout}

    y_predicts=sess.run(y_predict,feed_dict=dict_)

    Y_=[*Y_,*y_predicts]



Y_=np.array(Y_)



minst=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

minst=minst.drop('Label',axis=1)

minst['Label']=np.argmax(Y_,axis=1)

minst.to_csv('sample_submission03.csv',index=None)