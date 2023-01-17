import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
# Import Data

data = pd.read_csv( '../input/train.csv' ).as_matrix()

                                                     

# Train / Test Split

from sklearn.model_selection import train_test_split

train_data, test_data, train_label, test_label = train_test_split( data[:,1:].astype( 'float32' ), data[:,0].astype( 'float32' ), test_size=0.33,random_state=0 )

train_label = pd.get_dummies( train_label ).as_matrix()

test_label = pd.get_dummies( test_label ).as_matrix()

train_data /=255

test_data /=255

# Create Model

def weight(shape): #weight

    return tf.Variable(tf.truncated_normal(shape,stddev=0.1),name='W') #stddev=標準差

def bias(shape): #bias

    return tf.Variable(tf.constant(0.1, shape=shape), name = 'b')

def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') 

    #strides=[1,由左而右一次移動幾步,由上而下一次移動幾步,1]

    #padding=邊緣處理方法

def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    #ksize=[1,height,width,1]



#Input Layer

with tf.name_scope('Input_Layer'):

    x = tf.placeholder("float",shape=[None, 784],name="x")   #None是因為不確定會讀幾張圖 

    x_image = tf.reshape(x, [-1, 28, 28, 1]) #輸入必須要是4維

    #第一個-1是因為訓練時要透過placeholder輸入的筆數不固定所以設定-1

    #第二個1是通道數(灰階=1、彩色=3)



#x->28*28*1

#Convolutional Layer 1 

with tf.name_scope('C1_Conv'): 

    W1 = weight([5,5,1,16]) #filter size 5x5,灰階=1,有16個filter要產生16個影像

    b1 = bias([16])

    Conv1=conv2d(x_image, W1)+ b1

    C1_Conv = tf.nn.relu(Conv1 )#->28*28*16

with tf.name_scope('C1_Pool'):

    C1_Pool = max_pool_2x2(C1_Conv)#->14*14*16

    

# Convolutional Layer 2   

with tf.name_scope('C2_Conv'):

    W2 = weight([5,5,16,36])

    b2 = bias([36])

    Conv2 = conv2d(C1_Pool, W2) + b2

    C2_Conv = tf.nn.relu(Conv2)#->14*14*36

with tf.name_scope('C2_Pool'):

    C2_Pool = max_pool_2x2(C2_Conv)#->7*7*36

    

# Fully Connected

# Flat Layer

with tf.name_scope('D_Flat'): 

    D_Flat = tf.reshape(C2_Pool, [-1, 1764]) #將Convolutional Layer 2 的output7*7*36轉換為一維的像量1*1764

# Hidden Layer

with tf.name_scope('D_Hidden_Layer'): 

    W3= weight([1764, 128])#hidden layer有128個神經元

    b3= bias([128])

    D_Hidden = tf.nn.relu(tf.matmul(D_Flat, W3)+b3)

    D_Hidden_Dropout= tf.nn.dropout(D_Hidden,keep_prob=0.8)#(要執行dropout的神經網路層,要保留的神經元比例)

#Output Layer

with tf.name_scope('Output_Layer'):

    W4 = weight([128,10])

    b4 = bias([10])    

    y_predict= tf.nn.softmax(tf.matmul(D_Hidden_Dropout,W4)+b4)





# Optimizer

with tf.name_scope("optimizer"):

    

    y_label = tf.placeholder("float", shape=[None, 10], name="y_label")

    

    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict , labels=y_label))#cross entropy

    

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_function)#adam學習方式

# Accuracy

with tf.name_scope("evaluate_model"):

    correct_prediction = tf.equal(tf.argmax(y_predict, 1),tf.argmax(y_label, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Train

train_size = train_data.shape[0]

batch_size = 100

epoch_size = 10

from time import time 

startTime = time()



sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)



for epoch in range(epoch_size):

    for i in range( 0, train_size, batch_size ):

        batch = i%train_size

        sess.run( optimizer, feed_dict={x: train_data[batch:batch+batch_size],y_label:train_label[batch:batch+batch_size] })

    loss, acc = sess.run( [loss_function, accuracy], feed_dict={x: train_data[batch:batch+batch_size],y_label:train_label[batch:batch+batch_size] } )

    print('Step: ' + str(epoch) + ' loss: ' + str(loss) + ' accuracy: ' + str(acc) )

    

print('time:',time()-startTime)

  

# Unseen data

print("Accuracy:", sess.run(accuracy,feed_dict={x: test_data,y_label: test_label}))

# Predict

df = pd.read_csv( '../input/test.csv' )

predict_x = df.as_matrix() / 255

predict_y = tf.argmax( y_predict, 1 )

predicted_y = sess.run( predict_y, feed_dict = {x: predict_x} )



# Save Ouput

pd.DataFrame({"ImageId": list(range(1,len(predicted_y)+1)), "Label": predicted_y}).to_csv('output.csv', index=False, header=True)

print("over")

sess.close()