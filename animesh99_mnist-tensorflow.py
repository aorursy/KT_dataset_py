import pandas as pd

import numpy as np

import keras as  kr

import matplotlib.pyplot as plt
def get_mnist_train_data():

    train=pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

    train_label=np.array(train["label"])

    train_data=np.reshape(np.array(train.drop("label",axis=1)),(train.drop("label",axis=1).shape[0],28,28,1))

    return train_data,train_label
def get_mnist_test_data():

    test=pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

    test_label=np.array(test["label"])

    test_data=np.reshape(np.array(test.drop("label",axis=1)),(test.drop("label",axis=1).shape[0],28,28,1))

    return test_data,test_label
def plot_img(pass_data,pass_label,fig_size,number,channel):

    fig,ax=plt.subplots(figsize=fig_size,dpi=80)

    ax.set_yticklabels([])

    ax.set_xticklabels([])

    for i,data in enumerate(pass_data[number[0]:number[1]]):

        count=1

        if channel=="ALL": 

            if data.shape[2]%5==0:

                row,col=data.shape[2]//5,5

            else:

                row,col=data.shape[2]//5+1,5

                

            for x in range(0,data.shape[2]):

                sub1 = plt.subplot(row,col,count)

                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

                sub1.imshow(data[:,:,x],cmap='gray')

                count+=1

        elif channel>0:

            if channel<5:

                row,col=1,channel

            elif channel%5:

                row,col=data.shape[2]//5,5

            else:

                row,col=data.shape[2]//5+1,5

          

            for x in range(0,channel):                

                sub1 = plt.subplot(row,col,count)

                sub1.imshow(data[:,:,x],cmap='gray')

                count+=1

        else:

            sub1 = plt.subplot(5, 5,i+1)

            sub1.imshow(data[:,:,0],cmap='gray')

#     fig.tight_layout()
train_data,train_label=get_mnist_train_data()
test_data,test_label=get_mnist_test_data()
test_data.shape
pred_one=[test_data[0]]

np.shape(pred_one)

plot_img(pred_one,0,(10,10),[0,1],0)
np.shape(pred_one)
plot_img(train_data,0,(10,10),[0,20],0)
import keras

train_label = keras.utils.to_categorical(train_label,10)

test_label=keras.utils.to_categorical(test_label,10)
import tensorflow as tf
train_data=train_data.astype(np.float32)
initializer = tf.contrib.layers.xavier_initializer()

weight1=tf.Variable(initializer((5,5,1,16)),name="weight1")

weight2=tf.Variable(initializer((3,3,16,32)),name="weight2")

weight3=tf.Variable(initializer((3,3,32,64)),name="weight3")

weight4=tf.Variable(initializer((3,3,64,128)),name="weight4")

weight5=tf.Variable(initializer((3,3,128,128)),name="weight5")
placeholder1=tf.placeholder(tf.float32,[None,28,28,1],name="DataHolder")

label=tf.placeholder(tf.float32,[None,10],name="DataLabel")
conv_layer1=tf.nn.conv2d(placeholder1,weight1,strides=[1,1,1,1],padding="VALID")

relu1=tf.nn.relu(conv_layer1,name="relu1")

conv_layer2=tf.nn.conv2d(relu1,weight2,strides=[1,1,1,1],padding="VALID")

relu2=tf.nn.relu(conv_layer2,name="relu2")

maxpool1=tf.nn.max_pool(relu2,[1,2,2,1],[1,1,1,1],padding="VALID")

conv_layer3=tf.nn.conv2d(maxpool1,weight3,strides=[1,1,1,1],padding="VALID")

relu3=tf.nn.relu(conv_layer3,name="relu3")

conv_layer4=tf.nn.conv2d(relu3,weight4,strides=[1,1,1,1],padding="SAME")

relu4=tf.nn.leaky_relu(conv_layer4,name="relu4")

maxpool2=tf.nn.max_pool(relu4,[1,2,2,1],[1,1,1,1],padding="VALID")

conv_layer5=tf.nn.conv2d(maxpool2,weight5,strides=[1,1,1,1],padding="SAME")

relu5=tf.nn.leaky_relu(conv_layer5,name="relu5")
relu3.get_shape()
flat=tf.layers.flatten(relu5)

dense1=tf.layers.dense(flat,units=200,activation="sigmoid", kernel_initializer=tf.contrib.layers.xavier_initializer())

dense2=tf.layers.dense(dense1,units=75,activation="sigmoid", kernel_initializer=tf.contrib.layers.xavier_initializer())

dense3=tf.layers.dense(dense2,units=10,activation="sigmoid", kernel_initializer=tf.contrib.layers.xavier_initializer())
final_pred = tf.nn.softmax(dense3,name="prediction")

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense3,labels=label)

cost_op = tf.reduce_mean(cross_entropy)

correct_prediction = tf.equal(tf.argmax(final_pred,1),tf.argmax(label,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

train_op = tf.train.GradientDescentOptimizer(0.006).minimize(cost_op)
sess=tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(50):

    for b_num in range(train_data.shape[0]//100):

        t_data,t_label=train_data[b_num*100:(b_num*100)+100],train_label[b_num*100:(b_num*100)+100]

        val=sess.run([train_op,cost_op,accuracy],feed_dict={placeholder1:t_data,label:t_label})

    print("Epoch    : ",epoch)

    print("Loss     : ",val[1])

    print("accurecy : ",accuracy.eval(feed_dict={placeholder1:test_data,label:test_label},session=sess))

    print("--------------------")

print("Accuracy : ", accuracy.eval(feed_dict={placeholder1:test_data,label:test_label},session=sess))
saver=tf.train.Saver();

saver.save(sess,"model")
sess.close()
sess=tf.Session()    

#First let's load meta graph and restore weights

saver = tf.train.import_meta_graph('../input/mnist-tensorflow/model.meta')

saver.restore(sess,tf.train.latest_checkpoint('../input/mnist-tensorflow/'))
graph = tf.get_default_graph()

dataPlaceholder = graph.get_tensor_by_name("DataHolder:0")

feed_dict ={dataPlaceholder:pred_one}



# Now, access the op that you want to run. 

op_to_restore = graph.get_tensor_by_name("prediction:0")

out=sess.run(op_to_restore,feed_dict)

np.argmax(out[0])

#This will print 60 which is calculated 
final=sess.run(final_pred,feed_dict={placeholder1:pred_one})
np.argmax(final[0])