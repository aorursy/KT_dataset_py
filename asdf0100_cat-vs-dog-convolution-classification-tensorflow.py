import cv2

import numpy as np

import os

from random import shuffle

from tqdm import tqdm

import time

from datetime import timedelta

import matplotlib.pyplot as plt



import tensorflow as tf







 

TRAIN_DIR = "DogVsCat/train"

TEST_DIR = "DogVsCat/test"



IMG_SIZE = 50

LR = 1e-3







train_data_name = 'train_dvs_data.npy'

test_data_name = 'test_dvs_data.npy'



MODEL_NAME ="DVS-{}-{}.model".format(LR,'2dconv')
#TRAIN_DIR1 = "notebook/convClass/DogVsCat/train"

path = os.path.dirname(os.path.realpath('__file__'))

train_dir = os.path.join(path,TRAIN_DIR)

test_dir = os.path.join(path,TEST_DIR)
print("done")
def label_img(img):

    

    word_label = img.split('.')[-3]

  

    if word_label == 'cat': return [1,0]

    elif word_label == 'dog' : return [0,1]

def create_train_data():

    training_data = []

    for each_img in tqdm(os.listdir(train_dir)):

        label = label_img(each_img)

        each_img_path = os.path.join(train_dir,each_img)

        img_data = cv2.imread(each_img_path,cv2.IMREAD_GRAYSCALE)

        img_data = cv2.resize(img_data,(IMG_SIZE,IMG_SIZE))

        training_data.append([np.array(img_data),np.array(label)])

    shuffle(training_data)

    np.save(train_data_name,training_data)

    return training_data

create_train_data()
def process_test_data():

    testing_data = []

    for each_img in tqdm(os.listdir(test_dir)):

        path = os.path.join(test_dir,each_img)

        img_num = each_img.split('.')[0]

        img_data = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

        img_data = cv2.resize(img_data,(IMG_SIZE,IMG_SIZE))

        testing_data.append([np.array(img_data),img_num])

    shuffle(testing_data)

    np.save(test_data_name,testing_data)

    return testing_data
process_test_data()
#load data if exits

train_data = np.load(train_data_name)

test_data = np.load(test_data_name)

test_size = 0.25

testing_size = int(test_size*len(train_data))



train_x = list(train_data[:,0][:-testing_size])

train_y = list(train_data[:,1][:-testing_size])





test_x = list(train_data[:,0][-testing_size:])

test_y = list(train_data[:,1][-testing_size:])



testing_x = list(test_data[:,0])

testing_y =list(test_data[:,1])
len(testing_x)








filter_size = 5



num_filter1 = 32

num_filter2 =32

num_filter3=64

num_filter4=128



keep_rate=0.8

beta=0.1









fc_size = 512

fc_size2 =1024



num_channels = 1

num_classes = 2









x = tf.placeholder(tf.float32, [None, IMG_SIZE*IMG_SIZE], name='x')

y_true = tf.placeholder(tf.float32,[None,2],name ='y')





x = tf.reshape(x,[-1,IMG_SIZE,IMG_SIZE,num_channels])









save_path = 'tf_checkpoints/'







if not os.path.exists(save_path):

    os.makedirs(save_path)

    

save_validation_path = os.path.join(save_path,'best_validation')







def con2d(x,W):

    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')



def maxpool2d(x):

    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')



def new_Weight(shape):

    return tf.Variable(tf.random_normal(shape))

                       

def new_biases(length):

    return tf.Variable(tf.constant(0.05, shape=[length]))





def new_conv_neural_layer(x,filter_size,num_channels,num_filter):

    shape = [filter_size,filter_size,num_channels,num_filter]

    weights = new_Weight(shape)

    biases = new_biases(num_filter)

    conv = con2d(x,weights) + biases

    conv = maxpool2d(conv)

    conv = tf.nn.relu(conv)

    return conv ,weights



def new_fc_layer(input,num_inputs,num_outputs,use_relu=True):

    weights = new_Weight(shape=[num_inputs,num_outputs])

    biases = new_biases(length=num_outputs)

    fc_layer = tf.matmul(input,weights)+biases

    

    if use_relu:

        fc_layer = tf.nn.relu(fc_layer)

    return fc_layer,weights







def flatten_layer(conv_output):

    layer_shape = conv_output.get_shape()

    num_features = layer_shape[1:4].num_elements()

    flatten_conv = tf.reshape(conv_output,[-1,num_features])

    return flatten_conv,num_features



def build_network(dropout):

    layer_conv1,weights_conv1 = new_conv_neural_layer(x,filter_size,num_channels,num_filter1)

    layer_conv2,weights_conv2 = new_conv_neural_layer(layer_conv1,filter_size,num_filter1,num_filter2)   

    layer_conv3,weights_conv3 = new_conv_neural_layer(layer_conv2,filter_size,num_filter2,num_filter3)

    layer_conv4,weights_conv4 = new_conv_neural_layer(layer_conv3,filter_size,num_filter3,num_filter4)

    

    flatten_conv,num_feature = flatten_layer(layer_conv4)

    fc_layer1,weights_flat1 = new_fc_layer(flatten_conv,num_feature,fc_size,True)

    

        

    fc_layer2,weights_flat2 = new_fc_layer(fc_layer1,fc_size,fc_size2,True)



        

    fc_layer3,weights_output = new_fc_layer(fc_layer2,fc_size2,num_classes,False)

    

    if(dropout):

        fc_layer3 = tf.nn.dropout(fc_layer3,keep_rate)

    



    return fc_layer3,[weights_conv1,weights_conv2,weights_conv3,weights_conv4,weights_flat1,weights_flat2,weights_output]
def predict(y_pred_class,session):

    test_batch_size=256



    num_test = len(test_x)

    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    

    

    i = 0





    while i < num_test:

           

            

        j = min(i + test_batch_size, num_test)

            

        batch_x= np.array(test_x[i:j])

        batch_x = np.reshape(batch_x,[-1,IMG_SIZE,IMG_SIZE,1])

        batch_y= np.array(test_y[i:j])

        feed_dict = {x: batch_x,y_true:batch_y}



        cls_pred[i:j] = session.run(y_pred_class, feed_dict=feed_dict)



        i = j

    correct = np.equal(cls_pred,np.argmax(test_y,1))

    accuracy = np.mean(correct.astype('float'))

    

    print("validation accuracy",accuracy)

    print(correct[:100])

    return cls_pred



            

    


def train_model():

    

    

    prediction,list_of_weights = build_network(True)

    

    

    cross_entrpy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,

                                                                               labels=y_true))

    

    

    regularizer = tf.nn.l2_loss(list_of_weights[0])+tf.nn.l2_loss(list_of_weights[1])+tf.nn.l2_loss(list_of_weights[2])+tf.nn.l2_loss(list_of_weights[3])+tf.nn.l2_loss(list_of_weights[4])+tf.nn.l2_loss(list_of_weights[5])+tf.nn.l2_loss(list_of_weights[6])       

    

    cross_entrpy_loss = tf.reduce_mean(cross_entrpy_loss + beta * regularizer)

    

    

    

    optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(cross_entrpy_loss)

    batch_size = 128





    hm_epochs = 40

    with tf.Session() as sessin:

        

        saver1=tf.train.Saver()

        best_validation=0.0

        last_improvement=0

        

        y_pred = tf.nn.softmax(prediction)

        y_pred_class = tf.argmax(y_pred,dimension=1)

    

            

        sessin.run(tf.global_variables_initializer())

        start_time = time.time()

        for epoch in tqdm(range(hm_epochs)):

            epoch_loss = 0

            batch=0

            

            while batch < len(train_x):

                start = batch

                end = batch+batch_size

                batch_x = np.array(train_x[start:end])

                batch_x = np.reshape(batch_x,[-1,IMG_SIZE,IMG_SIZE,1])

                batch_y = np.array(train_y[start:end])

                _, c = sessin.run([optimizer, cross_entrpy_loss], feed_dict={x: batch_x, y_true: batch_y})

                epoch_loss += c

                batch+=batch_size

               

                



            end_time =time.time()

            difference = end_time - start_time

            print("Time usage: " + str(timedelta(seconds=int(round(difference)))))

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_true, 1))

            

           

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            acc = sessin.run(accuracy, feed_dict={x:batch_x, y_true: batch_y})

        

               

            saver1.save(sess=sessin,save_path=save_validation_path)

                

            print("last improvment at epoch",epoch+1,"with accuracy",acc)

        predict(y_pred_class,sessin)

        

            

train_model()
print("hello")
def plot_images(images, cls_true, cls_pred=None):

  #  assert len(images) == len(cls_true) == 9

    

    # Create figure with 3x3 sub-plots.

    fig, axes = plt.subplots(3, 3)

    fig.subplots_adjust(hspace=0.3, wspace=0.3)



    for i, ax in enumerate(axes.flat):

        # Plot image.

        ax.imshow(images[i], cmap='binary')



        # Show true and predicted classes.

        if cls_pred is None:

            xlabel = "True: {0}".format(cls_true[i])

        else:

            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])



        # Show the classes as the label on the x-axis.

        ax.set_xlabel(xlabel)

        

        # Remove ticks from the plot.

        ax.set_xticks([])

        ax.set_yticks([])

    

    # Ensure the plot is shown correctly with multiple plots

    # in a single Notebook cell.

    plt.show()



prediction,list_of_weights = build_network(False)

print(prediction)

y_pred = tf.nn.softmax(prediction)







y_pred_class = tf.argmax(y_pred,dimension=1)



batch_n= np.array(test_x[5634:5643])

batch_x = np.reshape(batch_n,[-1,IMG_SIZE,IMG_SIZE,1])

batch_y=np.array(test_y[5634:5643])

batch_y = np.argmax(batch_y,1)







    

with tf.Session() as sess:  

    sess.run(tf.global_variables_initializer())



    ckpt = tf.train.get_checkpoint_state(save_path)

    if ckpt and ckpt.model_checkpoint_path:

        saver2= tf.train.import_meta_graph("{}.meta".format(ckpt.model_checkpoint_path))

        saver2.restore(sess, ckpt.model_checkpoint_path)

    else:

        print("No checkpoint")

    feed_dict = {x: batch_x}



    cls_pred = sess.run(y_pred_class, feed_dict=feed_dict)

print(cls_pred)

plot_images(batch_n,batch_y,cls_pred)



prediction,list_of_weights = build_network(False)



y_pred = tf.nn.softmax(prediction)







y_pred_class = tf.argmax(y_pred,dimension=1)



batch_n= np.array(testing_x[0:9])



batch_x = np.reshape(batch_n,[-1,IMG_SIZE,IMG_SIZE,1])

batch_y=np.array(testing_y[0:9])



print(batch_x.shape)







    

with tf.Session() as sess:  

    sess.run(tf.global_variables_initializer())



    ckpt = tf.train.get_checkpoint_state(save_path)

    if ckpt and ckpt.model_checkpoint_path:

        saver2= tf.train.import_meta_graph("{}.meta".format(ckpt.model_checkpoint_path))

        saver2.restore(sess, ckpt.model_checkpoint_path)

    else:

        print("No checkpoint")

    feed_dict = {x: batch_x}



    cls_pred = sess.run(y_pred_class, feed_dict=feed_dict)

print(cls_pred)





fig, axes = plt.subplots(3, 3)

fig.subplots_adjust(hspace=0.3, wspace=0.3)

for i, ax in enumerate(axes.flat):

        # Plot image.

    ax.imshow(batch_n[i], cmap='binary')



       

    xlabel = "True: {0} id{1}".format(cls_pred[i],batch_y[i])



        # Show the classes as the label on the x-axis.

    ax.set_xlabel(xlabel)

        

        # Remove ticks from the plot.

    ax.set_xticks([])

    ax.set_yticks([])

    

    # Ensure the plot is shown correctly with multiple plots

    # in a single Notebook cell.

plt.show()