import os

import cv2 

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.image as mpimg 

import numpy as np

import tensorflow as tf

from tensorflow.python.framework import ops

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
directory_list = list()

for root, dirs, files in os.walk("images/Images", topdown=False):

    for name in dirs:

        directory_list.append(os.path.join(root, name))



print (directory_list[:10])
data = pd.DataFrame(columns=['Class','ImageDir'])



for directory_ubication in directory_list:

    for root,dirs,files in os.walk(directory_ubication,topdown=False):

        for file in files:

            image_class = directory_ubication.split('\\')[1].split('-')[1]

            image_ubication = directory_ubication + '/' + file

            image_ubication = image_ubication.replace('\\','/')

            data = data.append({'Class':image_class,'ImageDir':image_ubication}, ignore_index=True)
data.head(3)
img = cv2.imread(data['ImageDir'][0]) 

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

plt.figure(figsize = (10,10))

plt.imshow(img);
data = shuffle(data)
hash_classes_encodes = {}

idx = 0

for label in data['Class'].unique().tolist():

    hash_classes_encodes[label]= idx

    idx+=1
data['Class']= [hash_classes_encodes[x] for x in data['Class']]
data.head(3)


def get_batch_data(batch_size,data):

    image_data_random = data.sample(batch_size)

    images_data = pd.DataFrame(columns=['Image_Data','Label'])

    width = 32

    height = 32

    dim = (width, height)



    for image in image_data_random.iterrows():

        image_data = cv2.imread(image[1]['ImageDir'])

        image_data = cv2.resize(image_data,dim,interpolation = cv2.INTER_AREA)

        image_label = image[1]['Class']

        images_data = images_data.append({'Image_Data':image_data,'Label':image_label}, ignore_index=True)

        

    return images_data
x_train, x_test = train_test_split(data,test_size = 0.3, random_state=0)
batch_size = 100

learning_rate = 0.005

evaluation_size = 500

image_width = 32#px

image_height = 32#px

target_size = np.amax(data['Class'].unique()) + 1   

num_chanels = 3 # rgb

generations = 20000

eval_every = 200

conv1_features = 60

conv2_features = 10

max_pool_size1 = 2

max_pool_size2 = 2

full_connected_size1 =200
# Inputs

x_input_shape = (batch_size, image_width, image_height, num_chanels)

x_input = tf.placeholder(tf.float32, shape = x_input_shape)

y_target = tf.placeholder(tf.int32, shape=(batch_size))



# Evaluation

eval_input_shape = (evaluation_size, image_width, image_height, num_chanels)

eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)

eval_target = tf.placeholder(tf.float32, shape = (evaluation_size))
# 4,4 matrix

conv1_weight = tf.Variable(tf.truncated_normal([4,4,num_chanels, conv1_features], stddev=0.1, dtype=tf.float32))

conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))

                    

conv2_weight = tf.Variable(tf.truncated_normal([4,4,conv1_features, conv2_features], stddev=0.1, dtype=tf.float32))

conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))
resulting_width = image_width // (max_pool_size1*max_pool_size2)

resulting_height = image_height // (max_pool_size1 * max_pool_size2)



#Input for first layer

full1_input_size = resulting_width*resulting_height*conv2_features

# First Layer

full1_weight = tf.Variable(tf.truncated_normal([full1_input_size, full_connected_size1], stddev=0.1, dtype=tf.float32))

full1_bias = tf.Variable(tf.truncated_normal([full_connected_size1], stddev=0.1, dtype = tf.float32))

# Second Layer 

full2_weight = tf.Variable(tf.truncated_normal([full_connected_size1, target_size], stddev=0.1, dtype=tf.float32))

full2_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))
def conv_neural_net(input_data):

    ## First Layer Conv+ReLU+Maxpool

    conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1,1,1,1], padding="SAME")

    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))

    max_pool1 = tf.nn.max_pool(relu1, ksize=[1,max_pool_size1, max_pool_size1,1], 

                               strides=[1, max_pool_size1, max_pool_size1,1], padding="SAME")

    

    ## Second Layer Conv+ReLU+Maxpool

    conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1,1,1,1], padding="SAME")

    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))

    max_pool2 = tf.nn.max_pool(relu2, ksize=[1,max_pool_size2, max_pool_size2,1], 

                               strides=[1, max_pool_size2, max_pool_size2,1], padding="SAME")

    

    ## Flattening operation to convert image to vector

    final_conv_shape = max_pool2.get_shape().as_list()

    final_shape = final_conv_shape[1]*final_conv_shape[2]*final_conv_shape[3]

    flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])

    

    ## Third Layer (First full connected)

    fully_connected_1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))

    

    ## Fourth Layer, (Second full connected)

    fully_connected_2 = tf.add(tf.matmul(fully_connected_1, full2_weight), full2_bias)

    

    return fully_connected_2

model_ouput = conv_neural_net(x_input)

test_model_output = conv_neural_net(eval_input)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_ouput, labels = y_target))
prediction = tf.nn.softmax(model_ouput)

test_prediction = tf.nn.softmax(test_model_output)



def get_accuracy(logits, targets):

    batch_predictions = np.argmax(logits, axis = 1)

    num_corrects = np.sum(np.equal(batch_predictions, targets))

    return 100.0*num_corrects/batch_predictions.shape[0]
my_optim = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)

train_step = my_optim.minimize(loss)
init = tf.global_variables_initializer()

session = tf.Session()

session.run(init)
train_loss = []

train_acc = []

test_acc = []

i_vals = []

for i in range(generations):

    

    train_images = get_batch_data(batch_size,x_train)

    rand_x = train_images['Image_Data'].values.tolist()

    rand_y = train_images['Label'].values.tolist()

    

    train_dict = {x_input:rand_x, y_target:rand_y}

    session.run(train_step, feed_dict=train_dict)

    temp_train_loss, temp_train_preds = session.run([loss, prediction], feed_dict=train_dict)

    temp_train_acc = get_accuracy(temp_train_preds, rand_y)

    

    if(i+1) % eval_every == 0:

        test_images = get_batch_data(evaluation_size,x_test)

        rand_x_eval = test_images['Image_Data'].values.tolist()

        rand_y_eval = test_images['Label'].values.tolist()

        test_dict = {eval_input:rand_x_eval, eval_target:rand_y_eval}



        temp_test_preds = session.run( test_prediction, feed_dict=test_dict) 

        temp_test_acc = get_accuracy(temp_test_preds, rand_y_eval)

        

        i_vals.append(i+1)

        train_loss.append(temp_train_loss)

        train_acc.append(temp_train_acc)

        test_acc.append(temp_test_acc)

 

        acc_and_loss = [(i+1),temp_train_loss, temp_train_acc, temp_test_acc]

        acc_and_loss = [np.round(x,3) for x in acc_and_loss]

        print("Iteración {}. Train Loss: {:.3f}. Train Acc: {:.3f}. Test Acc: {:.3f}".format(*acc_and_loss))
plt.plot(i_vals, train_loss, 'k-')

plt.title("Softmax Loss para cada Iteración")

plt.xlabel("Iteración")

plt.ylabel("Pérdida Softmax")

plt.show()
plt.plot(i_vals, train_acc, 'r-', label="Precisión en entrenamiento")

plt.plot(i_vals, test_acc, 'b--', label="Precisión en testing")

plt.xlabel("Iteración")

plt.ylabel("Precisión")

plt.ylim([0,100])

plt.title("Precisión en la predicción")

plt.legend(loc="lower right")

plt.show()
x_train