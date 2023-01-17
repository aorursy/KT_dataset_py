# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))
import sklearn
# Any results you write to the current directory are saved as output.
images_dir_name = '../input/stage_1_test_images/stage_1_test_images'
#print(os.listdir("../input/stage_1_test_images/stage_1_test_images"))
input_dir = '../input/'
# retrieve all the labels and store those into a collection
classes_trainable = pd.read_csv(input_dir+'classes-trainable.csv')
print(classes_trainable)
all_labels = classes_trainable['label_code']
print ('The number of unique labels is {}'.format(len(all_labels)))
# retrieve all the labels and store those into a collection
classes_trainable = pd.read_csv(input_dir+'classes-trainable.csv')
all_labels = classes_trainable['label_code']
print(all_labels)
print ('The number of unique labels is {}'.format(len(all_labels)))
# set the number of labels which will be used as an output layer size for a model
num_labels = len(all_labels)

# build the index dictionary based on the labels collection
labels_index = {label:idx for idx, label in enumerate(all_labels)}
print(labels_index)
# retrieve the list of train images (in our case we'll be using the test images just to get the model up and running)
# this will be changed to the train data set in the future.
train_image_names = [img_name[:-4] for img_name in os.listdir(images_dir_name)]
#print (train_image_names)
print ("number of training images is {}".format(len(train_image_names)))
# retrieve the list of train labels (machine labels for now; need to work on replacing the machine labels with human ones if available)
# for now I'll be using tuning labels
labels = pd.read_csv('../input/tuning_labels.csv')
print(labels)
#print(labels)
labels.head()
#print(labels.head)
train_images = []
train_labels_raw = []
for index, row in labels.iterrows():
    #print(str(index)+row[0])
    train_images.append(row[0])
    labels_raw = row[1].split(' ')
    train_labels_raw.append([labels_index[label] for label in labels_raw])
    #print(labels_raw.shape)
# do the multi-hot encoding
def multi_hot_encode(x, num_classes):
    encoded = []
    for labels in x:
        labels_encoded = np.zeros(num_classes)
        
        for item in labels:
            labels_encoded[item] = 1
            
        encoded.append(labels_encoded)
        
    encoded = np.array(encoded)
    
    return encoded
train_labels = multi_hot_encode(train_labels_raw, num_labels)
print (train_labels.shape)
from sklearn.utils import shuffle
#import tensorflow as tf
import cv2
# define the normalization logic for an image data
def normalize(x):
    return (x.astype(float) - 128)/128
# define the dimensions of the processed image
x_dim = 100
y_dim = 100
n_channels = 3
# define scaling for logic for an image data
def scale(x):
    return cv2.resize(x, (x_dim, y_dim))
# read and pre-process image
def preprocess(image_name):
    #print("IMAGE NAME")
    img = cv2.imread(image_name)
    img1=plt.imshow(img)
    print(img)
    scaled = scale(img)
    normalized = normalize(scaled)
    
    return np.array(normalized)
# prepare the collection of labels
def get_labels(image_name):
    labels = []
    
    # todo implement
    
    return labels
# build the generator for training
def generator(samples, sample_labels, batch_size=32):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples, sample_labels)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_labels = sample_labels[offset:offset+batch_size]

            images = []
            labels = []

            for i, batch_sample in enumerate(batch_samples):
                #if((images_dir_name+'/'+batch_sample)!='None')
                #print("--------------------")
                #print(images_dir_name+'/'+batch_sample+'.jpg')
                image = preprocess(images_dir_name+'/'+batch_sample+'.jpg')

                # this will be needed later once get the real data
                #image_labels = get_labels(batch_sample)

                images.append(image)
                labels.append(batch_labels[i])

            X_train = np.array(images)
            y_train = np.array(labels)
            #yield sklearn.utils.shuffle(X_train, y_train)
            return X_train,y_train
from sklearn.model_selection import train_test_split
Xtrain, Xvalid, ytrain, yvalid = train_test_split(train_images, train_labels, test_size=0.1)
#print(Xtrain)
#print(ytrain)
Xtrain,ytrain=generator(Xtrain,ytrain,batch_size=32)
#print(Xtrain)
#print(Xtrain.shape)
#print(ytrain.shape)
x = tf.placeholder("float", [None,30000])

W = tf.Variable(tf.zeros([30000,8000]))
b = tf.Variable(tf.zeros([8000]))
W1 = tf.Variable(tf.zeros([8000,7178]))
b1 = tf.Variable(tf.zeros([7178]))                       
y = tf.nn.relu(tf.matmul(x,W) + b)
#y2 = tf.nn.relu(tf.matmul(y,W2) + b)
y1 = tf.nn.sigmoid(tf.matmul(y,W1) + b1)
y1_ = tf.placeholder("float", [None,7178])

#cross_entropy = -tf.reduce_sum(y1_*tf.log(y1))
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y1,labels=y1_)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
saver = tf.train.Saver([W,b,W1,b1])
#Train our modelSS
print("----")
iter = 10
#correct_prediction=tf.equal(tf.argmax(y1,1), tf.argmax(y1_,1))
correct_prediction=tf.equal(tf.round(y1), y1_)
req_list=[correct_prediction, train_step]
for i in range(iter):
  #print(i)
  Xtrain=np.reshape(Xtrain,(32,30000))
  #batch_xs, batch_ys = Xtrain.train.next_batch(100)
  list=sess.run(req_list, feed_dict={x: Xtrain, y1_: ytrain})
  correct_prediction, train_step=list
  print(correct_prediction.shape)
  saver.save(sess,"./tenIrisSave/saveOne")
  
#Evaluationg our model:
#correct_prediction=tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
saver.restore(sess, "./tenIrisSave/saveOne")
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
print ("Accuracy: ", sess.run(accuracy, feed_dict={x: Xtrain, y1_: ytrain}))


