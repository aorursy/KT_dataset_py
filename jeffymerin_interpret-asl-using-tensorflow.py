# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
import tensorflow as tf
from tqdm import tqdm
import os
import numpy as np

train_dir = "../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/"
test_dir =  "../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/"
IMG_SIZE = 50
labels_map = {'A':0,'B':1,'C': 2, 'D': 3, 'E':4,'F':5,'G':6, 'H': 7, 'I':8, 'J':9,'K':10,'L':11, 'M': 12, 'N': 13, 'O':14, 
                'P':15,'Q':16, 'R': 17, 'S': 18, 'T':19, 'U':20,'V':21, 'W': 22, 'X': 23, 'Y':24, 'Z':25, 
                'del': 26, 'nothing': 27,'space':28}
def create_train_data():
    x_train = []
    y_train = []
    for folder_name in os.listdir(train_dir):
        label = labels_map[folder_name]
        for image_filename in tqdm(os.listdir(train_dir + folder_name)):
            path = os.path.join(train_dir,folder_name,image_filename)
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),(IMG_SIZE, IMG_SIZE ))
            x_train.append(np.array(img))
            y_train.append(np.array(label))
    print("Done creating train data")
    return x_train, y_train

def create_test_data():
    x_test = []
    y_test = []
    for folder_name in os.listdir(test_dir):
        label = folder_name.replace("_test.jpg","")
        label = labels_map[label]
        path = os.path.join(test_dir,folder_name)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),(IMG_SIZE, IMG_SIZE ))
        x_test.append(np.array(img))
        y_test.append(np.array(label))
    print("Done creating test data")
    return x_test,y_test
x_train, y_train= create_train_data()  
x_test,y_test = create_test_data()
num_features = 2500
num_classes = 29

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
x_train, x_test = x_train / 255., x_test / 255.
%matplotlib inline
import matplotlib.pyplot as plt

def display_image(num):
    label = y_train[num]
    plt.title('Label: %d' % (label))
    image = x_train[num].reshape([IMG_SIZE,IMG_SIZE])
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
display_image(5)
# Training parameters.
learning_rate = 0.001
training_steps = 5000
batch_size = 250
display_step = 500

# Network parameters.
n_hidden =  300# Number of neurons.
# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(87000).batch(batch_size).prefetch(1)
# Store layers weight & bias

# A random value generator to initialize weights initially
random_normal = tf.initializers.RandomNormal()

weights = {
    'h1': tf.Variable(random_normal([num_features, n_hidden])),
    'h2': tf.Variable(random_normal([n_hidden, n_hidden])),
    'out': tf.Variable(random_normal([n_hidden, num_classes]))
}
biases = {
    'b': tf.Variable(tf.zeros([n_hidden])),
    'out': tf.Variable(tf.zeros([num_classes]))
}
def neural_nets(input_data):
    hidden_layer1 = tf.add(tf.matmul(input_data,weights['h1']),biases['b'])
    hidden_layer1 = tf.nn.sigmoid(hidden_layer1)
    
    hidden_layer2 = tf.add(tf.matmul(hidden_layer1,weights['h2']),biases['b'])
    hidden_layer2 = tf.nn.sigmoid(hidden_layer2)
    
    out_layer = tf.add(tf.matmul(hidden_layer1,weights['out']),biases['out'])
    
    return tf.nn.softmax(out_layer)
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))
optimizer = tf.keras.optimizers.SGD(learning_rate)

def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = neural_nets(x)
        loss = cross_entropy(pred, y)
        
    # Variables to update, i.e. trainable variables.
    trainable_variables = list(weights.values()) + list(biases.values())

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    #print("argmax:",tf.argmax(y_pred,1))
    #print("cast",tf.cast(y_true, tf.int64))
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)
# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)
    
    if step % display_step == 0:
        pred = neural_nets(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("Training epoch: %i, Loss: %f, Accuracy: %f" % (step, loss, acc))
# Test model on validation set.
pred = neural_nets(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))
def get_key(val):
    for key, value in labels_map.items(): 
         if val == value: 
            return key 
n_images = 28
predictions = neural_nets(x_test)
for i in range(n_images):
    model_prediction = np.argmax(predictions.numpy()[i])
    plt.imshow(np.reshape(x_test[i], [50, 50]), cmap='gray_r')
    plt.show()
    print("Original Labels: %s" % get_key(y_test[i]))
    print("Model prediction: %s" % get_key(model_prediction))