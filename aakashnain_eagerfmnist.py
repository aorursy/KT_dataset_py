# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import glob
from pathlib import Path
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution() #enable the eager mode before doing any operation

from keras.preprocessing import image
from skimage.io import imread, imsave, imshow
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

np.random.seed(111)
color = sns.color_palette()
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#Read the train and test csv first
train = pd.read_csv('../input/fashion-mnist_train.csv')
test = pd.read_csv('../input/fashion-mnist_test.csv')

print("Number of training samples: ", len(train))
print("Number of test samples: ", len(test))
# Let's look at how the train dataset looks like
train.sample(10)
# Random samples from test data
test.sample(10)
# Get the count for each label
label_count = train["label"].value_counts()

# Get total number of samples
total_samples = len(train)

# Make a dictionary for all the labels. 
labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
         5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

for i in range(len(label_count)):
    label = labels[label_count.index[i]]
    count = label_count.values[i]
    pct = (count / total_samples) * 100
    print("{:<15s}:   {} or {}%".format(label, count, pct))
# An empty list to collect some samples
sample_images = []

# Iterate over the keys of the labels dictionary defined in the above cell
for k in labels.keys():
    # Get two samples for each category
    samples = train[train["label"] == k].head(2)
    # Append the samples to the samples list
    for j, s in enumerate(samples.values):
        # First column contain labels, hence index should start from 1
        img = np.array(samples.iloc[j, 1:]).reshape(28,28)
        sample_images.append(img)
        
print("Total number of sample images to plot: ", len(sample_images))
# Plot the sample images now
f, ax = plt.subplots(5,4, figsize=(15,10))

for i, img in enumerate(sample_images):
    ax[i//4, i%4].imshow(img, cmap='gray')
    ax[i//4, i%4].axis('off')
plt.show()    
# Separate the labels from train and test dataframe
tr_labels = train["label"]
ts_labels = test["label"]

# Drop the labels column from train dataframe as well as test dataframe
train = train.drop(["label"], axis =1)
test = test.drop(["label"], axis=1)

# Split the training dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(train, tr_labels, test_size=0.2, random_state=111)
print("Number of samples in the train set: ", len(X_train))
print("Number of samples in the validation set: ", len(X_valid))

# Just a consistency check
print("Train and validation shapes: ", end=" ")
print(X_train.shape,y_train.shape,X_valid.shape, y_valid.shape)
# Reshape the data values
X_train = np.array(X_train.iloc[:, :]).reshape(len(X_train),28,28,1)
X_valid = np.array(X_valid.iloc[:, :]).reshape(len(X_valid), 28, 28,1)
X_test = np.array(test.iloc[:,:]).reshape(len(test), 28, 28,1)

# Some more preprocessing
X_train = X_train.astype(np.float32)
X_valid = X_valid.astype(np.float32)
X_test = X_test.astype(np.float32)

train_mean = X_train.mean()

# Mean subtraction from pixels
X_train -= train_mean
X_valid -= train_mean
X_test -= train_mean

# Normalization
X_train /=255.
X_valid /=255.
X_test /=255.

# One Hot Encoding(OHE)
y_train = to_categorical(y_train, num_classes=10).astype(np.int8)
y_valid = to_categorical(y_valid, num_classes=10).astype(np.int8)

print("X_train shape: {}, y_train shape: {} ".format(X_train.shape, y_train.shape))
print("X_valid shape: {}, y_valid shape: {} ".format(X_valid.shape, y_valid.shape))
print("X_test shape: ", X_test.shape)
# A simple data generator
def data_gen(data, labels, batch_size=8):
    # Get total number of samples in the data
    n = len(data)
    
    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, 28, 28, 1), dtype=np.float32)
    batch_labels = np.zeros((batch_size,10), dtype=np.int8)
    
    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)
    
    # Initialize a counter
    i =0
    while True:
        np.random.shuffle(indices)
        # Get the next batch 
        next_batch = indices[(i*batch_size):(i+1)*batch_size]
        for j, idx in enumerate(next_batch):
            batch_data[j] = data[idx]
            batch_labels[j] = labels[idx]
        
        yield batch_data, batch_labels
        i +=1  
# Class represnting our model
class FMNIST(object):
    def __init__(self, data_format):
        # Set the input shape according to the availability of GPU 
        if data_format == 'channels_first':
            self._input_shape = [-1, 1, 28, 28]
        else:
            self._input_shape = [-1, 28, 28, 1]
        
        # Start defining the type of layers that you want in your network
        self.conv1 = tf.layers.Conv2D(32, 3, 
                                      activation=tf.nn.relu, 
                                      padding='same', 
                                      data_format=data_format)
        
        self.maxpool = tf.layers.MaxPooling2D((2,2), (2,2), 
                                            padding='same', 
                                            data_format=data_format)
        
        self.conv2 = tf.layers.Conv2D(64, 3, 
                                      activation=tf.nn.relu, 
                                      padding='same', 
                                      data_format=data_format)
        self.conv3 = tf.layers.Conv2D(128, 3, 
                                      activation=tf.nn.relu, 
                                      padding='same', 
                                      data_format=data_format)
        
        self.dense1 = tf.layers.Dense(1024, activation=tf.nn.relu)
        self.dense2 = tf.layers.Dense(512, activation=tf.nn.relu)
        self.dropout = tf.layers.Dropout(0.5)
        self.dense3 = tf.layers.Dense(10)
        
        
    #Combine the layers to form the architecture
    def predict(self, inputs, drop=False):
        x = tf.reshape(inputs, self._input_shape)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = tf.layers.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=drop) #enable at training and disable at testing
        x = self.dense2(x)
        x = self.dropout(x, training=drop)
        x = self.dense3(x)
        return x
# There are 10 categories, hence we will be using the cross-entropy loss here 
def loss(model, inputs, targets, drop=False):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                          logits=model.predict(inputs, drop=drop), labels=targets))
# In our case, accuracy will be the metric that we are going to use for evaluation
def compute_accuracy(predictions, labels):
    model_pred = tf.argmax(predictions, axis=1,output_type=tf.int64)
    actual_labels = tf.argmax(labels, axis=1, output_type=tf.int64)
    return tf.reduce_sum(tf.cast(tf.equal(model_pred, actual_labels),dtype=tf.float32)) / float(predictions.shape[0].value)
# Device selection
device = "gpu:0" if tfe.num_gpus() else "cpu:0"

# Get an instance of your model
model = FMNIST('channels_first' if tfe.num_gpus() else 'channels_last')

# Define an optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

# Automatic gradient calculation
grad = tfe.implicit_gradients(loss)
# Define a batch size
batch_size = 32

# Train data generator
train_data_gen = data_gen(X_train, y_train)
# Validation data generator
valid_data_gen = data_gen(X_valid, y_valid)

# Get the number of batches
nb_tr_batches = len(X_train) // batch_size
nb_val_batches = len(X_valid) // batch_size
print("Number of train and validation batches: {}, {}".format(nb_tr_batches, nb_val_batches))

# Define number of epochs for which you want to train your model
nb_epochs = 4

# Train and validate
for i in range(nb_epochs):
    print("\n========== Epoch: {} ==============================\n".format(i+1))
    with tf.device(device):
        epoch_avg_loss = []
        epoch_avg_acc = []
        for j in range(nb_tr_batches):
            inputs, targets = next(train_data_gen)
            optimizer.apply_gradients(grad(model, inputs, targets))
            if j % 500 == 0:
                batch_loss = loss(model, inputs, targets, drop=True).numpy()
                batch_acc = (compute_accuracy(model.predict(inputs, drop=True), targets).numpy())*100
                epoch_avg_loss.append(batch_loss)
                epoch_avg_acc.append(batch_acc)
                print("Step {:<5s} ------> Loss: {:.4f}".format(str(j), batch_loss))
        
        val_loss = loss(model, X_valid, y_valid).numpy()
        val_acc = (compute_accuracy(model.predict(X_valid), y_valid).numpy())*100
        print("\ntrain_loss: {:.4f}  train_acc: {:.2f}%".format(np.mean(epoch_avg_loss), np.mean(epoch_avg_acc)))        
        print("val_loss: {:.4f}   val_acc: {:.2f}%".format(val_loss, val_acc))
test_samples = X_test[:10]
test_labels = ts_labels[:10]
print(test_samples.shape, test_labels.shape)
prob = model.predict(inputs=test_samples).numpy()
predicted_labels = list(tf.argmax(prob, axis=1,output_type=tf.int64).numpy())
print("True labels:     ", test_labels.tolist())
print("Predicted labels: ", predicted_labels)
# Let's visualize the results
f, ax = plt.subplots(2,5, figsize=(20,5))

for i in range(10):
    img = X_test[i].reshape(28,28)
    ax[i//5, i%5].imshow(img)
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_title(labels[predicted_labels[i]])
plt.show()    
