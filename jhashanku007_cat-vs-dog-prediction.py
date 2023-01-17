# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#This is helpful in visualising matplotlib graphs
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow import keras #The deep learning model we will use to train our dataset will make use of this
import tensorflow as tf
from PIL import Image as IMG #To read the image file
import os #To move through the folders and fetching the images
import matplotlib.pyplot as plt #To render Plots of our data
import sklearn.model_selection as smodel #To split the data for training and cross validation set
# Any results you write to the current directory are saved as output.
from keras.utils import to_categorical
# Any results you write to the current directory are saved as output.
print(os.listdir("../input/"))
import skimage.data as dt
from skimage import transform 
from skimage.color import rgb2gray
import random
NODES = 1000
SIZE = 100
EPOCHS = 1000

# Any results you write to the current directory are saved as output.
def making_matrices(root_dir):
    count = 0
    imagefiles = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
    images = np.zeros((len(imagefiles),SIZE,SIZE,3))
    labels = np.zeros((len(imagefiles)),dtype='S140')
    for image in imagefiles:
        images[count] = transform.resize(dt.imread(image), (SIZE,SIZE))
        labels[count] = image.split('/')[-1].split('.')[-3]
        count += 1
    return images,labels
rootdir = '../input/train/train/'
images,label = making_matrices(rootdir)
#Let's Visualize some images 
plt.figure(figsize=(10,10))
rand = np.random.randint(1,1000,25)
for i in range(len(rand)):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(images[rand[i]], cmap=plt.cm.binary)
    plt.xlabel(label[rand[i]])
"""The color in the pictures matters less when we're trying to answer 
a classification question. That’s why we'll also go through the trouble of converting the images to grayscale."""
images = rgb2gray(images)
"""Let's Plot the gray scale image"""
plt.figure(figsize=(10,10))
rand = np.random.randint(1,1000,25)
for i in range(len(rand)):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(images[rand[i]], cmap=plt.cm.binary)
    plt.xlabel(label[rand[i]])
"""The images have been sucessfully converted to grey scale let's check for biasedness in label if one label is more than others"""
plt.figure()
plt.hist(label,label="label of images")
plt.xlabel("Label")
plt.ylabel("count of each label")
plt.legend()
"""Since both label are equal hence there is no bias in label
But the label are string let's change them to integer to do that let's make a dictionary with string label as key and a int as value """
labeldict = {}
for i in range(len(np.unique(label)[:])):
    labeldict[np.unique(label)[i]] = i

def yvectorize(dict,data):
    '''This will assign the numeric label to each string label in the label matrix'''
    return dict[data]
vect = np.vectorize(yvectorize)
label = vect(labeldict,label)


def maporiginallabel(dic,data):
    """This will reverse map the label i.e given a int label it will 
    return the original string label"""
    for key, value in dic.items():    # for name, age in list.items():  (for Python 3.x)
        if(value == data):
            return (key)
# Change the labels from integer to categorical data
labels_one_hot = to_categorical(label)
print('Original label 0 : ', label[0])
print('After conversion to categorical ( one-hot ) : ',labels_one_hot[0])
#Lets's split training data to trainig set and cross validation set
x_train,x_cross,y_train,y_cross = smodel.train_test_split(images,labels_one_hot,test_size=0.3)
"""Let's create and train our model"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(SIZE, SIZE)),
    keras.layers.Dense(NODES, activation=tf.nn.relu),
    keras.layers.Dense(500, activation=tf.nn.relu),
    keras.layers.Dense(50, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train,y_train, batch_size=256, epochs=EPOCHS, verbose=1, 
                   validation_data=(x_cross,y_cross))

"""Let's test our model"""
cross_loss, cross_acc = model.evaluate(x_cross,y_cross)
train_loss, train_acc = model.evaluate(x_train,y_train)
predictions = np.argmax(model.predict(x_cross),axis=1)
print("Train Accuracy",train_acc)
print("Cross Validation Accuracy",cross_acc)
#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
"""let's do it with regularisation"""
"""Let's create and train our model"""
model_reg = keras.Sequential([
    keras.layers.Flatten(input_shape=(SIZE, SIZE)),
    keras.layers.Dense(NODES, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(500, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(50, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])
model_reg.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train,y_train, batch_size=256, epochs=EPOCHS, verbose=1, 
                   validation_data=(x_cross,y_cross))

"""Let's test our model"""
cross_loss, cross_acc = model_reg.evaluate(x_cross,y_cross)
train_loss, train_acc = model_reg.evaluate(x_train,y_train)
predictions = np.argmax(model_reg.predict(x_cross),axis=1)
print("Train Accuracy",train_acc)
print("Cross Validation Accuracy",cross_acc)
#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
"""Let's check some of the image"""
sample_indexes = random.sample(range(len(x_cross)), 10)
sample_images = [x_cross[i] for i in sample_indexes]
sample_labels = [np.where(y_cross[i]==1)[0][0] for i in sample_indexes]
predicted =       [predictions[i] for i in sample_indexes]     
# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = maporiginallabel(labeldict,sample_labels[i])
    prediction = maporiginallabel(labeldict,predicted[i])
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i],  cmap="gray")

plt.show()
def making_matrices(root_dir):
    count = 0
    imagefiles = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
    images = np.zeros((len(imagefiles),SIZE,SIZE,3))
    for image in imagefiles:
        images[count] = transform.resize(dt.imread(image), (50,50))
        count += 1
    return images
test_dir = '../input/test/test'
images = making_matrices(test_dir)
images = rgb2gray(images)
predictions = np.argmax(model.predict(images),axis=1)

"""Let's see some of the predicted images"""
"""Let's check some of the image"""
sample_indexes = np.random.randint(1,7000,10)
sample_images = [images[i] for i in sample_indexes]
predicted =       [predictions[i] for i in sample_indexes]     
# Print the real and predicted labels
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    prediction = maporiginallabel(labeldict,predicted[i])
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    plt.imshow(sample_images[i],  cmap="gray")
    plt.xlabel(prediction)

plt.show()