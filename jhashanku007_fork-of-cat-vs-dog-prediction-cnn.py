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
# Any results you write to the current directory are saved as output.
print(os.listdir("../input/"))
import skimage.data as dt
from skimage import transform 
from skimage.color import rgb2gray
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
SIZE = 75
EPOCHS = 500
MAX_FILTER1 = 4 #Filter size of pooling layer1
MAX_FILTER2 = 2 #Filter size of pooling layer2
MAX_FILTER3 = 2 #Filter size of pooling layer3
FILTER1 = 2 #Filter size in convulation layer1
FILTER2 = 2 #Filter size in convulation layer2
FILTER3 = 5#Filter size in convulation layer3 
FILTER_NO1 = 100
FILTER_NO2 = 75 #No. Filter of second convolution layer
FILTER_NO3 = 25 #No. Filter of third convolution layer

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
input_shape = images.shape[1:]
input_shape
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
one_hot_labels = to_categorical(label)
"""let's check the consistency of our label """
plt.figure()
plt.hist(label,label="label of images")
plt.xlabel("Label")
plt.ylabel("count of each label")
plt.legend()
#Lets's split training data to trainig set and cross validation set
x_train,x_cross,y_train,y_cross = smodel.train_test_split(images,one_hot_labels,test_size=0.3)
def createModel():
    model = Sequential()
    model.add(Conv2D(FILTER_NO1, (FILTER1, FILTER1), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(FILTER_NO1, (FILTER1, FILTER1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(MAX_FILTER1, MAX_FILTER1)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(FILTER_NO2, (FILTER2, FILTER2), padding='same', activation='relu'))
    model.add(Conv2D(FILTER_NO2, (FILTER2, FILTER2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(MAX_FILTER2,MAX_FILTER2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(FILTER_NO3, (FILTER3, FILTER3), padding='same', activation='relu'))
    model.add(Conv2D(FILTER_NO3, (FILTER3, FILTER3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(MAX_FILTER3,MAX_FILTER3)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
     
    return model
"""Let's increase the no. of images"""
model1 = createModel()
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
 
batch_size = 256
datagen = ImageDataGenerator(
    rotation_range=10.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.,
    zoom_range=.1,
    horizontal_flip=True,
    vertical_flip=True)  # randomly flip images
 
# Fit the model on the batches generated by datagen.flow().
history2 = model1.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
                              epochs=EPOCHS,
                              validation_data=(x_cross, y_cross),
                              workers=4)
 
model1.evaluate(x_cross,y_cross)
"""Let's test our model with more generated data"""
cross_loss, cross_acc = model1.evaluate(x_cross,y_cross)
train_loss, train_acc = model1.evaluate(x_train,y_train)
predictions = np.argmax(model1.predict(x_cross),axis=1)
print("Train Accuracy",train_acc)
print("Cross Validation Accuracy",cross_acc)
# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history2.history['loss'],'r',linewidth=3.0)
plt.plot(history2.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history2.history['acc'],'r',linewidth=3.0)
plt.plot(history2.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
"""Let's check some of the image"""
sample_indexes = random.sample(range(len(x_cross)), 10)
sample_images = [x_cross[i] for i in sample_indexes]
sample_labels = [y_cross[i] for i in sample_indexes]
predicted =       [predictions[i] for i in sample_indexes]     
# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = maporiginallabel(labeldict,np.where(sample_labels[i]==1)[0][0])
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
        images[count] = transform.resize(dt.imread(image), (SIZE,SIZE))
        count += 1
    return images
test_dir = '../input/test/test'
images = making_matrices(test_dir)
images = rgb2gray(images)
predictions = np.argmax(model.predict(images),axis=1)

"""Let's see some of the predicted images"""
"""Let's check some of the image"""
sample_indexes = random.sample(range(len(images)), 10)
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