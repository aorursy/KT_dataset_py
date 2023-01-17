import pandas as pd

import numpy as np                          

import os                                   

from sklearn.metrics import confusion_matrix

import seaborn as sns                    

from sklearn.utils import shuffle           

import matplotlib.pyplot as plt

%matplotlib inline

import cv2                                  

import tensorflow as tf 

import glob as gb

import keras

from keras.models import Sequential

from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D

from keras.applications import VGG16
print(os.listdir("../input"))
#path file

trainpath = '../input/seg_train/'

testpath = '../input/seg_test/'

predpath = '../input/seg_pred/'

#6 categories that we have to classify.

class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']

class_names_label = {'mountain': 0,'street' : 1,'glacier' : 2,'buildings' : 3,'sea' : 4,'forest' : 5}

nb_classes = 6
for folder in  os.listdir(trainpath + 'seg_train') : 

    files = gb.glob(pathname= str( trainpath +'seg_train//' + folder + '/*.jpg'))

    print(f'For training data , found {len(files)} in folder {folder}')
for folder in  os.listdir(testpath +'seg_test') : 

    files = gb.glob(pathname= str( testpath +'seg_test//' + folder + '/*.jpg'))

    print(f'For testing data , found {len(files)} in folder {folder}')
def load_data():

    

    datasets = [trainpath +'seg_train', testpath +'seg_test']

    size = (150,150)

    output = []

    for dataset in datasets:

        directory = "../input/" + dataset

        images = []

        labels = []

        for folder in os.listdir(directory):

            curr_label = class_names_label[folder]

            for file in os.listdir(directory + "/" + folder):

                img_path = directory + "/" + folder + "/" + file

                curr_img = cv2.imread(img_path)

                curr_img = cv2.resize(curr_img, size)

                images.append(curr_img)

                labels.append(curr_label)

        images, labels = shuffle(images, labels)     ### Shuffle the data !!!

        images = np.array(images, dtype = 'float32') ### Our images

        labels = np.array(labels, dtype = 'int32')   ### From 0 to num_classes-1!

        

        output.append((images, labels))



    return output
(train_images, train_labels), (test_images, test_labels) = load_data()
print ("Number of training examples: " + str(train_labels.shape[0]))

print ("Number of testing examples: " + str(test_labels.shape[0]))

print ("Each image is of size: " + str(train_images.shape[1:]))
train_images = train_images / 255.0 

test_images = test_images / 255.0
index = np.random.randint(train_images.shape[0])

plt.figure()

plt.imshow(train_images[index])

plt.grid(False)

plt.title('Image #{} : '.format(index) + class_names[train_labels[index]])

plt.show()
fig = plt.figure(figsize=(14,14))

fig.suptitle("Some examples of images of the dataset", fontsize=16)

for i in range(20):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap=plt.cm.binary)

    plt.xlabel(class_names[train_labels[i]])

plt.show()
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)), # the nn will learn the good filter to use

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(6, activation=tf.nn.softmax)

])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=128, epochs=5, validation_split = 0.2)
fig = plt.figure(figsize=(10,5))

plt.subplot(221)

plt.plot(history.history['acc'],'bo--', label = "acc")

plt.plot(history.history['val_acc'], 'ro--', label = "val_acc")

plt.title("train_acc vs val_acc")

plt.ylabel("accuracy")

plt.xlabel("epochs")

plt.legend()



plt.subplot(222)

plt.plot(history.history['loss'],'bo--', label = "loss")

plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")

plt.title("train_loss vs val_loss")

plt.ylabel("loss")

plt.xlabel("epochs")





plt.legend()

plt.show()
test_loss = model.evaluate(test_images, test_labels)
score = model.evaluate(test_images, test_labels, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
index = np.random.randint(test_images.shape[0])



img = (np.expand_dims(test_images[index], 0))

predictions = model.predict(img) 

pred_img = np.argmax(predictions[0])

pred_label = class_names[pred_img]

true_label = class_names[test_labels[index]] 



plt.figure()

plt.imshow(test_images[index])

plt.grid(False)

plt.title('Predict : {} \n true : {}  '.format(pred_label , true_label ))

plt.axis('off')

plt.show()
def print_mislabeled_images(class_names, test_images, test_labels, pred_label):



    mis_img = (test_labels == pred_label)

    mislabeled_indices = np.where(mis_img == 0)

    mislabeled_images = test_images[mislabeled_indices]

    mislabeled_labels = pred_label[mislabeled_indices]

    fig = plt.figure(figsize=(15,15))

    fig.suptitle("examples of mislabeled images by classifier:", fontsize=16)

    for i in range(20):

        plt.subplot(5,5,i+1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(mislabeled_images[i], cmap=plt.cm.binary)

        plt.xlabel(class_names[mislabeled_labels[i]])

    plt.show()
predictions = model.predict(test_images)

pred_label = np.argmax(predictions, axis = 1)

print_mislabeled_images(class_names, test_images, test_labels, pred_label)
# Plotting confusion matrix

CM = confusion_matrix(test_labels, pred_label)

ax = plt.axes()

fig = plt.figure(figsize = (7, 7))

sns.heatmap(CM, annot = True,annot_kws={"size": 10},  xticklabels=class_names, yticklabels=class_names,fmt = 'd',ax = ax, square = True, cmap="YlOrBr")

ax.set_title('Confusion matrix')

plt.show()
print(CM)