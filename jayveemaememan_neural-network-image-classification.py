#import needed library

import tensorflow.keras.layers as Layers

import tensorflow.keras.activations as Actications

import tensorflow.keras.models as Models

import tensorflow.keras.optimizers as Optimizer

import tensorflow.keras.metrics as Metrics

import tensorflow.keras.utils as Utils

import os

import cv2

import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sn; sn.set(font_scale=1.4)

from keras.utils.vis_utils import model_to_dot

from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix 

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import classification_report

from random import randint

from sklearn.utils import shuffle                

from tqdm import tqdm

import matplotlib.gridspec as gridspec

%matplotlib inline
#Data Input folders

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#defining scenes to be classified using classes

class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']

class_names_label = {class_name:i for i, class_name in enumerate(class_names)}



nb_classes = len(class_names)



IMAGE_SIZE = (150, 150)
#loading of data

def load_data():



    datasets = ['../input/seg_train/seg_train', '../input/seg_test/seg_test']

    output = []

    

    # Iterate through training and test sets

    for dataset in datasets:

        

        images = []

        labels = []

        

        print("Loading {}".format(dataset))

        

        # Iterate through each folder corresponding to a category

        for folder in os.listdir(dataset):

            label = class_names_label[folder]

            

            # Iterate through each image in our folder

            for file in tqdm(os.listdir(os.path.join(dataset, folder))):

                

                # Get the path name of the image

                img_path = os.path.join(os.path.join(dataset, folder), file)

                

                # Open and resize the img

                image = cv2.imread(img_path)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = cv2.resize(image, IMAGE_SIZE) 

                

                # Append the image and its corresponding label to the output

                images.append(image)

                labels.append(label)

                

        images = np.array(images, dtype = 'float32')

        labels = np.array(labels, dtype = 'int32')   

        

        output.append((images, labels))



    return output
#load images

(train_images, train_labels), (test_images, test_labels) = load_data()
#shuffle images

train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
n_train = train_labels.shape[0]

n_test = test_labels.shape[0]



print ("Number of training examples: {}".format(n_train))

print ("Number of testing examples: {}".format(n_test))

print ("Each image is of size: {}".format(IMAGE_SIZE))
#data scaling

train_images = train_images / 255.0 

test_images = test_images / 255.0
#visualize random image from training set 

def display_examples(class_names, images, labels):

    

    fig = plt.figure(figsize=(10,10))

    fig.suptitle("Some examples of images of the dataset", fontsize=16)

    for i in range(25):

        plt.subplot(5,5,i+1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(images[i], cmap=plt.cm.binary)

        plt.xlabel(class_names[labels[i]])

    plt.show()
#Display 25 images from the images array with its corresponding labels

display_examples(class_names, train_images, train_labels)
#model used to train

#simple convolutional network

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(200, (3, 3), activation = 'relu', input_shape = (150, 150, 3)), 

    tf.keras.layers.Conv2D(180, (3, 3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(5,5),

    tf.keras.layers.Conv2D(180, (3, 3), activation = 'relu'),

    tf.keras.layers.Conv2D(100, (3, 3), activation = 'relu'),

    tf.keras.layers.Conv2D(75, (3, 3), activation = 'relu'),

    tf.keras.layers.Conv2D(50, (3, 3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(5,5),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(180, activation=tf.nn.relu),

    tf.keras.layers.Dense(100, activation=tf.nn.relu),

    tf.keras.layers.Dense(75, activation=tf.nn.relu),

    tf.keras.layers.Dense(50, activation=tf.nn.relu),

    tf.keras.layers.Dropout(rate=0.5),

    tf.keras.layers.Dense(6, activation=tf.nn.softmax)

])
#compile the model

model.compile (optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
#train or fit data into model

history = model.fit(train_images, train_labels, epochs=10, validation_split = 0.2)
def plot_accuracy_loss(history):

    """

        Plot the accuracy and the loss during the training of the nn.

    """

    fig = plt.figure(figsize=(15,15))



    # Plot accuracy

    plt.subplot(221)

    plt.plot(history.history['acc'],'bo--', label = "Train accuracy")

    plt.plot(history.history['val_acc'], 'ro--', label = "Validation accuracy")

    plt.title("Accuracy")

    plt.ylabel("Accuracy Value")

    plt.xlabel("Epoch")

    plt.legend()



    # Plot loss function

    plt.subplot(222)

    plt.plot(history.history['loss'],'bo--', label = "Train loss")

    plt.plot(history.history['val_loss'], 'ro--', label = "Validation loss")

    plt.title("Loss")

    plt.ylabel("Loss Value")

    plt.xlabel("Epoch")



    plt.legend()

    plt.show()
plot_accuracy_loss(history)
#evaluate model performace

test_loss = model.evaluate(test_images, test_labels)
def display_random_image(class_names, images, labels):

    

    index = np.random.randint(images.shape[0])

    plt.figure()

    plt.imshow(images[index])

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.title(class_names[labels[index]])

    plt.show()
#Display a random image

predictions = model.predict(test_images)     

pred_labels = np.argmax(predictions, axis = 1) 



display_random_image(class_names, test_images, pred_labels)
#error analysis

def print_mislabeled_images(class_names, test_images, test_labels, pred_labels):

    

    BOO = (test_labels == pred_labels)

    mislabeled_indices = np.where(BOO == 0)

    mislabeled_images = test_images[mislabeled_indices]

    mislabeled_labels = pred_labels[mislabeled_indices]



    title = "Some examples of mislabeled images by the classifier:"

    display_examples(class_names,  mislabeled_images, mislabeled_labels)

#25 examples of mislabeled images by the classifier, e.g when test_labels != pred_labels

print_mislabeled_images(class_names, test_images, test_labels, pred_labels)
#confusion matrix

CM = confusion_matrix(test_labels, pred_labels)

ax = plt.axes()

sn.heatmap(CM, annot=True, 

           annot_kws={"size": 10}, 

           xticklabels=class_names, 

           yticklabels=class_names, ax = ax)

ax.set_title('Confusion matrix')

plt.show()
#Feature extraction with VGG ImageNet

from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input



model = VGG16(weights='imagenet', include_top=False)
#Get features directly from VGG16

train_features = model.predict(train_images)

test_features = model.predict(test_images)
#Visualize the features through PCA

n_train, x, y, z = train_features.shape

n_test, x, y, z = test_features.shape

numFeatures = x * y * z
from sklearn import decomposition



pca = decomposition.PCA(n_components = 2)



X = train_features.reshape((n_train, x*y*z))

pca.fit(X)



C = pca.transform(X) 

C1 = C[:,0]

C2 = C[:,1]
#Plot of PCA Projection



plt.subplots(figsize=(10,10))



for i, class_name in enumerate(class_names):

    plt.scatter(C1[train_labels == i][:1000], C2[train_labels == i][:1000], label = class_name, alpha=0.4)

plt.legend()

plt.title("PCA Projection")

plt.show()
#train a simple one-layer Neural Network on the features extracted from VGG

model2 = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape = (x, y, z)),

    tf.keras.layers.Dense(50, activation=tf.nn.relu),

    tf.keras.layers.Dense(6, activation=tf.nn.softmax)

])



model2.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])



history2 = model2.fit(train_features, train_labels, epochs=15, validation_split = 0.2)
plot_accuracy_loss(history2)
test_loss = model2.evaluate(test_features, test_labels)
#Ensemble Neural Networks



np.random.seed(seed=1997)

# Number of estimators

n_estimators = 10

# Proporition of samples to use to train each training

max_samples = 0.8



max_samples *= n_train

max_samples = int(max_samples)
#Define n_estimators Neural Networks.

#Each Neural Network will be trained on random subsets of the training dataset. 

#Each subset contains max_samples samples.



models = list()

random = np.random.randint(50, 100, size = n_estimators)



for i in range(n_estimators):

    

    # Model

    model = tf.keras.Sequential([ tf.keras.layers.Flatten(input_shape = (x, y, z)),

                                # One layer with random size

                                    tf.keras.layers.Dense(random[i], activation=tf.nn.relu),

                                    tf.keras.layers.Dense(6, activation=tf.nn.softmax)

                                ])

    

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

    

    # Store model

    models.append(model)
histories = []



for i in range(n_estimators):

    # Train each model on a bag of the training data

    train_idx = np.random.choice(len(train_features), size = max_samples)

    histories.append(models[i].fit(train_features[train_idx], train_labels[train_idx], epochs=10, validation_split = 0.1))
#aggregate each model individual predictions to form a final prediction.

predictions = []

for i in range(n_estimators):

    predictions.append(models[i].predict(test_features))

    

predictions = np.array(predictions)

predictions = predictions.sum(axis = 0)

pred_labels = predictions.argmax(axis=1)
#improve result having a lower variance

from sklearn.metrics import accuracy_score

print("Accuracy : {}".format(accuracy_score(test_labels, pred_labels)))
#Fine Tuning VGG ImageNet

from keras.models import Model



model = VGG16(weights='imagenet', include_top=False)

model = Model(inputs=model.inputs, outputs=model.layers[-5].output)
train_features = model.predict(train_images)

test_features = model.predict(test_images)
from keras.layers import Input, Dense, Conv2D, Activation , MaxPooling2D, Flatten



model2 = VGG16(weights='imagenet', include_top=False)



input_shape = model2.layers[-4].get_input_shape_at(0) 

layer_input = Input(shape = (9, 9, 512)) 

# https://stackoverflow.com/questions/52800025/keras-give-input-to-intermediate-layer-and-get-final-output



x = layer_input

for layer in model2.layers[-4::1]:

    x = layer(x)

    

x = Conv2D(64, (3, 3), activation='relu')(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)

x = Dense(100,activation='relu')(x)

x = Dense(6,activation='softmax')(x)



# create the model

new_model = Model(layer_input, x)
new_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
new_model.summary()
history = new_model.fit(train_features, train_labels, epochs=10, validation_split = 0.2)
plot_accuracy_loss(history)
from sklearn.metrics import accuracy_score



predictions = new_model.predict(test_features)    

pred_labels = np.argmax(predictions, axis = 1)

print("Accuracy : {}".format(accuracy_score(test_labels, pred_labels)))