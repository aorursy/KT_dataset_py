#Generic Packages

import numpy as np

import os

import pandas as pd

import random



#Machine Learning Library

from sklearn.metrics import confusion_matrix

from sklearn.utils import shuffle           



#Plotting Libraries

import seaborn as sn; sn.set(font_scale=1.4)

import matplotlib.pyplot as plt             



#openCV

import cv2                                 



#Tensor Flow

import tensorflow as tf    



#Display Progress

from tqdm import tqdm



#Garbage Collector

import gc
class_names = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

class_names_label = {class_name:i for i, class_name in enumerate(class_names)}



nb_classes = len(class_names)



IMAGE_SIZE = (150, 150)
#Function to Load Images & Labels

def load_data():

    

    datasets = ['../input/image-dataset/_train', '../input/image-dataset/_test']

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
#Loading Data (Training & Test Dataset)

(train_images, train_labels), (test_images, test_labels) = load_data()
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
#Label Dataset Shape

n_train = train_labels.shape[0]

n_test = test_labels.shape[0]



print ("Number of training examples: {}".format(n_train))

print ("Number of testing examples: {}".format(n_test))

print ("Each image is of size: {}".format(IMAGE_SIZE))
_, train_counts = np.unique(train_labels, return_counts=True)

_, test_counts = np.unique(test_labels, return_counts=True)

pd.DataFrame({'train': train_counts,'test': test_counts}, index=class_names).plot.bar(figsize=(10,8))

plt.title('Label Count Per Dataset')

plt.show()
#Scale the data

train_images = train_images / 255.0

test_images = test_images / 255.0
#Visualise the data [random image from training dataset]



def display_random_img(class_names, images, labels):

    index = np.random.randint(images.shape[0])

    plt.figure()

    plt.imshow(images[index])

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.title('Image #{} : '.format(index) + class_names[labels[index]])

    plt.show()

    



display_random_img (class_names, train_images, train_labels)
#Build Model

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(8, activation=tf.nn.softmax)

])
model.summary()
#Compile Model

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
#Training the Model

history = model.fit(train_images, train_labels, batch_size=100, epochs=10, validation_split = 0.2)
#garbage collection to save memory

gc.collect()
test_loss = model.evaluate(test_images, test_labels)
#garbage collection to save memory

gc.collect()
predictions = model.predict(test_images)     # Vector of probabilities

pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability



display_random_img(class_names, test_images, pred_labels)