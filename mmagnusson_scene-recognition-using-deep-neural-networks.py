# UTD CS 4375.0U2



import cv2

import os

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.utils import shuffle



def process_imgs(filepath):

    imgs = []

    labels = []

    curr_label = 0

    

    for x in os.listdir(filepath):

        if (x == 'buildings'):

            curr_label = 0

        elif (x == 'forest'):

            curr_label = 1

        elif (x == 'glacier'):

            curr_label = 2

        elif (x == 'mountain'):

            curr_label = 3

        elif (x == 'sea'):

            curr_label = 4

        elif (x == 'street'):

            curr_label = 5

            

        for file in os.listdir(filepath + x):

            curr_image = cv2.imread(filepath + x + r'/' + file)

            curr_image = cv2.resize(curr_image, (150, 150))

            

            # Add current image to list of images

            imgs.append(curr_image)

            # Add current label to list of labels

            labels.append(curr_label)

    

    return shuffle(imgs, labels, random_state=123)



imgs, labels = process_imgs('/kaggle/input/intel-image-classification/seg_train/seg_train/')



imgs = np.array(imgs)

labels = np.array(labels)



print("Images successfully processed.")



num_each_category = np.unique(labels, return_counts=True)

category_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']





# Pie chart for category distribution



colors = ['#6491ed', '#eb3144', '#f0de18', '#1ebda8', '#3fed2f', '#c41f67']

plt.pie(num_each_category[1], autopct='%1.1f%%', labels=category_names, colors=colors)

plt.axis('equal')



# Sample of images in the dataset



sample_plot = plt.figure(figsize=(6,6))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(imgs[i+50])

    plt.xlabel(category_names[labels[i+50]])

plt.show()



# Create neural network

nn = tf.keras.Sequential()

nn.add(tf.keras.layers.Conv2D(100, (3, 3), activation='relu', input_shape=(150,150,3)))

nn.add(tf.keras.layers.MaxPooling2D(2,2))

nn.add(tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'))

nn.add(tf.keras.layers.MaxPooling2D(2,2))

nn.add(tf.keras.layers.Flatten())

nn.add(tf.keras.layers.Dense(128, activation = 'relu'))

nn.add(tf.keras.layers.Dropout(rate = 0.5))

nn.add(tf.keras.layers.Dense(6, activation=tf.nn.softmax))



nn.summary()

nn.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss = 'sparse_categorical_crossentropy', metrics=["acc"])



print("Neural Net successfully created.")





# SELECT DESIRED EPOCHS

num_epochs = 30



# Fit the neural network

fit_net = nn.fit(imgs, labels, epochs=num_epochs, validation_split=0.2, batch_size=64)



# Create plot for training accuracy

# fig = plt.figure(figsize=(10,5))



plt.plot(fit_net.history['acc'], 'r.-', label='training accuracy')

plt.plot(fit_net.history['val_acc'], 'g.-', label='test accuracy')

plt.title('Accuracy Over Time')

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()