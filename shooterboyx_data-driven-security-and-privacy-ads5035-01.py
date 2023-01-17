# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import cv2

import matplotlib.pyplot as plt 

import tensorflow as tf

from tqdm import tqdm

from sklearn.utils import shuffle
# To ensure we are talking about the same thing

class_names = ['real', 'fake']

class_names_label = {class_name:i for i, class_name in enumerate(class_names)}



IMAGE_SIZE=(96,96) # <- smallest size picture found in train set
def loadData():

    datasets = ['../input/ads5035-01/train', '../input/ads5035-01/validation']

    output = []

    for dataset in datasets:

        

        images = []

        labels = []

        

        print("Loading {}".format(dataset))

        

        for folder in os.listdir(dataset):

            label = class_names_label[folder]

            

            for file in tqdm(os.listdir(os.path.join(dataset, folder))):

                

                image_path = os.path.join(os.path.join(dataset, folder), file)

                

                image = cv2.imread(image_path)



                image = cv2.resize(image, IMAGE_SIZE, interpolation = cv2.INTER_AREA)



                

                images.append(image)

                labels.append(label)

                

        image_array = np.array(images)

        label_array = np.array(labels, dtype = 'int32')

    

        output.append((image_array, label_array))

    

    return output

    

def loadTest():

    testsets = ['../input/ads5035-01/test']

    output = []

    

    for testset in testsets:

    

        images = []

        

        print("Loading {}".format(testset))

        

        for file in tqdm(os.listdir(testset)):

            

            image_path = os.path.join(testset, file)

            

            image = cv2.imread(image_path)

            image = cv2.resize(image, IMAGE_SIZE)

            

            images.append(image)

        

        image_array = np.array(images)

        output.append(image_array)

        

    return output
(train_images, train_labels), (val_images, val_labels) = loadData()

train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

val_images, val_labels = shuffle(val_images, val_labels, random_state=30)

test_images = loadTest()
def display_random_image(class_names, images, labels):

    """

        Display a random image from the images array and its correspond label from the labels array.

    """

    

    index = np.random.randint(images.shape[0])

    plt.figure()

    plt.imshow(images[index])

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.title('Image #{} : '.format(index) + class_names[labels[index]])

    plt.show()
display_random_image(class_names, train_images, train_labels)
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (96, 96, 3)), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(6, activation=tf.nn.softmax)

])



model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(train_images, train_labels, batch_size=50, epochs=5, validation_split = 0.2)
test_loss = model.evaluate(val_images, val_labels)
pred = model.predict(test_images)

pred_labels = np.argmax(pred, axis = 1)

pred_labels = [class_names[i].capitalize() for i in pred_labels]



output = pd.DataFrame({'id': range(1,len(pred)+1), 'category': pred_labels})

output.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")