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

import pandas as pd

import os

%matplotlib inline

import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm

from sklearn.metrics import confusion_matrix

import seaborn as sn; sn.set(font_scale=1.4)

from sklearn.utils import shuffle           

class_names = ['buildings','forest','glacier','mountain', 'sea','street', ]

class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

print(nb_classes)

IMAGE_SIZE = (150, 150)
def load_data():

    """

        Load the data:

            - 14,034 images to train the network.

            - 3,000 images to evaluate how accurately the network learned to classify images.

    """

    

    datasets = ['../input/intel-image-classification/seg_train/seg_train', '../input/intel-image-classification/seg_test/seg_test']

    #datasets = ['C:/Users/dines/Desktop/practice datasets/Imageclassification/seg_train/seg_train' ,'C:/Users/dines/Desktop/practice datasets/Imageclassification/seg_test/seg_test']

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
(train_images, train_labels), (test_images, test_labels) = load_data()
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
#test_images, test_images = shuffle(test_images, test_labels, random_state = 23)
train_images = train_images/255

test_images = test_images/255
from keras.models import Sequential

from keras.layers import core

from keras.layers import convolutional, pooling

from keras.utils import np_utils

model = Sequential()

model.add(convolutional.Conv2D(32, (3, 3), activation='relu', input_shape=(150,150,3)))

model.add(pooling.MaxPooling2D(pool_size=(2, 2)))

model.add(convolutional.Conv2D(32, (3, 3), activation='relu'))

model.add(pooling.MaxPooling2D(pool_size=(2, 2)))

model.add(core.Dropout(0.25))

model.add(core.Flatten())

model.add(core.Dense(128, activation='relu'))

model.add(core.Dropout(0.5))

model.add(core.Dense(6, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(train_images, train_labels, batch_size=128, epochs=20, verbose=2,validation_split=0.25)
score = model.evaluate(train_images, train_labels, verbose=0)

print(score)
predictions = model.predict(test_images)
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
pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability



for i in range(1,11):

    display_random_image(class_names, test_images, pred_labels)

test_score = model.evaluate(test_images, test_labels)

print(test_score)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

cm = confusion_matrix(test_labels, pred_labels)

print("Accuracy on test is:",accuracy_score(test_labels,pred_labels))
