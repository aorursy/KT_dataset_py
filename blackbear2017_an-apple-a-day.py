# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.image import imread

%matplotlib inline

import os
my_data_dir = '/kaggle/input/fruit-recognition/'
os.listdir(my_data_dir)
class_names = ['Carambola', 'Pear', 'Plum', 'Pomegranate', 'Tomatoes', 'muskmelon']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}



nb_classes = len(class_names)



IMAGE_SIZE = (150, 150)
class_names_set = set(class_names)
import cv2

def load_data():

    """

        Load the data:

            - 14,034 images to train the network.

            - 3,000 images to evaluate how accurately the network learned to classify images.

    """

    

    #datasets = ['../input/seg_train/seg_train', '../input/seg_test/seg_test']

    output = []

    

    # Iterate through training and test sets

    

    images = []

    labels = []

    for root, dirs, files in os.walk(my_data_dir):

        for drir in dirs:

            for file in os.listdir(os.path.join(root, drir)):

                if file.endswith(".png") and drir in class_names_set:

                    img_path = os.path.join(my_data_dir, drir, file)

                    curr_label = class_names_label[drir]

                    # Open and resize the img

                    curr_img = cv2.imread(img_path)

                    try:

                        curr_img = cv2.resize(curr_img, IMAGE_SIZE)

                        # Append the image and its corresponding label to the output

                        images.append(curr_img)

                        labels.append(curr_label)

                    except:

                        print("Error on {}".format(img_path))

                

    images = np.stack(images).astype(np.float32)

    labels = np.stack(labels).astype(np.uint8)   



    return images, labels
images, labels = load_data()
from sklearn.model_selection import train_test_split
images.shape
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)
print ("Number of training examples: " + str(X_train.shape[0]))

print ("Number of testing examples: " + str(X_test.shape[0]))

print ("Each image is of size: " + str(X_train.shape[1:]))
len(class_names)
class_names
# Plot a pie chart

sizes = np.bincount(y_train)

explode = (0, 0, 0, 0, 0, 0)  

plt.pie(sizes, explode=explode, labels=class_names,

autopct='%1.1f%%', shadow=True, startangle=150)

plt.axis('equal')

plt.title('Proportion of each observed category')



plt.show()
X_train = X_train / 255.0 

X_test = X_test / 255.0
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
display_random_image(class_names, X_train, y_train)
def display_examples(class_names, images, labels):

    """

        Display 25 images from the images array with its corresponding labels

    """

    

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
display_examples(class_names, X_train, y_train)
import keras

from keras import layers
model = keras.Sequential()



model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(150,150,3)))

model.add(layers.AveragePooling2D())



model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))

model.add(layers.AveragePooling2D())



model.add(layers.Flatten())



model.add(layers.Dense(units=120, activation='relu'))



model.add(layers.Dense(units=84, activation='relu'))



model.add(layers.Dense(units=len(class_names), activation = 'softmax'))
model.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split = 0.2)
def plot_accuracy_loss(history):

    """

        Plot the accuracy and the loss during the training of the nn.

    """

    fig = plt.figure(figsize=(10,5))



    # Plot accuracy

    plt.subplot(221)

    plt.plot(history.history['accuracy'],'bo--', label = "accuracy")

    plt.plot(history.history['val_accuracy'], 'ro--', label = "val_accuracy")

    plt.title("train_acc vs val_acc")

    plt.ylabel("accuracy")

    plt.xlabel("epochs")

    plt.legend()



    # Plot loss function

    plt.subplot(222)

    plt.plot(history.history['loss'],'bo--', label = "loss")

    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")

    plt.title("train_loss vs val_loss")

    plt.ylabel("loss")

    plt.xlabel("epochs")



    plt.legend()

    plt.show()
plot_accuracy_loss(history)
test_loss = model.evaluate(X_test, y_test)
test_loss
predictions = model.predict(X_test)     # Vector of probabilities

pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability



display_random_image(class_names, X_test, pred_labels)