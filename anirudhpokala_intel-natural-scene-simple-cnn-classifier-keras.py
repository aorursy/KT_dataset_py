# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
 #       print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle           
import matplotlib.pyplot as plt             
import cv2                                 
import tensorflow as tf                
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd

import random

from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation, Flatten, Dense, concatenate, add, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from skimage.util import img_as_ubyte
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import tensorflow.keras.utils as Utils
class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (150, 150)
class_names_label
def load_data():
    """
        Load the data:
            - 14,034 images to train the network.
            - 3,000 images to evaluate how accurately the network learned to classify images.
    """
    
    datasets = ['/kaggle/input/intel-image-classification/seg_train/seg_train', '/kaggle/input/intel-image-classification/seg_test/seg_test']
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
(train_img, train_lab), (test_images, test_labels) = load_data()

train_images,val_images,train_labels,val_labels = train_test_split(train_img,train_lab,test_size=0.2)
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

n_train = train_labels.shape[0]
n_test = test_labels.shape[0]

print ("Number of training examples: {}".format(n_train))
print ("Number of testing examples: {}".format(n_test))
print ("Each image is of size: {}".format(IMAGE_SIZE))
train_images = train_images / 255.0 
val_images = val_images /255.0
test_images = test_images / 255.0
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
display_examples(class_names, train_images, train_labels)

visible = Input(shape = (150,150,3))
layer = Conv2D(8, (5,5), padding = 'same', activation = 'relu')(visible)
layer = Conv2D(8, (3,3), padding = 'same', activation = 'relu')(layer)
layer = MaxPooling2D((2,2), strides = (2,2), padding = 'same')(layer)
layer = Conv2D(16,(5,5),padding ='same',activation= 'relu')(layer)
layer = Conv2D(16, (3,3), padding = 'same', activation = 'relu')(layer)
layer = MaxPooling2D((2,2), strides = (2,2), padding = 'same')(layer)
layer = Conv2D(32, (5,5), padding = 'same', activation = 'relu')(layer)
layer = Conv2D(32, (3,3), padding = 'same', activation = 'relu')(layer)
layer = MaxPooling2D((2,2), strides = (2,2), padding = 'same')(layer)
layer = Conv2D(64, (3,3), padding = 'same', activation = 'relu')(layer)
layer = MaxPooling2D((2,2), strides = (2,2), padding = 'same')(layer)
layer = Conv2D(128, (3,3), padding = 'same', activation = 'relu')(layer)
layer = Flatten()(layer)
layer = Dense(256, activation = 'relu')(layer)
layer = Dense(64, activation = 'relu')(layer)
output = Dense(6, activation = "softmax" )(layer)
model = Model(inputs=visible, outputs=output)
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
# Set annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
datagen = ImageDataGenerator(
        rotation_range=10, 
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
        )

datagen.fit(train_images)
history = model.fit_generator(datagen.flow(train_images,train_labels, batch_size=64),
                              epochs = 50, validation_data = (val_images,val_labels),
                              verbose = 2, steps_per_epoch=train_images.shape[0] // 64,
                              callbacks=[learning_rate_reduction]
                              )

model.save("hybrid_classifier.h5")
SVG(model_to_dot(model).create(prog='dot', format='svg'))
Utils.plot_model(model,to_file='model.png',show_shapes=True)
def plot_accuracy_loss(history):
    """
        Plot the accuracy and the loss during the training of the nn.
    """
    fig = plt.figure(figsize=(10,5))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'],'--', label = 'accuracy')
    plt.plot(history.history['val_accuracy'], '--', label = "val_accuracy")
    plt.title("train_accuracy vs val_accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'],'--', label = "loss")
    plt.plot(history.history['val_loss'], '--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()
plot_accuracy_loss(history)
model.evaluate(test_images,test_labels, verbose=1)
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
predictions = model.predict(test_images)     # Vector of probabilities
pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability

display_examples(class_names, test_images, pred_labels)
predictions = model.predict(test_images)     # Vector of probabilities
pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability

display_random_image(class_names, test_images, pred_labels)
predictions = model.predict(test_images)     # Vector of probabilities
pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability

display_random_image(class_names, test_images, pred_labels)
