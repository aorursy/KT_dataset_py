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
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
print(os.listdir("../input"))
os.listdir("../input/isl-dataset-double-handed")
train_dir = '../input/isl-dataset-double-handed/ISL_Dataset'
def load_unique():
    size_img = 224,224 
    images_for_plot = []
    labels_for_plot = []
    for folder in os.listdir(train_dir):
        k_t=os.listdir(train_dir + '/' + folder)
        i_t=np.random.randint(low=1, high=len(k_t), size=1)[0]
        file=os.listdir(train_dir + '/' + folder)[i_t]
        filepath = train_dir + '/' + folder + '/' + file
        image = cv2.imread(filepath)
        final_img = cv2.resize(image, size_img)
        final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        images_for_plot.append(final_img)
        labels_for_plot.append(folder)
    return images_for_plot, labels_for_plot

images_for_plot, labels_for_plot = load_unique()
print("unique_labels = ", labels_for_plot)

fig = plt.figure(figsize = (15,15))
def plot_images(fig, image, label, row, col, index):
    fig.add_subplot(row, col, index)
    plt.axis('off')
    plt.imshow(image)
    plt.title(label)
    return

image_index = 0
row = 4
col = 6
for i in range(1,25):
    plot_images(fig, images_for_plot[image_index], labels_for_plot[image_index], row, col, i)
    image_index = image_index + 1
plt.show()
l1=[]
def load_data():
    """
    Loads data and preprocess. Returns train and test data along with labels.
    """
    images = []
    labels = []
    size = 224,224
    print("LOADING DATA FROM : ",end = "")
    for folder in os.listdir(train_dir):
        print(folder, end = ' | ')
        for image in os.listdir(train_dir + "/" + folder):
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)
            temp_img = cv2.resize(temp_img, size)
            images.append(temp_img)
            labels.append(ord(folder)-97)
    
    images = np.array(images)
    #images = images.astype('float32')/255.0
    for i in range(len(images)):
        images[i]=images[i].astype('float32')/255
    l1=labels
    labels = keras.utils.to_categorical(labels)
    
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.25)
    
    print()
    print('Loaded', len(X_train),'images for training,','Train data shape =',X_train.shape)
    print('Loaded', len(X_test),'images for testing','Test data shape =',X_test.shape)
    
    return X_train, X_test, Y_train, Y_test,l1
X_train, X_test, Y_train, Y_test,l1 = load_data()
def create_model1():
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in vgg_conv.layers[:-4]:
	    layer.trainable = False
    for layer in vgg_conv.layers:
	    print(layer, layer.trainable)
        
    model = models.Sequential()
    model.add(vgg_conv)
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(26, activation='softmax'))
    
    model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])
    model.summary()
    
    return model
def fit_model():
    model_hist = model.fit(X_train, Y_train, batch_size = 64, epochs = 8, validation_split = 0.15)
    return model_hist 
model = create_model1()
curr_model_hist = fit_model()
plt.plot(curr_model_hist.history['accuracy'])
plt.plot(curr_model_hist.history['val_accuracy'])
plt.legend(['train', 'test'], loc='lower right')
plt.title('accuracy plot - train vs test')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

plt.plot(curr_model_hist.history['loss'])
plt.plot(curr_model_hist.history['val_loss'])
plt.legend(['training loss', 'validation loss'], loc = 'upper right')
plt.title('loss plot - training vs vaidation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
evaluate_metrics = model.evaluate(X_test, Y_test)
print("\nEvaluation Accuracy = ", "{:.2f}%".format(evaluate_metrics[1]*100),"\nEvaluation loss = " ,"{:.6f}".format(evaluate_metrics[0]))
model.save("model.h5")
