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
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import cv2 
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

##############################
uniq_labels=['yes','no']
directory="../input/brain-mri-images-for-brain-tumor-detection"
def load_images(directory,uniq_labels):
    images = []
    labels = []
    for idx, label in enumerate(uniq_labels):
        for file in os.listdir(directory + "/" + label):
            filepath = directory + "/" + label + "/" + file
            image = cv2.resize(cv2.imread(filepath), (128, 128))
            images.append(image)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)        
    return images,labels  
        
#load images and labels
images,labels=load_images(directory,uniq_labels)

#preprosessing (one hot encoding for labels & scaling for images)
labels = keras.utils.to_categorical(labels)
images = images.astype("float32")/ 255.0

#split data to train and test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, stratify = labels)
#the model
model = keras.models.Sequential()
model.add( keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding="same", activation="relu", input_shape= (128,128,3)))#64
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Dropout(0.4))
model.add( keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu"))#128
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Dropout(0.4))
model.add( keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Flatten())
layer0 = keras.layers.Dense(512, activation="relu",kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.01))#512
layer1 = keras.layers.Dense(128, activation="relu",kernel_initializer="he_normal",kernel_regularizer=keras.regularizers.l2(0.01))
layer_output = keras.layers.Dense(2, activation="sigmoid",kernel_initializer="glorot_uniform")#sigmoid, softmax
model.add(layer0)
model.add(keras.layers.Dropout(0.2))
model.add(layer1)
model.add(keras.layers.Dropout(0.2))
model.add(layer_output)

# The model’s summary() method displays all the model’s layers
print(model.summary())              
# initialize the training data augmentation object
trainAug = keras.preprocessing.image.ImageDataGenerator(rotation_range=15, fill_mode="nearest")

# Compiling the model
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss="categorical_crossentropy", optimizer= opt, metrics=["accuracy"])#categorical_crossentropy,binary_crossentropy

# Training and evaluating the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)


# plot the learning curves
pd.DataFrame(history.history).plot(figsize=(12, 8))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
# Evaluate the model
model_evaluate = model.evaluate(X_test, y_test)
print("Loss     : ",model_evaluate[0])
print("accuracy : ",model_evaluate[1])
#prediction for test images
y_pred = model.predict_classes(X_test)
#real values for test images
y_test_=np.argmax(y_test, axis=1)
# Compute classification report
print("Classification report : \n",classification_report(y_test_, y_pred))
# Confusion Matrix
cm = confusion_matrix(y_test_, y_pred)
cm
# Function to draw confusion matrix
def draw_confusion_matrix(true,preds):
    # Compute confusion matrix
    conf_matx = confusion_matrix(true, preds)
    print("Confusion matrix : \n")
    sns.heatmap(conf_matx, annot=True,annot_kws={"size": 12},fmt='g', cbar=False, cmap="viridis")
    plt.show()
    return conf_matx
con_mat = draw_confusion_matrix(y_test_, y_pred)
