# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!ls
from matplotlib import pyplot
from matplotlib.image import imread
import matplotlib.pyplot as plt
# define location of dataset
img1 = '/kaggle/input/agriculture-crop-images/test_crop_image/rice-field.jpg'
img2 = '/kaggle/input/agriculture-crop-images/test_crop_image/wheatarial02.jpg'
img3 = '/kaggle/input/agriculture-crop-images/test_crop_image/juteplants.jpg'
img4 = '/kaggle/input/agriculture-crop-images/test_crop_image/sugarcane-farm-in-the-mountain-countryside-of-thailand.jpg'
# plot first few images
image = imread(img1)
pyplot.imshow(image)
pyplot.show()
image = imread(img2)
pyplot.imshow(image)
pyplot.show()
image = imread(img3)
pyplot.imshow(image)
pyplot.show()
image = imread(img4)
pyplot.imshow(image)
pyplot.show()
from keras.preprocessing.image import img_to_array
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
def summarize_diagnostics(history):
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.title("Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), history.history["accuracy"], label="accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
IMAGE_DIMS = (224, 224, 3)
train_data_dir = '../input/agriculture-crop-images/kag2'
batch_size=64
train_datagen = ImageDataGenerator(rescale=1.0/255.0, horizontal_flip=True,vertical_flip=True, rotation_range=90)
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(IMAGE_DIMS[0], IMAGE_DIMS[1]),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
nb_train_samples=804 
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model


img_rows = 224
img_cols = 224 

pre_model = VGG16(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))

for layer in pre_model.layers:
    layer.trainable = False
    
for (i,layer) in enumerate(pre_model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)
def addTopModel(bottom_model, num_classes, D=512):
    top_model = bottom_model.output
    top_model = Flatten(name = "flatten")(top_model)
    top_model = Dense(D, activation = "relu")(top_model)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(D, activation = "relu")(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(num_classes, activation = "softmax")(top_model)
    return top_model
num_classes = 5
FC_Head = addTopModel(pre_model, num_classes)
model = Model(inputs=pre_model.input, outputs=FC_Head)
print(model.summary())
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(lr = 0.0001),
              metrics = ['accuracy'])

nb_train_samples=804 
epochs = 40
batch_size = 64
checkpoint = ModelCheckpoint("./weights.h5",
                             monitor="loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)
callbacks = [ checkpoint]
history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks)
summarize_diagnostics(history)
test_data_dir = '../input/agriculture-crop-images/crop_images'
test_datagen = ImageDataGenerator(rescale=1./255)
 
# Change the batchsize according to your system RAM
test_batchsize = 64
 
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=test_batchsize,
        class_mode='categorical',
        shuffle=False)
model.load_weights("weights.h5")
class_labels = test_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())
Y_pred = model.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = list(class_labels.values())
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

plt.figure(figsize=(8,8))
cnf_matrix = confusion_matrix(test_generator.classes, y_pred)

plt.imshow(cnf_matrix, interpolation='nearest')
plt.colorbar()
tick_marks = np.arange(len(classes))
_ = plt.xticks(tick_marks, classes, rotation=90)
_ = plt.yticks(tick_marks, classes)

import cv2
import numpy as np
def predict_crop(path,actual,class_labels):
    predict_datagen = ImageDataGenerator(rescale=1./255)
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img).reshape((1, 224, 224, 3))
    Y_pred = model.predict(img)
    y_pred = np.argmax(Y_pred, axis=1)
    if y_pred == actual:
        print('Correct prediction')
    else:
        print("Messed up!!")
    print('Actual class "{0}" and predicted class "{1}"'.format(class_labels[int(y_pred)],class_labels[actual]))


predict_crop('../input/agriculture-crop-images/test_crop_image/sugarcane fields.jpg',3,class_labels)
predict_crop('../input/agriculture-crop-images/test_crop_image/wheatss.jpg',4,class_labels)
predict_crop('../input/agriculture-crop-images/test_crop_image/wheatcrop01.jpg',4,class_labels)
predict_crop('../input/agriculture-crop-images/test_crop_image/maize-field.jpg',1,class_labels)
predict_crop('../input/agriculture-crop-images/test_crop_image/jute003.jpg',0,class_labels)
predict_crop('../input/agriculture-crop-images/test_crop_image/rice8122f869e3f.jpg',2,class_labels)
predict_crop('../input/agriculture-crop-images/test_crop_image/sugarcane fields.jpg',3,class_labels)
class_labels

