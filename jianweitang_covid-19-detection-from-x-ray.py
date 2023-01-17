# Data preprocessing

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import glob



# Deep learning

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D

from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.optimizers import Adam, SGD, RMSprop

import tensorflow as tf 

from tensorflow.keras.preprocessing import image



import os
dirname = '/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database'

os.listdir(dirname)
dirname2 = '../input/actualmedcovidchestxraydataset'

os.listdir(dirname2)
metadata = pd.read_csv(os.path.join(dirname2, 'metadata.csv'))

metadata
metadata['finding'].value_counts()
metadata[metadata['finding']=='COVID-19'].head()
images_covid19 = metadata[metadata['finding']=='COVID-19']['imagename'].to_list()
normal_images = []

count = 0



# load images

for img_path in glob.glob(dirname + "/NORMAL/*"):

    count += 1

    normal_images.append(image.load_img(str(img_path), target_size = (150,150,3)))

    if count > 280:   # we keep 280 images since there are only 219+58 images of covid-19, then the distribution of classes can be balanced.

        break



# plot the first normal lung images in the dataset

fig = plt.figure()

fig.suptitle("Normal Lungs")

plt.imshow(normal_images[0], cmap="gray")

plt.show()
covid_images = []

count = 0



for img_path in glob.glob(dirname + "/COVID-19/*"):

    count += 1

    covid_images.append(image.load_img(str(img_path), target_size = (150,150,3)))

    if count > 280:   # we keep 280 images since there are only 219+58 images of covid-19, then the distribution of classes can be balanced.

        break



for image_name in images_covid19:

    img_path = os.path.join(dirname2, 'images', image_name)

    normal_images.append(image.load_img(str(img_path), target_size = (150,150,3)))



# plot the first image of covid-19 infection.

fig = plt.figure()

fig.suptitle("Covid-19 Patient's Lungs")

plt.imshow(covid_images[0], cmap="gray")

plt.show()
pneumonia_images = []

count = 0



for img_path in glob.glob(dirname + "/Viral Pneumonia/*"):

    count += 1

    pneumonia_images.append(image.load_img(str(img_path), target_size = (150,150,3)))

    if count > 280:   # we keep 280 images since there are only 219+58 images of covid-19, then the distribution of classes can be balanced.

        break



# plot the first Viral Pneumonia image

fig = plt.figure()

fig.suptitle("Viral Pneumonia Lungs")

plt.imshow(pneumonia_images[0], cmap="gray")

plt.show()
images_together = []



for i in normal_images:

    images_together.append(img_to_array(i))

    

for i in covid_images:

    images_together.append(img_to_array(i))

    

for i in pneumonia_images:

    images_together.append(img_to_array(i))



# normal-> 0, covid-19-> 1, pneumonia-> 2

targets = np.zeros((len(images_together), 3), int)

targets[:len(normal_images)] = [1, 0, 0]

targets[len(normal_images):] = [0, 1, 0]

targets[len(normal_images)+len(covid_images):] = [0, 0, 1]
print("image list length: ",len(images_together))

print("target list length: ",len(targets))
targets = np.array(targets)

print("targets: ",targets.shape)

# targets = targets.reshape(-1,1)

# print("new shape of targets: ",targets.shape)
# look at the shape of the imges

images_together = np.array(images_together)

print("shape of images together: ",images_together.shape)
# Re-construct the dataset to meet the input dimensions.

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(images_together, targets, test_size=0.2, stratify=targets)



images_together = np.concatenate((X_train, X_val))

targets = np.concatenate((y_train, y_val))
# Define HyperParameters

input_shape = (150, 150, 3)

num_classes = 3

epochs = 32

batch_size = 40
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = input_shape, activation = "relu"))

model.add(MaxPool2D(pool_size = (2,2)))



model.add(Conv2D(filters = 48, kernel_size = (3,3), activation = "relu"))

model.add(MaxPool2D(pool_size = (2,2)))



model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = "relu"))

model.add(MaxPool2D(pool_size = (2,2), strides = (1,1)))

model.add(Dropout(0.25))



#fully connected

model.add(Flatten())

model.add(Dense(32, activation = "relu"))

model.add(Dropout(0.2))

model.add(Dense(3, activation = "softmax"))



# compile 

model.compile(loss = "categorical_crossentropy",

             optimizer = "rmsprop",

             metrics = ["accuracy"])
model.summary()
tf.config.experimental.list_physical_devices('GPU')
# Create iterable training set with Data Augmentation methodologies such as rescale, shear, zoom and filp the data.

train_datagen = ImageDataGenerator(rescale = 1./255,

                                  shear_range = 0.2,

                                  zoom_range = 0.2,

                                  horizontal_flip = True,

                                  validation_split = 0.2)





# training iterable

train_generator = train_datagen.flow(

images_together, targets,

batch_size = batch_size,

subset = "training")



# validation iterable

validation_generator = train_datagen.flow(

images_together, targets,

batch_size = batch_size,

shuffle = False,

subset = "validation")



#Train the model using GPU acceleration

with tf.device('/GPU:0'):

    hist = model.fit_generator(

    train_generator,

    steps_per_epoch = (450*0.8)//batch_size,

    validation_data = validation_generator,

    validation_steps = (450*0.2)// batch_size,

    epochs = epochs)
# plot the model accuracy changes through training

plt.figure(figsize = (13,7))

plt.plot(hist.history["accuracy"])

plt.plot(hist.history["val_accuracy"])

plt.title("Model Accuracy")

plt.ylabel("Accuracy")

plt.xlabel("Epoch")

plt.legend(["Train", "Test"], loc = "upper left")

#plt.text(23,0.5,"Current Training Accuracy: "+str(np.round(hist.history["accuracy"][-1]*100,2))+"%",fontsize = 18,color = "black")

#plt.text(23,0.46,"Current Validation Accuracy: "+str(np.round(hist.history["val_accuracy"][-1]*100,2))+"%",fontsize = 18,color = "black")

plt.show()
print("Training Accuracy: "+str(np.round(hist.history["accuracy"][-1]*100,2))+"%")

print("Validation Accuracy: "+str(np.round(hist.history["val_accuracy"][-1]*100,2))+"%")
# measure the execution time

import time

start = time.time()

# making predictions

preds = model(X_val)

end = time.time()

print(f'in order to predict {len(X_val)} images, it takes {end-start} seconds')



# plot the confusion matrix

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns



y_pred = np.argmax(preds, axis=1)

y_true = np.argmax(y_val, axis=1)

matrix = confusion_matrix(y_true, y_pred)



ax = sns.heatmap(matrix, annot=True, fmt='d')

ax.set(xlabel='Predicted', ylabel='True')

labels = ['Normal', 'COVID-19', 'Viral Pneumonia']

plt.title('Confusion matrix of the classifier')

ax.set_xticklabels(labels)

ax.set_yticklabels(labels)

plt.show()