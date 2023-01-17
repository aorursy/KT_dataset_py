# load libraries

import os

####*IMPORANT*: Have to do this line *before* importing tensorflow

os.environ['PYTHONHASHSEED']=str(1)



import tensorflow as tf

import numpy as np

import random



def reset_random_seeds():

   os.environ['PYTHONHASHSEED']=str(1)

   tf.random.set_seed(1)

   np.random.seed(1)

   random.seed(1)

    

reset_random_seeds()



import pandas as pd

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

from keras import models

import matplotlib.pyplot as plt

import seaborn as sns

from tensorflow.keras import models, layers

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.layers import Dropout, Flatten, Input, Dense
#https://www.kaggle.com/ibtesama/siim-baseline-keras-vgg16

train_dir='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

test_dir='/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

train=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

test=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')

submission=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

train.head()
labels=[]

data=[]

for i in range(train.shape[0]):

    data.append(train_dir + train['image_name'].iloc[i]+'.jpg')

    labels.append(train['target'].iloc[i])

df=pd.DataFrame(data)

df.columns=['images']

df['target']=labels



test_data=[]

for i in range(test.shape[0]):

    test_data.append(test_dir + test['image_name'].iloc[i]+'.jpg')

df_test=pd.DataFrame(test_data)

df_test.columns=['images']



X_train, X_val, y_train, y_val = train_test_split(df['images'],df['target'], test_size=0.2, random_state=1234)



train=pd.DataFrame(X_train)

train.columns=['images']

train['target']=y_train



validation=pd.DataFrame(X_val)

validation.columns=['images']

validation['target']=y_val



train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,horizontal_flip=True)

val_datagen=ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(

    train,

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    batch_size=8,

    shuffle=True,

    class_mode='raw')



validation_generator = val_datagen.flow_from_dataframe(

    validation,

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode='raw')
model = models.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu',

                        input_shape=(224, 224, 3)))

model.add(layers.MaxPooling2D((2, 2),strides=2))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(layers.Flatten())

model.add(layers.Dense(units=1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

fit = model.fit_generator(train_generator, steps_per_epoch=5, epochs=2,

                         validation_data=validation_generator, validation_steps=5)
metrics = list(fit.history.keys())

loss_values = fit.history[metrics[2]]

val_loss_values = fit.history[metrics[0]]

acc_values = fit.history[metrics[3]]

val_acc_values = fit.history[metrics[1]]

print("\nFinal validation loss function is", val_loss_values[-1])

print("Final validation accuracy is", val_acc_values[-1])



# summarize history for accuracy

plt.plot(fit.history['accuracy'])

plt.plot(fit.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(fit.history['loss'])

plt.plot(fit.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# Extracts the outputs of the all the layers

layer_outputs = [layer.output for layer in model.layers]

# Creates a model that will return these outputs, given the model input:

activation_model = models.Model(inputs=model.input, outputs=layer_outputs)



layer_names = []

for layer in model.layers:

    layer_names.append(layer.name)

    

layer_names
import cv2



img=cv2.imread(validation.images[0])

img = cv2.resize(img, (224,224))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.astype(np.float32)/255.



class_names = ['Benign', 'Malignant']



plt.imshow(img)

plt.axis('off')

plt.title(class_names[validation.target[0]], fontsize=12)

plt.show()
img1 = cv2.imread(validation.images[0])/255.

img1 = cv2.resize(img1, (224, 224), 3)

img1 = img1.reshape((1, 224, 224, 3))

img1.shape
activations = activation_model.predict(img1)

len(activations)
first_layer_activation = activations[0]

print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

plt.show()
plt.matshow(first_layer_activation[0, :, :, 27], cmap='viridis')

plt.show()
plt.matshow(first_layer_activation[0, :, :, 10], cmap='viridis')

plt.show()
layer_names = []

for layer in model.layers:

    layer_names.append(layer.name)



images_per_row = 16



# Now let's display our feature maps

for layer_name, layer_activation in zip(layer_names, activations):

    

    if layer_name == 'flatten': 

        break

    # This is the number of features in the feature map

    n_features = layer_activation.shape[-1]



    # The feature map has shape (1, size, size, n_features)

    size = layer_activation.shape[1]



    # We will tile the activation channels in this matrix

    n_cols = n_features // images_per_row

    display_grid = np.zeros((size * n_cols, images_per_row * size))



    # We'll tile each filter into this big horizontal grid

    for col in range(n_cols):

        for row in range(images_per_row):

            channel_image = layer_activation[0,

                                             :, :,

                                             col * images_per_row + row]

            # Post-process the feature to make it visually palatable

            channel_image -= channel_image.mean()

            channel_image /= channel_image.std()

            channel_image *= 64

            channel_image += 128

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[col * size : (col + 1) * size,

                         row * size : (row + 1) * size] = channel_image



    # Display the grid

    scale = 1. / size

    plt.figure(figsize=(scale * display_grid.shape[1],

                        scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')

    

plt.show()
img=cv2.imread(validation.images.iloc[32])

img = cv2.resize(img, (224,224))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.astype(np.float32)/255.



class_names = ['Benign', 'Malignant']



plt.imshow(img)

plt.axis('off')

plt.title(class_names[validation.target.iloc[32]], fontsize=12)

plt.show()
img1 = cv2.imread(validation.images.iloc[32])/255.

img1 = cv2.resize(img1, (224, 224), 3)

img1 = img1.reshape((1, 224, 224, 3))

print(img1.shape)



activations = activation_model.predict(img1)

print(len(activations))



first_layer_activation = activations[0]

print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

plt.show()
plt.matshow(first_layer_activation[0, :, :, 27], cmap='viridis')

plt.show()
plt.matshow(first_layer_activation[0, :, :, 10], cmap='viridis')

plt.show()
layer_names = []

for layer in model.layers:

    layer_names.append(layer.name)



images_per_row = 16



# Now let's display our feature maps

for layer_name, layer_activation in zip(layer_names, activations):

    

    if layer_name == 'flatten': 

        break

    # This is the number of features in the feature map

    n_features = layer_activation.shape[-1]



    # The feature map has shape (1, size, size, n_features)

    size = layer_activation.shape[1]



    # We will tile the activation channels in this matrix

    n_cols = n_features // images_per_row

    display_grid = np.zeros((size * n_cols, images_per_row * size))



    # We'll tile each filter into this big horizontal grid

    for col in range(n_cols):

        for row in range(images_per_row):

            channel_image = layer_activation[0,

                                             :, :,

                                             col * images_per_row + row]

            # Post-process the feature to make it visually palatable

            channel_image -= channel_image.mean()

            channel_image /= channel_image.std()

            channel_image *= 64

            channel_image += 128

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[col * size : (col + 1) * size,

                         row * size : (row + 1) * size] = channel_image



    # Display the grid

    scale = 1. / size

    plt.figure(figsize=(scale * display_grid.shape[1],

                        scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')

    

plt.show()