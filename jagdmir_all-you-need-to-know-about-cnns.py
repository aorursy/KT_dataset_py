import numpy as np

import pandas as pd

import glob

import cv2

import matplotlib.pyplot as plt
train_dir='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

train=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
train.head()
train['path'] = train_dir + train.image_name + ".jpg"

train.head()
img=cv2.imread('../input/siim-isic-melanoma-classification/jpeg/train/ISIC_0015719.jpg')   

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
train.target.value_counts()
df_0=train[train['target']==0].sample(600)

df_1=train[train['target']==1]

train=pd.concat([df_0,df_1])

train=train.reset_index()
train.shape
train.head()
# we will resize the given images to 128 x 128 size images for faster processing

IMG_DIM = (128, 128)
# Keras provides some amazing libraries to work with images, lets import them

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
# We will reserve 20% of our training data for the validation purpose

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train, train.target, test_size=0.2, random_state=42)
# save image path

train_files = X_train.path

val_files = X_val.path



# load images using load_img function from keras preprocessing 

# target_size is used to load the images with smaller size

# img_to_array will tranform the loaded image to an array

train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files]

validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in val_files]



# convert the list of arrays to array

train_imgs = np.array(train_imgs)

train_labels = y_train



validation_imgs = np.array(validation_imgs)

val_labels = y_val





print('Train dataset shape:', train_imgs.shape, 

      '\tValidation dataset shape:', validation_imgs.shape)
train_imgs_scaled = train_imgs.astype('float32')



validation_imgs_scaled  = validation_imgs.astype('float32')



# divide the pixels by 255 to scale the pixels between 0 and 1

train_imgs_scaled /= 255

validation_imgs_scaled /= 255



print(train_imgs[0].shape)



# array_to_img function will convert the given array to image

array_to_img(train_imgs[0])
# setup basic configuration

batch_size = 30

num_classes = 2

epochs = 30

input_shape = (128, 128, 3)
# Here we will import the necessary libraries

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



# we will import sequential model and add different layers to it

from keras.models import Sequential



# import optimizers, please go through online tutorials if you want to learn what is the purpose of an optimizer

from keras import optimizers



# creating and instance of Sequential

model = Sequential()



# add Conv2D layer(this is the convolutional layer we discussed earlier),filter size,kernel size,activation and padding are the parameters used

# This layer would create feature maps for each and every filter used

# feature maps created here are then taken through an activation function(relu here), which decides whether a certain feature is present 

# at a given location in the image.

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', 

                 input_shape=input_shape))

# Pooling layer used here will select the largest values on the feature maps and use these as inputs to subsequent layers

model.add(MaxPooling2D(pool_size=(2, 2)))





# another set of Convolutional & Max Pooling layers

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

# Finally the Dense Layer

model.add(Dense(512, activation='relu'))

# sigmoid function here will help us in perform binary classification

model.add(Dense(1, activation='sigmoid'))





model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(),

              metrics=['accuracy'])



model.summary()
history = model.fit(x=train_imgs_scaled, y=train_labels,

                    validation_data=(validation_imgs_scaled, val_labels),

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

t = f.suptitle('Basic CNN Performance', fontsize=12)

f.subplots_adjust(top=0.85, wspace=0.3)



epoch_list = list(range(1,31))

ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')

ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')

ax1.set_xticks(np.arange(0, 31, 5))

ax1.set_ylabel('Accuracy Value')

ax1.set_xlabel('Epoch')

ax1.set_title('Accuracy')

l1 = ax1.legend(loc="best")



ax2.plot(epoch_list, history.history['loss'], label='Train Loss')

ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')

ax2.set_xticks(np.arange(0, 31, 5))

ax2.set_ylabel('Loss Value')

ax2.set_xlabel('Epoch')

ax2.set_title('Loss')

l2 = ax2.legend(loc="best")
model = Sequential()



model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',

                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))





model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(),

              metrics=['accuracy'])

              

              

history = model.fit(x=train_imgs_scaled, y=train_labels,

                    validation_data=(validation_imgs_scaled, val_labels),

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1)                      
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

t = f.suptitle('CNN Model with Regularization', fontsize=12)

f.subplots_adjust(top=0.85, wspace=0.3)



epoch_list = list(range(1,31))

ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')

ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')

ax1.set_xticks(np.arange(0, 31, 5))

ax1.set_ylabel('Accuracy Value')

ax1.set_xlabel('Epoch')

ax1.set_title('Accuracy')

l1 = ax1.legend(loc="best")



ax2.plot(epoch_list, history.history['loss'], label='Train Loss')

ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')

ax2.set_xticks(np.arange(0, 31, 5))

ax2.set_ylabel('Loss Value')

ax2.set_xlabel('Epoch')

ax2.set_title('Loss')

l2 = ax2.legend(loc="best")
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,

                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 

                                   horizontal_flip=True, fill_mode='nearest')



val_datagen = ImageDataGenerator(rescale=1./255)
# lets take a random image and see how transformated images actually looks

img_id = 1



img_generator = train_datagen.flow(train_imgs[img_id:img_id+1], train_labels[img_id:img_id+1],

                                   batch_size=1)



img = [next(img_generator) for i in range(0,5)]



fig, ax = plt.subplots(1,5, figsize=(16, 6))

print('Labels:', [item[1][0] for item in img])

l = [ax[i].imshow(img[i][0][0]) for i in range(0,5)]
train_generator = train_datagen.flow(train_imgs, train_labels, batch_size=30)

val_generator = val_datagen.flow(validation_imgs, val_labels, batch_size=20)



input_shape = input_shape



model = Sequential()



model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', 

                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['accuracy'])

              

history = model.fit_generator(train_generator, steps_per_epoch=32, epochs=100,

                              validation_data=val_generator, validation_steps=12, 

                              verbose=1)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

t = f.suptitle('CNN with Regularization & Augmentation', fontsize=12)

f.subplots_adjust(top=0.85, wspace=0.3)



epoch_list = list(range(1,101))

ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')

ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')

ax1.set_xticks(np.arange(0, 101, 5))

ax1.set_ylabel('Accuracy Value')

ax1.set_xlabel('Epoch')

ax1.set_title('Accuracy')

l1 = ax1.legend(loc="best")



ax2.plot(epoch_list, history.history['loss'], label='Train Loss')

ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')

ax2.set_xticks(np.arange(0, 101, 5))

ax2.set_ylabel('Loss Value')

ax2.set_xlabel('Epoch')

ax2.set_title('Loss')

l2 = ax2.legend(loc="best")
# lets check the layers we added to our model



for l in model.layers:

    print(l.name,l)
# Visualise Filters



# lets save layer names and layers to a dictionary 

layer_dict = dict([(layer.name, layer) for layer in model.layers])



# lets pick one layer and visualise it

layer_name = 'conv2d_7'

filter_index = 0 # index of the filter we will visualize



# Grab the filter and bias weights for the selected layer

filters, biases = layer_dict[layer_name].get_weights()



# Normalize filter values to a range of 0 to 1 so we can visualize them

# This will help create a clear visualisation when we show the weights as colours on the screen.

f_min, f_max = np.amin(filters), np.amax(filters)

filters = (filters - f_min) / (f_max - f_min)



# Plot first few filters

n_filters, index = 6, 1

for i in range(n_filters):

    f = filters[:, :, :, i]

    

    # Plot each channel separately

    for j in range(3):



        ax = plt.subplot(n_filters, 3, index)

        ax.set_xticks([])

        ax.set_yticks([])

        

        plt.imshow(f[:, :, j], cmap='viridis') 

        index += 1

        

plt.show()