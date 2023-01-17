import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

%matplotlib inline

import cv2 as cv

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

from keras.models import Sequential

from keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,Dropout,AveragePooling2D,BatchNormalization,LeakyReLU,DepthwiseConv2D

from keras.regularizers import l2

from sklearn.metrics import confusion_matrix
test_path='../input/intel-image-classification/seg_test/seg_test'

train_path='../input/intel-image-classification/seg_train/seg_train'
folders=os.listdir(train_path)

print("Number of categories :{len(folders)}")

print(folders)

folders_test=os.listdir(test_path)

print(folders_test)
train_no,test_no=dict(),dict()

for folder in folders:

    train_no[folder]=os.listdir(train_path+'/'+folder)

    test_no[folder]=os.listdir(test_path+'/'+folder)
print("TEST NO: ")

for _ in test_no:

    print(_,":",len(test_no[_]))

print("==============================")

for _ in train_no:

    print(_,":",len(train_no[_]))
rows,cols=1,5

for folder in folders:

    path =train_path+"/"+folder

    fig = plt.figure(figsize = (13, 8))

    print(folder + " images:")

    for i in range(rows * cols):

        fig.add_subplot(rows, cols, i+1)

        image_id = os.listdir(path)[np.random.randint(0, 2000)]

        image = cv.imread(path + f'/{image_id}')

        plt.imshow(image[:, :, ::-1])

        plt.title(image_id)

        plt.axis('off')
target_size=(100,100)

colormode='rgb'

batch_size=84

seed=400

input_shape=(100,100,3)

reg=None

axis=1

datagen = ImageDataGenerator(rescale = 1.0/255.0,

                            validation_split = 0.1)



# Creating train, valid and test generators

train_generator = datagen.flow_from_directory(directory = train_path, # Path to directory which contains images classes

                                             target_size = target_size, # Whether resize images or not

                                             color_mode = colormode, 

                                             batch_size = batch_size,

                                             class_mode = 'categorical',

                                             shuffle = True,

                                             seed = seed,

                                             subset = 'training') # Train or validation dataset



valid_generator = datagen.flow_from_directory(directory = train_path,

                                             target_size = target_size,

                                             color_mode = colormode,

                                             batch_size = batch_size,

                                             class_mode = 'categorical',

                                             shuffle = True,

                                             seed = seed,

                                             subset = 'validation')



test_generator = datagen.flow_from_directory(directory = test_path,

                                            target_size = target_size,

                                            color_mode = colormode,

                                            batch_size = 1,

                                            class_mode = None,

                                            shuffle = False, 

                                            seed = seed)

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
momentum = 0.9

model = Sequential()

model.add(Conv2D(64, 3, input_shape = input_shape, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = reg))

model.add(BatchNormalization(momentum = momentum, center=True, scale=False))

model.add(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = reg))

model.add(BatchNormalization(momentum = momentum, center=True, scale=False))

model.add(MaxPooling2D(2, 2))



model.add(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = reg))

model.add(BatchNormalization(momentum = momentum, center=True, scale=False))

model.add(Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = reg))

model.add(BatchNormalization(momentum = momentum, center=True, scale=False))

model.add(MaxPooling2D(2, 2))



model.add(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = reg))

model.add(BatchNormalization(momentum = momentum, center=True, scale=False))

model.add(Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = reg))

model.add(BatchNormalization(momentum = momentum, center=True, scale=False))

model.add(MaxPooling2D(2, 2))



model.add(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = reg))

model.add(BatchNormalization(momentum = momentum, center=True, scale=False))

model.add(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = reg))

model.add(BatchNormalization(momentum = momentum, center=True, scale=False))

model.add(MaxPooling2D(2, 2))



model.add(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = reg))

model.add(BatchNormalization(momentum = momentum, center=True, scale=False))

model.add(Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer = reg))

model.add(BatchNormalization(momentum = momentum, center=True, scale=False))

model.add(MaxPooling2D(2, 2))



model.add(Flatten())

model.add(Dense(512, activation = 'relu', kernel_initializer = 'he_normal'))

model.add(Dense(256, activation = 'relu', kernel_initializer = 'he_normal'))

model.add(Dense(6, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



# Defining checkpoint callback

checkpoint = ModelCheckpoint('../working/best_model.hdf5', verbose = 1, monitor = 'val_accuracy', save_best_only = True)



# Fit model

history = model.fit_generator(generator = train_generator,

                             steps_per_epoch = STEP_SIZE_TRAIN,

                             validation_data = valid_generator,

                             validation_steps = STEP_SIZE_VALID,

                             epochs = 30, callbacks = [checkpoint])

# Accuracy/validation plots

h = history.history

fig = plt.figure(figsize = (13, 4))



plt.subplot(121)

plt.plot(h['accuracy'], label = 'acc')

plt.plot(h['val_accuracy'], label = 'val_acc')

plt.legend()

plt.grid()

plt.title(f'accuracy')



plt.subplot(122)

plt.plot(h['loss'], label = 'loss')

plt.plot(h['val_loss'], label = 'val_loss')

plt.legend()

plt.grid()

plt.title(f'loss')
# Loading weights from best model

model.load_weights('../working/best_model.hdf5')



# Saving all model

model.save('../working/model.hdf5')



# Evaluate mmodel

evaluated = model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID)

print(f'Best model loss: {round(evaluated[0], 2)}')

print(f'Best model accuracy: {round(evaluated[1] * 100, 2)}%')
# Predict classes in test dataset

STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

test_generator.reset()

preds = model.predict_generator(test_generator, steps = STEP_SIZE_TEST, verbose = 1)
# Creating y_true and y_pred for confusion matrix

predicted_class_indices = np.argmax(preds, axis = 1)

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

y_pred = [labels[k] for k in predicted_class_indices]

y_true = [labels[k] for k in test_generator.labels]
# Plotting confusion matrix

l = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

confusion = confusion_matrix(y_true, y_pred)

fig = plt.figure(figsize = (6, 6))

sns.heatmap(confusion, annot = True, fmt = 'd', xticklabels = l, yticklabels = l, square = True)
# Showing examples of images with true and predicted classes

filenames = test_generator.filenames

rows, cols = (4, 4)



fig = plt.figure(figsize = (12, 12))

for i in range(rows * cols):

    r = np.random.randint(0, 2999)

    image_path = test_path + '/' + filenames[r]

    image = cv.imread(image_path)

    fig.add_subplot(rows, cols, i+1)

    plt.imshow(image[:, :, ::-1])

    plt.title(f'True: {filenames[r].split("/")[0]}\nPred: {y_pred[r]}')

    plt.axis('off')



plt.tight_layout()