import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)v

import matplotlib.pyplot as plt # data visualization

import seaborn as sns



from keras.datasets import cifar10 # CIFAR-10 dataset

from keras.utils import np_utils, plot_model # np_utils is used to do one-hot encoding

from keras.optimizers import Adam # adam optmizer

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, BatchNormalization # layers used in the neural network

from keras.regularizers import l2 # l2 regularizer

from keras.models import Sequential, load_model # the model used has the Sequential structure

from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import confusion_matrix, classification_report





from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

labels = pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv')
# Datasets shapes

print('Train dataset shape: {}'.format(X_train.shape))

print('Test dataset shape: {}'.format(X_test.shape))
print('Amount of images in each class of the training dataset:')

print(labels.label.value_counts())
# Getting a dictionary with 3 images indexes for each label

labels_dict = {}

for label in labels.label.unique():

    labels_dict[label] = list(labels[labels['label'] == label].index[0:3])

    

# Showing 3 images of every class

for key in labels_dict:

    for i in range(3):

        plt.subplot(330+ 1 + i)

        plt.imshow(X_train[labels_dict[key][i]])

    print('Class {}'.format(key))

    plt.show()

# Initializing a ImageDataGenerator Objetc

image_data_generator = ImageDataGenerator(

                rotation_range = 15, # degree angle for random rotations

                horizontal_flip = True, # randomly flips the image horizontally

                width_shift_range = 0.1, # fraction of total width shift

                height_shift_range = 0.1 # fraction of total height shift

                )



# Fitting the Image Data Generator for the training images



image_data_generator.fit(X_train)
training_generator = image_data_generator.flow(X_train, y_train, batch_size = 6)
plt.figure(figsize=(10,5))

for i in range(6):

    plt.subplot(2,3,i+1)

    for x,y in training_generator:

        plt.imshow(x[i].astype(np.uint8))

        plt.axis('off')

        break

plt.tight_layout()

plt.show()
# Transform the data type to float 32 to performe the divison operation 

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



# Functions from numpy to calculate mean and std

mean = np.mean(X_train)

std = np.std(X_train)



# Applying the z-score



X_train = (X_train-mean)/(std+1e-7)

X_test = (X_test-mean)/(std+1e-7)
# every image has the same input_shape

input_shape = (32,32,3)



# number of classes in the CIFAR-10

n_classes = 10



# applying the one-hot encoding

y_train = np_utils.to_categorical(y_train,n_classes)

y_test = np_utils.to_categorical(y_test,n_classes)
def model():

    # L2 λ = 0.0005

    regularizer = l2(0.0005)

    

    model = Sequential()

    

    model.add(Conv2D(32, (3,3), activation = 'relu', kernel_regularizer = regularizer, input_shape = input_shape, padding = 'same'))

    model.add(BatchNormalization(axis = -1))

    model.add(Conv2D(32,(3,3), activation = 'relu', kernel_regularizer = regularizer, padding = 'same'))

    model.add(BatchNormalization(axis = -1))

    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Dropout(0.1))

    

    model.add(Conv2D(64, (3,3), activation = 'relu', kernel_regularizer = regularizer, padding = 'same'))

    model.add(BatchNormalization(axis = -1))

    model.add(Conv2D(64,(3,3), activation = 'relu', kernel_regularizer = regularizer, padding = 'same'))

    model.add(BatchNormalization(axis = -1))

    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Dropout(0.1))

    

    model.add(Conv2D(128, (3,3), activation = 'relu', kernel_regularizer = regularizer, padding = 'same'))

    model.add(BatchNormalization(axis = -1))

    model.add(Conv2D(128,(3,3), activation = 'relu', kernel_regularizer = regularizer, padding = 'same'))

    model.add(BatchNormalization(axis = -1))

    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Dropout(0.1))

    



    

    model.add(Flatten())

    model.add(Dense(512, activation = 'relu', kernel_regularizer = regularizer))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation = 'softmax'))

    

    return model

    

model = model()

# Using keras Adam optimizer

AdamOpt = Adam(lr = 0.0003)



model.compile(optimizer = AdamOpt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()

    

    
def comparison_model():

    # No Regularizer

    regularizer = None

    

    model = Sequential()

    

    model.add(Conv2D(32, (3,3), activation = 'relu', kernel_regularizer = regularizer, input_shape = input_shape, padding = 'same'))

    # No batch normalization

    #model.add(BatchNormalization(axis = -1))

    model.add(Conv2D(32,(3,3), activation = 'relu', kernel_regularizer = regularizer, padding = 'same'))

    #model.add(BatchNormalization(axis = -1))

    model.add(MaxPooling2D(pool_size = (2,2)))

    

    # No dropout

    #model.add(Dropout(0.1))

    

    model.add(Conv2D(64, (3,3), activation = 'relu', kernel_regularizer = regularizer, padding = 'same'))

    #model.add(BatchNormalization(axis = -1))

    model.add(Conv2D(64,(3,3), activation = 'relu', kernel_regularizer = regularizer, padding = 'same'))

    #model.add(BatchNormalization(axis = -1))

    model.add(MaxPooling2D(pool_size = (2,2)))

    #model.add(Dropout(0.1))

    

    model.add(Conv2D(128, (3,3), activation = 'relu', kernel_regularizer = regularizer, padding = 'same'))

    #model.add(BatchNormalization(axis = -1))

    model.add(Conv2D(128,(3,3), activation = 'relu', kernel_regularizer = regularizer, padding = 'same'))

    #model.add(BatchNormalization(axis = -1))

    model.add(MaxPooling2D(pool_size = (2,2)))

    #model.add(Dropout(0.1))

    



    

    model.add(Flatten())

    model.add(Dense(512, activation = 'relu', kernel_regularizer = regularizer))

    #model.add(BatchNormalization())

    #model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation = 'softmax'))

    

    return model

    

comparison_model = comparison_model()

# Using keras Adam optimizer

AdamOpt = Adam(lr = 0.001)



comparison_model.compile(optimizer = AdamOpt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    

    
# CallBack functions

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)



# Training history

history = model.fit(image_data_generator.flow(X_train,y_train, batch_size = 200), steps_per_epoch = len(X_train)/200, epochs = 200, validation_data = (X_test,y_test), callbacks=[es, mc])



# load the saved model

saved_model = load_model('best_model.h5')
# CallBack functions

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=117)

mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)



# Training history

# 117 = number of epochs of the first model

comparison_history = comparison_model.fit(X_train, y_train, batch_size = 200, verbose = 1, epochs = 117, validation_data = (X_test,y_test), callbacks=[es, mc])

def plotmodelhistory(history): 

    fig, axs = plt.subplots(1,2,figsize=(15,7)) 

    # summarize history for accuracy

    axs[0].plot(history.history['accuracy']) 

    axs[0].plot(history.history['val_accuracy']) 

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy') 

    axs[0].set_xlabel('Epoch')

    axs[0].legend(['train', 'validate'], loc='upper left')

    # summarize history for loss

    axs[1].plot(history.history['loss']) 

    axs[1].plot(history.history['val_loss']) 

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss') 

    axs[1].set_xlabel('Epoch')

    axs[1].legend(['train', 'validate'], loc='upper left')

    plt.show()



# list all data in history



plotmodelhistory(history)
plotmodelhistory(comparison_history)
max_val_acc = max(history.history['val_accuracy'])

max_train_acc = max(history.history['accuracy'])



min_val_loss = min(history.history['val_loss'])

min_train_loss = min(history.history['loss'])



print('Validation set: \nAccuracy: {}\nLoss: {}'.format(max_val_acc,min_val_loss))

print()

print('Train set: \nAccuracy: {}\nLoss: {}'.format(max_train_acc,min_train_loss))

# Get the predictions probabilities

predictions = model.predict(X_test)



# Get the true predictions

y_predictions = np.argmax(predictions, axis=1)



# Remove the one-hot encoding from the test label

y_true = np.argmax(y_test, axis=1)





# Getting the confusion matrix

cm = confusion_matrix(y_true, y_predictions)



# Transform to DataFrame and change the names of indexes and columns

cm = pd.DataFrame(cm)



cm.rename(columns={0:'Airplane',

                  1: 'Automobile',

                  2: 'Bird',

                  3: 'Cat',

                  4: 'Deer',

                  5: 'Dog',

                  6: 'Frog',

                  7: 'Horse',

                  8: 'Ship',

                  9: 'Truck'

                        }, 

                 inplace=True)

cm.rename(index={0:'Airplane',

                  1: 'Automobile',

                  2: 'Bird',

                  3: 'Cat',

                  4: 'Deer',

                  5: 'Dog',

                  6: 'Frog',

                  7: 'Horse',

                  8: 'Ship',

                  9: 'Truck'

                        }, 

                 inplace=True)
# Set the width and height of the figure

plt.figure(figsize=(14,7))



# Add title

plt.title("Confusion matrix")



# Heatmap showing 

sns.heatmap(data=cm, annot=True, fmt="d", linewidths=.5, cmap="YlGnBu")



# Add label for horizontal axis

plt.xlabel("Predicted Label")

plt.xlabel("True Label Label")



plt.show()
print(classification_report(y_true, y_predictions))