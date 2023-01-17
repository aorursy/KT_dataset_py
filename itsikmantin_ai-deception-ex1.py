import pandas as pd

import numpy as np

import itertools



from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, MaxPool2D

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras import optimizers

from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding

from tensorflow.keras.optimizers import RMSprop



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



np.random.seed()

sns.set(style='white', context='notebook', palette='deep')



print("Done loading packages")
# Load the training data

dataset = pd.read_csv("../input/train.csv")



#Load the test data for the competition submission

competition_dataset = pd.read_csv("../input/test.csv")



print("Done loading data")

dataset.describe

# A label is the thing we're predicting

label = dataset["label"]



# A feature is an input variable, in this case a 28 by 28 pixels

# Drop 'label' column

feature = dataset.drop(labels = ["label"],axis = 1)



# let's check we have a good distribution of the handwritten digits

g = sns.countplot(label)



# free some space

del dataset 



print("Done deleting original dataset")
# Show a random example

rand_example = np.random.choice(feature.index)

_, ax = plt.subplots()

ax.imshow(feature.loc[rand_example].values.reshape(28, 28), cmap='gray_r')

ax.set_title("Label: %i" % label.loc[rand_example])

ax.grid(False)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

label = to_categorical(label, num_classes = 10)



# Normalize between 0 and 1 the data for both training and competition dataset (The pixel-value is an integer between 0 and 255)

feature = feature / 255.0

competition_dataset = competition_dataset / 255.0



print("Done")
# Split the dataset into train and validation set

# Keep 10% for the validation and 90% for the training

# Stratify is argument to keep trainingset evenly balanced ofver the labels (eg validation set not only the digit 5)



feature_train, feature_val, label_train, label_val = train_test_split(feature, label, test_size = 0.1, stratify=label)

print("Done train test split")
# First model is a dense neural network model with 5 layers

model_1 = Sequential()

model_1.add(Dense(200, activation = "relu", input_shape = (784,)))

model_1.add(Dense(100, activation = "relu"))

model_1.add(Dense(60, activation = "relu"))

model_1.add(Dense(30, activation = "relu"))

model_1.add(Dense(10, activation = "softmax"))



# Define the optimizer and compile the model

optimizer = optimizers.SGD(lr=0.03, clipnorm=5.)

model_1.compile(optimizer= optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



print("Done specifying architecture for model_1")

print (model_1.summary())
# With this model you should be able to achieve around 95.5% accuracy

# change epochs to 8 to have a full run



history = model_1.fit(feature_train, label_train, batch_size = 100, epochs = 8, 

          validation_data = (feature_val, label_val), verbose = 1)

print("Done fitting model_1 through 8 epochs")
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# First let's reshape the array into a 28*28 picture with 1 color channel (b/w picture)

#Take a random example to print it before and after the conversion

rand_example = np.random.choice(1000)

_, ax = plt.subplots()

ax.imshow(feature.loc[rand_example].values.reshape(28, 28), cmap='gray_r')

ax.set_title("Before")

ax.grid(False)



feature = feature.values.reshape(-1,28,28,1)

competition_dataset = competition_dataset.values.reshape(-1,28,28,1)



_, ax = plt.subplots()

g = plt.imshow(feature[rand_example][:,:,0], cmap='gray_r')

ax.set_title("After")

ax.grid(False)

# Split the dataset into train and validation set

# Keep 10% for the validation and 90% for the training

# Stratify is argument to keep trainingset evenly balanced ofver the labels (eg validation set not only the digit 5)



feature_train, feature_val, label_train, label_val = train_test_split(feature, label, test_size = 0.1, stratify=label)

print("Done train test split")
# Second model is a 3 layer convolutional network model with one dense layer at the end



model_2 = Sequential()

model_2.add(Conv2D(filters = 4, kernel_size = (5,5), strides = 1, padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model_2.add(Conv2D(filters = 8, kernel_size = (4,4), strides = 2, padding = 'Same', 

                 activation ='relu'))

model_2.add(Conv2D(filters = 12, kernel_size = (4,4), strides = 2, padding = 'Same', 

                 activation ='relu'))

model_2.add(Flatten())

model_2.add(Dense(200, activation = "relu"))

model_2.add(Dense(10, activation = "softmax"))



# Define the optimizer and compile the model

optimizer = optimizers.SGD(lr=0.03, clipnorm=5.)

model_2.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



print("Done building network architecture for model_2")

print (model_2.summary())

# With this model you should be able to achieve around 98% accuracy

# change epochs to 8 to have a full run



history = model_2.fit(feature_train, label_train, batch_size = 100, epochs = 8, 

          validation_data = (feature_val, label_val), verbose = 1)

print("Done fitting model_2")
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Generate 22 million more images by randomly rotating, scaling, and shifting 42,000 (-10% validation set) images

datagen = ImageDataGenerator(

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        featurewise_center=False,  # do not set input mean to 0 over the dataset

        samplewise_center=False,  # do not set each sample mean to 0

        featurewise_std_normalization=False,  # no divide inputs by std of the dataset

        samplewise_std_normalization=False,  # no divide each input by its std

        zca_whitening=False,  # No ZCA whitening

        horizontal_flip=False,  # no horizontal flip images

        vertical_flip=False)  # no vertical flip images, no 6 and 9 mismatches :-)



datagen.fit(feature_train)



print("Done")
# Third model is a 3 layer convolutional network model with one dense layer at the end, it contains more neurons, has dropout applied in the dense layer, 

# data augmentation and the adam optimizer



model_3 = Sequential()

model_3.add(Conv2D(filters = 6, kernel_size = (6,6), strides = 1, padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model_3.add(Conv2D(filters = 12, kernel_size = (5,5), strides = 2, padding = 'Same', 

                 activation ='relu'))

model_3.add(Conv2D(filters = 24, kernel_size = (4,4), strides = 2, padding = 'Same', 

                 activation ='relu'))

model_3.add(Flatten())

model_3.add(Dense(200, activation = "relu"))

model_3.add(Dropout(0.75))

model_3.add(Dense(10, activation = "softmax"))



# Define the optimizer and compile the model

model_3.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])

print("Done buliding network for model_3")

print (model_3.summary())
# With this model you should be able to achieve around 99.3% accuracy

# change epochs to 35 to have a full run



history = model_3.fit_generator(datagen.flow(feature_train,label_train, batch_size=100),

                            epochs = 10, validation_data = (feature_val, label_val),

                           verbose = 2)

print("Done fitting model_3")
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Fourth model with hyper parameter tuning 



model_4 = Sequential()

model_4.add(Conv2D(filters = 32, kernel_size = (5,5), strides = 1, padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model_4.add(BatchNormalization())

model_4.add(Conv2D(filters = 32, kernel_size = (5,5), strides = 1, padding = 'Same', 

                 activation ='relu'))

model_4.add(BatchNormalization())

model_4.add(Dropout(0.4))



model_4.add(Conv2D(filters = 64, kernel_size = (3,3), strides = 2, padding = 'Same', 

                 activation ='relu'))

model_4.add(BatchNormalization())

model_4.add(Conv2D(filters = 64, kernel_size = (3,3), strides = 2, padding = 'Same', 

                 activation ='relu'))

model_4.add(BatchNormalization())

model_4.add(Dropout(0.4))



model_4.add(Flatten())

model_4.add(Dense(256, activation = "relu"))

model_4.add(Dropout(0.4))

model_4.add(Dense(10, activation = "softmax"))



# Define the optimizer and compile the model

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model_4.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

print (model_4.summary())
# With this model you should be able to achieve around 99.6% accuracy

# change epochs to 35 to have a full run



history = model_4.fit_generator(datagen.flow(feature_train,label_train, batch_size=100),

                            epochs = 10, validation_data = (feature_val, label_val),

                           verbose = 2, callbacks=[learning_rate_reduction])
# Look at confusion matrix 



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

Y_pred = model_4.predict(feature_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(label_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 

# Display some error results 



# Errors are difference between predicted labels and true labels

errors = (Y_pred_classes - Y_true != 0)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

X_val_errors = feature_val[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    """ This function shows 6 images with their predicted and real labels"""

    n = 0

    nrows = 2

    ncols = 3

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)), cmap='gray_r')

            ax[row,col].set_title("Pred: {}; True: {}".format(pred_errors[error],obs_errors[error]))

            n += 1



# Probabilities of the wrong predicted numbers

Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)



# Predicted probabilities of the true values in the error set

true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors



# Sorted list of the delta prob errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)



# Top 6 errors 

most_important_errors = sorted_dela_errors[-6:]



# Show the top 6 errors

display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
# predict results

results = model_4.predict(competition_dataset)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission_MNIST.csv",index=False)

import os

print(os.listdir("../working"))



#kaggle competitions submit -c digit-recognizer -f submission_MNIST.csv -m "My fist submission"