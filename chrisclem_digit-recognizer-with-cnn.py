import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

import keras
from keras.utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
TRAIN_FILE = "../input/train.csv"
TEST_FILE = "../input/test.csv"

IMG_ROWS, IMG_COLS = 28, 28
NUM_CLASSES = 10

TEST_SIZE = 0.1

FILTERS_1 = 32
KERNEL_SIZE_1 = (5, 5)
FILTERS_2 = 64
KERNEL_SIZE_2 = (3, 3)
STRIDES = 2
ACTIVATION_CONV2D = 'relu'

UNITS_DENSE = 256
ACTIVATION_DENSE = 'relu'

LOSS = keras.losses.categorical_crossentropy
OPTIMIZER = 'adam'
#OPTIMIZER = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
METRICS = ['accuracy']

BATCH_SIZE = 64
EPOCHS = 30
def data_prep_train(train):
    Y_train = to_categorical(train.label, 
                             num_classes = NUM_CLASSES)

    num_images = train.shape[0]
    X_as_array = train.values[:,1:]
    X_shaped_array = X_as_array.reshape(num_images, IMG_ROWS, IMG_ROWS, 1)
    X_train = X_shaped_array / 255.0
    return X_train, Y_train

def data_prep_test(test):

    num_images = test.shape[0]
    test_as_array = test.values
    test_shaped_array = test_as_array.reshape(num_images, IMG_ROWS, IMG_ROWS, 1)
    test = test_shaped_array / 255.0
    return test

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
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)
X_train, Y_train = data_prep_train(train)

test = data_prep_test(test)

# Set the random seed
random_seed = 2
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, 
                                                  test_size = TEST_SIZE, 
                                                  random_state=random_seed)
model = Sequential()

model.add(Conv2D(filters = FILTERS_1, 
                 kernel_size = KERNEL_SIZE_1,
                 #strides = STRIDES,
                 activation = ACTIVATION_CONV2D,
                 input_shape = (IMG_ROWS, IMG_COLS, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = FILTERS_1, 
                 kernel_size = KERNEL_SIZE_1,
                 #strides = STRIDES,
                 activation = ACTIVATION_CONV2D))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters = FILTERS_2, 
                 kernel_size = KERNEL_SIZE_2,
                 #strides = STRIDES,
                 activation = ACTIVATION_CONV2D))
model.add(BatchNormalization())
model.add(Conv2D(filters = FILTERS_2, 
                 kernel_size = KERNEL_SIZE_2,
                 #strides = STRIDES,
                 activation = ACTIVATION_CONV2D))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(UNITS_DENSE, 
                activation = ACTIVATION_DENSE))
model.add(Dropout(0.5))
model.add(Dense(2*UNITS_DENSE, 
                activation = ACTIVATION_DENSE))
model.add(Dense(NUM_CLASSES, 
                activation = 'softmax'))
# Compile model 
model.compile(loss = LOSS,
              optimizer = OPTIMIZER,
              metrics = METRICS)
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)

datagen.fit(X_train)
# Fit model
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = BATCH_SIZE),
                              steps_per_epoch = X_train.shape[0] // BATCH_SIZE,
                              epochs = EPOCHS,
                              verbose = 1,
                              callbacks = [learning_rate_reduction],
                              validation_data = (X_val, Y_val))
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 
results = model.predict(test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)

