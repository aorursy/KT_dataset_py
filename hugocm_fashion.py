# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import itertools

import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Activation

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization



from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler





from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix



# Input data files are available in the "../input/" directory.

print(os.listdir("../input"))



# Setting Random Seed for Reproducibilty.

seed = 66

np.random.seed(seed)
data_train = pd.read_csv('../input/fashion-mnist_train.csv')

data_test = pd.read_csv('../input/fashion-mnist_test.csv')



data_train.shape
data_test.shape
data_train.head()
data_train.isnull().any().describe()
data_test.isnull().any().describe()
data_train.label.value_counts()
data_test.label.value_counts()
img = data_train.drop('label', axis=1).values[0].reshape(28,28)

plt.imshow(img, cmap='gray')

plt.colorbar()
img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)



X = np.array(data_train.iloc[:, 1:])

y = to_categorical(np.array(data_train.iloc[:, 0]))



# I have tried running kfold cross-validation, but the running time is longer and the performances aren't increased. 

# Therefore, I believed the dataset was large enough (especially with data augmentation) to only run a household cross-validation

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=seed)



#Test data

X_test = np.array(data_test.iloc[:, 1:])

y_test = to_categorical(np.array(data_test.iloc[:, 0]))



X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_val = X_val.astype('float32')

X_train /= 255

X_test /= 255

X_val /= 255
batch_size = 256

num_classes = 10

epochs = 75



data_generator = ImageDataGenerator(

        rotation_range = 3,

        zoom_range = 0.1,

        shear_range = 0.3,

        width_shift_range=0.08,

        height_shift_range=0.08,

        vertical_flip=False)



data_generator.fit(X_train)



reduce_lr = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
class basicConvNet():

    @staticmethod

    def build(input_shape, num_classes):

        # Builds a basic ConvNet

        # Returns Keras model object

        model = Sequential()

        

        model.add(Conv2D(32, kernel_size=(3, 3),

                         activation='relu',

                         kernel_initializer='he_normal',

                         input_shape=input_shape))

        model.add(MaxPooling2D((2, 2)))

        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation='relu'))

        model.add(Dropout(0.4))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))

        model.add(Dropout(0.3))

        model.add(Dense(num_classes, activation='softmax'))

        

        return model
class miniVGGNet():

    @staticmethod

    def build(input_shape, num_classes):

        # Builds a MiniVGGNet

        # Returns Keras model object

        model = Sequential()



        # first CONV => RELU => CONV => RELU => POOL layer set

        model.add(Conv2D(32, (3, 3), padding="same",

            input_shape=input_shape))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(32, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=-1))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))

 

        # second CONV => RELU => CONV => RELU => POOL layer set

        model.add(Conv2D(64, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(64, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=-1))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))

 

        # first (and only) set of FC => RELU layers

        model.add(Flatten())

        model.add(Dense(512))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(Dropout(0.5))

 

        # softmax classifier

        model.add(Dense(num_classes))

        model.add(Activation("softmax"))

 

        # return the constructed network architecture

        return model
# model = basicConvNet.build(input_shape, num_classes)

model = miniVGGNet.build(input_shape, num_classes)



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])

model.summary()
history = model.fit_generator(data_generator.flow(X_train, y_train, batch_size = batch_size), 

                              epochs = epochs, 

                              validation_data = (X_val, y_val),

                              verbose=1, 

                              steps_per_epoch=X_train.shape[0] // batch_size,

                              callbacks = [reduce_lr])
%matplotlib inline



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Model Loss")

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['Train', 'Test'])

plt.show()



plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title("Model Accuracy")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['Train', 'Test'])

plt.show()
scores = model.evaluate(X_test, y_test, verbose=0)

print('Loss on test dataset:', scores[0])

print('Accuracy on test dataset:', scores[1])
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Oranges):

    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

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

Y_pred = model.predict(X_test)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_test,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, 

            classes = ['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot'])