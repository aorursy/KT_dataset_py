import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, plot_confusion_matrix

import itertools

%matplotlib inline



from keras.utils.np_utils import to_categorical

from keras.models import Model

from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



sns.set(style='white', context='notebook', palette='deep')
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
y_train = train["label"]

X_train = train.drop(labels=["label"], axis=1)



y_train.value_counts()
X_train = X_train / 255.0

test = test / 255.0
X_train = X_train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, num_classes=10)
random_seed = 2

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=random_seed)
# Define model with Keras functional API

X_input = Input(shape=(28, 28, 1))

X = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(X_input)

X = BatchNormalization()(X)

X = Activation('relu')(X)

X = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(X)

X = BatchNormalization()(X)

X = Activation('relu')(X)



X = MaxPool2D(pool_size=(2,2))(X)

X = Dropout(0.1)(X)



X = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(X)

X = BatchNormalization()(X)

X = Activation('relu')(X)

X = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(X)

X = BatchNormalization()(X)

X = Activation('relu')(X)



X = MaxPool2D(pool_size=(2,2), strides=(2,2))(X)

X = Dropout(0.1)(X)





X = Flatten()(X)

X = Dense(units=256, activation='relu')(X)

X = BatchNormalization()(X)

X = Dropout(0.2)(X)

X = Dense(units=10, activation='softmax')(X)





model = Model(inputs=X_input, outputs=X)

model.summary()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.0001)

model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
datagen = ImageDataGenerator(

    shear_range=0.1,

    rotation_range=15,

    zoom_range = 0.1,

    width_shift_range=0.1,

    height_shift_range=0.1

)

datagen.fit(X_train)
# Fit the model

epochs = 30

batch_size = 64



history = model.fit(datagen.flow(X_train,y_train, batch_size=batch_size), epochs=epochs, callbacks=[reduce_lr], validation_data=(X_val, y_val), verbose=1)
final_loss, final_acc = model.evaluate(X_val, y_val, verbose=0)

print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
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

    

y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred, axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_val, axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10))
predictions = model.predict(test)

predictions = np.argmax(predictions, axis=1)

predictions = pd.Series(predictions, name="Label")

predictions = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predictions],axis = 1)



predictions.head()

predictions.to_csv("cnn3", index=False)