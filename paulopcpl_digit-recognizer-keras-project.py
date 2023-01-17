import math

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



np.random.seed(2099)



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import EarlyStopping



%matplotlib inline

sns.set(style='white', context='notebook', palette='deep')

plt.rcParams["figure.figsize"] = (15,7) # plot size
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
Y_train = train['label'].copy()

X_train = train.drop('label', axis = 1).copy()

del train # to free memory.
sns.countplot(Y_train)
Y_train.value_counts()
X_train = X_train/255.0

test = test/255.0
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2099)
X_train.shape
X_dev.shape
datagen = ImageDataGenerator(

        featurewise_center = False,                  # set input mean to 0 over the dataset

        samplewise_center = False,                   # set each sample mean to 0

        featurewise_std_normalization = False,       # divide inputs by std of the dataset

        samplewise_std_normalization = False,        # divide each input by its std

        zca_whitening = False,                       # apply ZCA whitening

        rotation_range = 10,                         # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1,                            # Randomly zoom image 

        width_shift_range = 0.1,                     # randomly shift images horizontally (fraction of total width)

        height_shift_range = 0.1,                    # randomly shift images vertically (fraction of total height)

        horizontal_flip = False,                     # randomly flip images

        vertical_flip = False)                       # randomly flip images



# compute quantities required for featurewise normalization

datagen.fit(X_train)
# Set the CNN model 

model = Sequential()



model.add(Conv2D(filters = 10, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(rate = 0.75))



model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(rate = 0.75))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(rate = 0.5))

model.add(Dense(10, activation = "softmax"))
# Compile the model

model.compile(optimizer = 'adam' , loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
epochs = 3

batch_size = 128
# Fit the model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_dev,Y_dev),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
Y_pred = model.predict_classes(X_dev)
df_cm = pd.DataFrame(confusion_matrix(Y_dev, Y_pred), index = [i for i in range(10)], columns = [i for i in range(10)])

ax = sns.heatmap(df_cm, annot = True, fmt = 'd', cmap = "Blues")

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)

plt.xlabel('Predicted Value')

plt.ylabel('True Value')

plt.title('Confusion Matrix', size = 30)
# predict results

results = model.predict_classes(test)

results = pd.Series(results,name="Label")
# Prepare the submission.

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv", index = False)