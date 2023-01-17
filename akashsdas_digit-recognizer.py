import itertools



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow import keras



from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten, Dense, Conv2D, Lambda, MaxPooling2D, Dropout, BatchNormalization



from tensorflow.keras.preprocessing.image import ImageDataGenerator



from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_df  = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_df.head()
test_df.head()
# Shuffling the training dataframe

train_df.sample(frac=1)



cols = list(train_df.columns)

cols.remove('label')



X = train_df[cols]

Y = train_df['label']



# Dividing X and Y into train and dev datasets

X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.1, random_state=0)



X_test = test_df[cols]
X_train = X_train.values.reshape(-1, 28, 28)

X_dev   = X_dev.values.reshape(-1, 28, 28)

X_test  = X_test.values.reshape(-1, 28, 28)
# Expanding the shape of the arrays and scaling the input data

X_train = np.expand_dims(X_train, axis=-1) / 255

X_dev   = np.expand_dims(X_dev, axis=-1) / 255

X_test  = np.expand_dims(X_test, axis=-1) / 255
plt.imshow(X_train[0].reshape((28, 28)), cmap=plt.cm.binary)
# Looking at first 25 training examples



plt.figure(figsize=(10, 10))

for i in range(25):

  plt.subplot(5, 5, i+1)

  plt.xticks([])

  plt.yticks([])

  plt.imshow(X_train[i].reshape((28, 28)), cmap=plt.cm.binary)



plt.show()
def data_augmentation(x_data, y_data, batch_size):

    datagen = ImageDataGenerator(

        featurewise_center=False,            # set input mean to 0 over the dataset

        samplewise_center=False,             # set each sample mean to 0

        featurewise_std_normalization=False, # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,                 # apply ZCA whitening

        rotation_range=10,                   # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1,                    # Randomly zoom image 

        width_shift_range=0.1,               # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,              # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,               # randomly flip images

        vertical_flip=False                  # randomly flip images

    )

    

    

    datagen.fit(x_data)

    train_data = datagen.flow(x_data, y_data, batch_size=batch_size, shuffle=True)

    

    return train_data
BATCH_SIZE = 64

aug_train_data = data_augmentation(X_train, Y_train, BATCH_SIZE)
# Neural Network Architecture

layers = [

    Conv2D(filters=96, kernel_size=(11, 11), strides=2, activation='relu', input_shape=(28, 28, 1)),

    MaxPooling2D(pool_size=(3, 3), strides=2),

    Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'),

    Flatten(),

    Dense(9216, activation='relu'),

    Dense(4096, activation='relu'),

    Dense(4096, activation='relu'),

    Dense(10, activation='softmax'),

]
model = Sequential(layers)



optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)



model.compile(

    optimizer=optimizer,

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)



callbacks = [

    tf.keras.callbacks.ReduceLROnPlateau(monitor="loss",factor=0.1, patience=2, min_lr=0.000001, verbose=1),

]



hist = model.fit(

    aug_train_data, 

    steps_per_epoch=X_train.shape[0] // BATCH_SIZE,

    batch_size=BATCH_SIZE, 

    validation_data=(X_dev, Y_dev), 

    epochs=50,

    callbacks=callbacks

)
model.summary()
plt.plot(hist.history['accuracy'][1:], label='train acc')

plt.plot(hist.history['val_accuracy'][1:], label='validation acc')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(loc='lower right')
plt.plot(hist.history['loss'][1:], label='train loss')

plt.plot(hist.history['val_loss'][1:], label='validation loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(loc='lower right')
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.YlGnBu):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    

    plt.figure(figsize=(8, 8))

    

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



# Predict the values from the development dataset

Y_pred = model.predict(X_dev)



# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred, axis=1) 



# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_dev, Y_pred_classes) 



# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# Verify the results



predictions = Y_pred



print(predictions[0]) # Confidence matrix



print('Predicted digit is: ' + str(np.argmax(predictions[0])))

print('Accuracy is: ' + str(np.max(predictions[0] * 100)) + '%')



# Actual Digit

plt.imshow(X_dev[0].reshape((28, 28)), cmap=plt.cm.binary)
# Seeing first 25 test images predictions

plt.figure(figsize=(12, 14))

for i in range(25):

  plt.subplot(5, 5, i+1)

  plt.xticks([])

  plt.yticks([])

  plt.imshow(X_dev[i].reshape((28, 28)), cmap=plt.cm.binary)



  plt.xlabel(

      'Predicted digit is: ' + str(np.argmax(predictions[i])) + 

      '\n' + 

      'Accuracy is: ' + str(np.max(predictions[i] * 100)) + '%'

    )
predictions = model.predict(X_test)
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission.head()
for i in submission.index:

    submission['Label'][i] = np.argmax(predictions[i])
submission.to_csv("sample_submission.csv", index=False)
model.save('model')       # SavedModel format

model.save('model.h5')    # HDF5 format