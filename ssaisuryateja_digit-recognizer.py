import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
# divide training data into features and labels

X_train = train.iloc[:,1:]

y_train = train.iloc[:,0]
# Reshape and normalize image

X_train = X_train.values.reshape(-1, 28, 28, 1)/255.

test = test.values.reshape(-1, 28, 28, 1)/255.

# One Hot encoding the label

y_train = to_categorical(y_train, 10)
random_seed = 0

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_seed)
datagen = ImageDataGenerator(

            rotation_range=10,

            width_shift_range=0.1,

            height_shift_range=0.1,

            zoom_range=0.1

            )
model = Sequential()



model.add(Conv2D(32, (5,5), padding='same', input_shape=X_train.shape[1:], activation='relu'))

model.add(Conv2D(32, (5,5), padding='same', activation='relu'))

model.add(MaxPool2D(2,2))



model.add(Conv2D(64, (3,3), padding='same', activation='relu'))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))

model.add(MaxPool2D(2,2))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))



model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
EPOCHS = 30

BATCH_SIZE = 20

callback_list = [

    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1),

    EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=4)

]



history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),

                   epochs=EPOCHS,

                   callbacks=callback_list,

                   validation_data=(X_val, y_val),

                   steps_per_epoch=X_train.shape[0] // BATCH_SIZE)
loss = history.history['loss']

val_loss = history.history['val_loss']



fig, ax = plt.subplots(figsize=(12,4))

ax.plot(loss, 'b', label='Training loss')

ax.plot(val_loss, 'r', label='Validation loss')

ax.legend()
def plot_confustion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Oranges):

    plt.figure(figsize=(10,7))

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

        plt.text(j, i , cm[i,j],

                horizontalalignment='center',

                color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()

    plt.xlabel('True label')

    plt.ylabel('Predicted label')
y_pred = model.predict(X_val)

y_pred_classes = np.argmax(y_pred, axis=1)

y_real_classes = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_pred_classes, y_real_classes)

plot_confustion_matrix(cm, classes=range(10))
results = model.predict(test)

results = np.argmax(results, axis=1)

results = pd.Series(results, name='Label')

submission = pd.concat([pd.Series(range(1,28001), name='ImageID'), results], axis=1)

submission.to_csv('submission.csv', index=False)