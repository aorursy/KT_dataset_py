import pandas as pd

import numpy as np

from zipfile import ZipFile

import matplotlib.pyplot as plt

import collections

import itertools

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_data = pd.read_csv('../input/digit-recognizer/test.csv')

train_data = pd.read_csv('../input/digit-recognizer/train.csv')
labels = train_data.label.values

label_count = collections.Counter(labels)

plt.bar(label_count.keys(), label_count.values())
train_data.drop(columns={'label'}, inplace=True)

train_data /= 255.

test_data /= 255.
test_images = np.reshape(test_data.values, (test_data.shape[0],28,28,1))

train_images = np.reshape(train_data.values, (train_data.shape[0],28,28,1))

test_images[0].shape
plt.figure(figsize=(20,10))

columns = 5

for i, image in enumerate(train_images[:20]):

    plt.subplot(len(train_images[:20]) / columns + 1, columns, i + 1)

    plt.imshow(image[:,:,0])
X_train, X_val, Y_train, Y_val = train_test_split(train_images, labels, test_size=0.2, random_state=4)
def create_model(print_summary=False):

    model = Sequential()



    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))

    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))

    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))

    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))

    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Dropout(0.25))

    

    model.add(Flatten())

    model.add(Dense(256, activation = "relu"))

    model.add(Dropout(0.25))

    model.add(Dense(10, activation = "softmax"))

    

    opt = Adam(lr=0.001)

    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    

    if print_summary:

        model.summary()

    return model 
model = create_model(print_summary=True)

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=0.00001)
datagen = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=0.1)



datagen.fit(X_train)
BS = 128 # batch size

history = model.fit(datagen.flow(X_train, Y_train, batch_size=BS), steps_per_epoch= X_train.shape[0] // BS, 

                    validation_data=(X_val, Y_val), epochs=100, verbose=2, callbacks=[reduce_lr])
plt.figure(figsize=(20,10))

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

ax[0].legend()



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

ax[1].legend()
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):



    plt.figure(figsize=(20,10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(-0.5, len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = np.round(100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],1)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')





# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# # Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_val, Y_pred_classes)

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10), normalize=True)
print(classification_report(Y_val, Y_pred_classes))
results = model.predict(test_images)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



submission = pd.DataFrame({

    'ImageID': range(1, len(results) + 1),

    'Label': results

})

submission.to_csv("digit_recognizer_cnn.csv",index=False)