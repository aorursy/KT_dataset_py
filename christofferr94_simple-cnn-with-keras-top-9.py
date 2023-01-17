import pandas as pd

import numpy as np

import matplotlib.pyplot as plt #for plotting

from collections import Counter

from sklearn.metrics import confusion_matrix

import itertools

import seaborn as sns



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.shape)

print(test.shape)

train.head()
print(Counter(train['label']))

sns.countplot(train['label'])
x_train = (train.iloc[:,1:].values).astype('float32')

y_train = train.iloc[:,0].values.astype('int32')



x_test = test.values.astype('float32')
%matplotlib inline

plt.figure(figsize=(12,6))

x, y = 10, 4

for i in range(40):

    plt.subplot(y, x, i+1)

    plt.imshow(x_train[i].reshape((28,28)), interpolation='nearest')

plt.show()
def visualize_input(img, ax):

    ax.imshow(img, cmap='gray')

    width, height = img.shape

    thresh = img.max()/2.5

    for x in range(width):

        for y in range(height):

            ax.annotate(str(round(img[x][y],2)), xy=(y,x),

                        horizontalalignment='center',

                        verticalalignment='center',

                        color='white' if img[x][y]<thresh else 'black')



x_train=(train.iloc[:,1:].values).astype('int32')

fig = plt.figure(figsize = (12,12)) 

ax = fig.add_subplot(111)

visualize_input(x_train[1].reshape(28,28), ax)

x_train=(train.iloc[:,1:].values).astype('float32')
x_train = x_train/255.0

x_test = x_test/255.0

y_train
print('x_train shape:', x_train.shape)
X_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

X_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
num_classes = 10



y_train = keras.utils.to_categorical(y_train, num_classes)



X_train, X_val, Y_train, Y_val= train_test_split(

    X_train, y_train, test_size = 0.1, random_state = 42)
batch_size = 64

epochs = 20

input_shape = (28, 28, 1)



model = Sequential()



model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', 

                kernel_initializer = 'he_normal', input_shape = input_shape))

model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', 

                kernel_initializer = 'he_normal'))

model.add(MaxPool2D((2, 2)))

model.add(Dropout(0.20))

model.add(Conv2D(64, (3, 3), activation='relu',

                 padding='same', kernel_initializer='he_normal'))

model.add(Conv2D(64, (3, 3), activation='relu',

                 padding='same',kernel_initializer='he_normal'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu',

                 padding='same',kernel_initializer='he_normal'))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Dense(num_classes, activation = 'softmax'))

          

model.compile(loss=keras.losses.categorical_crossentropy,

                       optimizer = keras.optimizers.Adam(),

                         metrics = ['accuracy'])



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',

                                           patience = 3,

                                           verbose = 1,

                                           factor = 0.5,

                                           min_lr = 0.0001)
datagen = ImageDataGenerator(featurewise_center=False, # set input mean to 0 over the dataset

                            samplewise_center = False, # set each sample mean to 0

                            featurewise_std_normalization = False, # divide inputs by std of the dataset

                            samplewise_std_normalization = False, # divide each input by its std

                            zca_whitening = False, # apply ZCA whitening

                            rotation_range = 15, # randomly rotate images in the range (degrees, 0 to 180)

                            zoom_range = 0.1, # Randomly zoom image 

                            width_shift_range = 0.1, # randomly shift images horizontally (fraction of total width)

                            height_shift_range = 0.1, # randomly shift images vertically (fraction of total height)

                            horizontal_flip = False, # randomly flip images

                            vertical_flip = False) # randomly flip images - we do not want this as it e.g. messes up the digits 6 and 9
model.summary()
datagen.fit(X_train)

h = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = batch_size),

                       epochs = epochs,

                       validation_data = (X_val, Y_val),

                       verbose = 1,

                       steps_per_epoch = X_train.shape[0] // batch_size,

                       callbacks = [learning_rate_reduction],)
final_loss, final_acc = model.evaluate(X_val, Y_val, verbose=0)

print("validation loss: {0:.6f}, validation accuracy: {1:.6f}".format(final_loss, final_acc))
def plot_confusion_matrix(cm, classes,

                         normalize = False,

                         title = 'Confusion matrix',

                         cmap = plt.cm.Blues):

    plt.imshow(cm, interpolation = 'nearest', cmap=cmap)

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

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred, axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val, axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(num_classes))
print(h.history.keys())
accuracy = h.history['acc']

val_accuracy = h.history['val_acc']

loss = h.history['loss']

val_loss = h.history['val_loss']



epochs = range(len(accuracy))



plt.plot(epochs, accuracy, 'bo', label = 'training accuracy')

plt.plot(epochs, val_accuracy, 'b', label = 'validation accuracy')



plt.title('training and validation accuracy')



plt.figure()

plt.plot(epochs, loss, 'bo', label = 'training loss')

plt.plot(epochs, val_loss, 'b', label = 'validation loss')

plt.title('training and validation loss')
predicted_classes = model.predict_classes(X_test)
submissions = pd.DataFrame({"ImageId": list(range(1, len(predicted_classes)+1)),

                           "Label": predicted_classes})

submissions.to_csv("mnistSubmission.csv", index = False, header = True)
model.save('my_model_1.h5')



json_string = model.to_json()