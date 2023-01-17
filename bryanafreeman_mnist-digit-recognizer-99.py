import itertools

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import warnings



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.utils.np_utils import to_categorical

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



SEED = 7

SPLIT = 0.1

EPOCHS = 40

BATCH_SIZE = 100

np.random.seed(SEED)

warnings.filterwarnings('ignore')

%matplotlib inline
# Read the CSV

mnist_train_path = '../input/digit-recognizer/train.csv'

mnist_test_path = '../input/digit-recognizer/test.csv'



train_original = pd.read_csv(mnist_train_path)

test_original = pd.read_csv(mnist_test_path)



# Preserve original

X_train = train_original.drop('label', axis=1).copy()

X_test = test_original.copy()

y_train = train_original['label'].copy()

print('Train data shape: {}'.format(X_train.shape))

print('Test data shape: {}'.format(X_test.shape))
fig, ax = plt.subplots()

fig.set_size_inches(11, 8)

fig=sns.countplot(y_train)

plt.xlabel('Labels')

plt.ylabel('Count')

plt.title('Distribution of Labels')

plt.show(fig)
y_train.value_counts()
X_train = X_train / 255.0

X_test = X_test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)
fig, ax = plt.subplots()

fig.set_size_inches(11, 8)

fig = plt.imshow(X_train[1][:,:,0])

plt.title('Sample Image')

plt.show(fig)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,

                                                  test_size = SPLIT,

                                                  random_state=SEED)
y_train = to_categorical(y_train, num_classes=10)

y_val = to_categorical(y_val, num_classes=10)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
# Get optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)



# Compile the model

model.compile(optimizer = optimizer , loss = 'categorical_crossentropy', metrics=['accuracy'])



# Make learning rate annealer

lrr = ReduceLROnPlateau(monitor='val_acc', 

                        patience=3, 

                        verbose=1, 

                        factor=0.5, 

                        min_lr=0.00001)
datagen = ImageDataGenerator(

        featurewise_center=False,

        samplewise_center=False,

        featurewise_std_normalization=False,

        samplewise_std_normalization=False,

        zca_whitening=False,

        rotation_range=10,

        zoom_range = 0.12, 

        width_shift_range=0.12,

        height_shift_range=0.12,

        horizontal_flip=False,

        vertical_flip=False)



datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=BATCH_SIZE),

                              epochs = EPOCHS,

                              validation_data = (X_val,y_val),

                              verbose = 2,

                              steps_per_epoch=X_train.shape[0] // BATCH_SIZE,

                              callbacks=[lrr])
fig, ax = plt.subplots(2,1)

fig.set_size_inches(11, 8)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="Validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Helper method for confusion matrix

def plot_confusion_matrix(confusion_mat, classes):

    fig, ax = plt.subplots()

    fig.set_size_inches(11, 8)

    fig = plt.imshow(confusion_mat, interpolation='nearest')

    plt.title('Confusion matrix')

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)

    plt.yticks(tick_marks, classes)

    plt.ylabel('Actual label')

    plt.xlabel('Predicted label')

    plt.show(fig)
y_pred = model.predict(X_val)

y_pred_classes = np.argmax(y_pred,axis = 1) 

y_actual = np.argmax(y_val,axis = 1) 

confusion_mat = confusion_matrix(y_actual, y_pred_classes) 

plot_confusion_matrix(confusion_mat, classes = range(10)) 
results = model.predict(X_test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name='Label')

sub = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

sub.to_csv('mnist_submission.csv',index=False)