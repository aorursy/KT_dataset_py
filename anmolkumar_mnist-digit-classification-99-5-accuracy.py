# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import keras

from keras.models import Sequential

from keras.utils import to_categorical

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras import optimizers

import tensorflow as tf

# tf.logging.set_verbosity(tf.logging.ERROR)



import itertools

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from keras.layers import Dense

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D # to add convolutional layers

from keras.layers.convolutional import MaxPooling2D # to add pooling layers

from keras.layers import Flatten # to flatten data for fully connected layers

from keras.layers import Dropout # to avoid overfitting

from keras.layers import Activation
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()
def plot_loss_accuracy(history):

    historydf = pd.DataFrame(history.history, index=history.epoch)

    plt.figure(figsize=(8, 6))

    historydf.plot(ylim=(0, max(1, historydf.values.max())))

    Val_loss = history.history['val_loss'][-1]

    Val_acc = history.history['val_accuracy'][-1]

    Train_loss = history.history['loss'][-1]

    Train_acc = history.history['accuracy'][-1]

    plt.title('Validation Loss: %.3f, Validation Accuracy: %.3f, Training Loss: %.3f, Training Accuracy: %.3f' % (Val_loss, Val_acc, Train_loss, Train_acc))
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

train_df.head()
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test_df.head()
columns = train_df.columns

X = train_df[columns[columns != 'label']]

y = train_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 4)
X_train.shape
X_test.shape
sns.countplot(y_train)
X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1).astype('float32')

X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
X_train = X_train / 255 # normalize training data

X_test = X_test / 255 # normalize test data
y_train = to_categorical(y_train)

y_test = to_categorical(y_test)



num_classes = y_test.shape[1] # number of categories
def convolutional_model():

    ADAMAX = optimizers.Adamax(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999)

    # create model

    model = Sequential()

    model.add(Conv2D(32, (4, 4), activation = 'relu', input_shape = (28, 28, 1)))

    model.add(Conv2D(64, (3, 3), activation = 'relu'))

    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(BatchNormalization(axis = -1))

    model.add(Dropout(0.2))

    

    model.add(Conv2D(128, (3, 3), activation = 'relu'))

    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(BatchNormalization(axis = -1))

    model.add(Dropout(0.2))

    

    model.add(Conv2D(128, (2, 2), activation = 'relu'))

    model.add(Conv2D(256, (2, 2), activation = 'relu'))

    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

    model.add(BatchNormalization(axis = -1))

    model.add(Dropout(0.2))

    

    model.add(Flatten())

    model.add(Dense(256, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dense(128, activation = 'relu'))

    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation = 'softmax'))

    

    # Compile model

    model.compile(optimizer = ADAMAX, loss = 'categorical_crossentropy',  metrics = ['accuracy'])

    return model
gen = ImageDataGenerator(rotation_range = 12, width_shift_range = 0.1, shear_range = 0.1,

                         height_shift_range = 0.1, zoom_range = 0.1, fill_mode = 'nearest', horizontal_flip = False,

                         vertical_flip = False, featurewise_center = False,

                         samplewise_center = False, featurewise_std_normalization = False,

                         samplewise_std_normalization = False)

test_gen = ImageDataGenerator()



# Create batches to  train models faster

train_generator = gen.flow(X_train, y_train, batch_size = 32)

test_generator = test_gen.flow(X_test, y_test, batch_size = 32)
# Use annelar to gradually decrese the learning rate to improve generalization



reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 3, verbose = 1, factor = 0.4, min_lr = 0.00002,

                                            mode = 'auto', cooldown = 0)
# build the model

model = convolutional_model()

epochs = 80

# fit the model



#history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 30, batch_size = 200, verbose = 2)



history = model.fit_generator(train_generator, steps_per_epoch = 40000//16, epochs = epochs, 

                              validation_data = test_generator, validation_steps = 10000//8, verbose = 1,

                              callbacks=[reduce_lr])



# evaluate the model

scores = model.evaluate(X_test, y_test, verbose = 0)

print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))

plot_loss_accuracy(history)
plot_loss_accuracy(history)
y_pred = np.argmax(model.predict(X_test), axis = 1)

Y_true = np.argmax(y_test, axis = 1)

cm = confusion_matrix(Y_true, y_pred)

plot_confusion_matrix(cm, classes = range(10))
test_data = test_df.values.reshape(test_df.shape[0], 28, 28, 1).astype('float32')

test_data = test_data / 255

Y_pred = model.predict(test_data)
Y_pred = np.argmax(Y_pred,axis = 1)

Y_pred = pd.Series(Y_pred, name = "Label")
submission_df = pd.DataFrame({

                  "ImageId": pd.Series(range(1, len(Y_pred)+1)),

                  "Label": pd.Series(Y_pred)})
submission_df.to_csv('/kaggle/working/Submission.csv', index = False)