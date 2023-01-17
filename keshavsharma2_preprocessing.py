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
EMNIST_PATH = '/kaggle/input'
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from matplotlib.pyplot import figure

import seaborn as sns





# Set the random seed

random_seed = 2

num_classes = 52

epochs = 50 

batch_size = 86



np.random.seed(random_seed)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical 

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau





sns.set(style='white', context='notebook', palette='deep')
train = pd.read_csv("../input/emnist1/processed_balanced_test.csv")

test = pd.read_csv("../input/emnist1/processed_balanced_train.csv")
Y_train = train["label"]

Y_test = test["label"]

X_train = train.drop(labels = ["label"], axis = 1) 

X_test = test.drop(labels = ["label"], axis = 1)


X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

Y_train = to_categorical(Y_train, num_classes = num_classes)

Y_test = to_categorical(Y_test, num_classes = num_classes)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
g = plt.imshow(X_train[10][:,:,0])


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

model.add(Dense(num_classes, activation = "softmax"))


optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

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

        rotation_range=5,  

        zoom_range = 0.1, 

        width_shift_range=0.1, 

        height_shift_range=0.1,  

        horizontal_flip=False, 

        vertical_flip=False)  





datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] 

                             )
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





Y_pred = model.predict(X_val)



Y_pred_classes = np.argmax(Y_pred,axis = 1) 



Y_true = np.argmax(Y_val,axis = 1) 



confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 



figure(figsize=(14, 14))

plot_confusion_matrix(confusion_mtx, classes = range(num_classes))
results = model.predict(X_test)



results = np.argmax(results,axis = 1)



Y_test = np.argmax(Y_test,axis = 1) 

conf_mat = confusion_matrix(results, Y_test)

acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)

print('Overall accuracy: {} %'.format(acc*100))
model.save('preprocessing.h5')