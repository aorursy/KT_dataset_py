# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.model_selection import train_test_split
from skimage.color import gray2rgb
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.models import Model
from keras.applications.nasnet import NASNetMobile, NASNetLarge, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('/kaggle/input/train.csv')
data_test  = pd.read_csv('/kaggle/input/test.csv')
# quick look at the label stats
data_train.head()

########## Preprocessing ##########
# Training images

X_train = data_train.drop(labels = ["label"],axis = 1)/255.0 # Normalization
X_train = X_train.values.reshape(-1,28,28,1) # The size of the images is 28x28
# The input of the pretrained model need to be at least 32x32
offset_down  = np.zeros((np.shape(X_train)[0],4,28,1))
X_train = np.concatenate((X_train,offset_down),axis = 1)
offset_right  = np.zeros((np.shape(X_train)[0],32,4,1))
X_train = np.concatenate((X_train,offset_right),axis = 2)
# grey to rgb because required by the pretrained model
X_train = np.squeeze(np.stack((X_train,)*3,axis = -1))
# just to check
plt.imshow(X_train[7][:,:,0])

# Other method to change the size of the input ... to test later 
#X_train = [cv2.resize(x,dsize=(32,32), interpolation=cv2.INTER_CUBIC) for x in X_train]

data_gen = ImageDataGenerator() 
data_gen.fit(X_train)
X_train = preprocess_input(X_train) # Processing for the pretrained model
plt.imshow(X_train[7][:,:,0])

# Training Labels
Y_train = data_train["label"]
Y_train = to_categorical(Y_train, num_classes = 10) # 2 -> [0,0,1,0,0,0,0,0,0,0]

# Testing 
data_test = data_test.values.reshape(-1,28,28,1)

random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
def my_model():
    inputs = Input((32, 32, 3))
    base_model = NASNetMobile(include_top=False, input_shape=(32, 32, 3))
    x = base_model(inputs)
    out = Flatten()(x)
    out = Dense(10, activation="softmax")(out)
    model = Model(inputs, out)
    model.summary()
    return model

model = my_model()
batch_size = 256
X_train = np.array(X_train)
X_val = np.array(X_val)
model.compile(optimizer = 'adam',loss = "categorical_crossentropy", metrics=["accuracy"])
history = model.fit_generator(data_gen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = 10, 
                              validation_data = (X_val,Y_val),
                              verbose = 1, 
                              steps_per_epoch=X_train.shape[0] // batch_size)
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

from sklearn.metrics import confusion_matrix
import itertools
# Look at confusion matrix 
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

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 



# Predict the values from the validation dataset
Y_pred = model.predict(X_train)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_train,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 

