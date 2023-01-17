# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test  = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
print(train.columns)
y_train = train["label"]

x_train = train.drop(labels=["label"],axis =1)



y_train.value_counts()
x_train = x_train/255.0

test = test/255.0



print(x_train["pixel454"])
x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

print("x_train shape: ",x_train.shape)

print("test shape: ",test.shape)



y_train = to_categorical(y_train, num_classes =10)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size = 0.1, random_state=2)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
learning_rate = 0.001

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer = optimizer,loss = "categorical_crossentropy",

             metrics=["accuracy"])
epochs = 2

batch_size = 250
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(x_train)
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                             epochs = epochs, validation_data = (x_val,y_val),

                             steps_per_epoch = x_train.shape[0] // batch_size)
# Plot the loss and accuracy curves for training and validation 

plt.plot(history.history['val_loss'], color='b', label="validation loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
# Confusion matrix e bakalım 



import seaborn as sns

# Modelimzi predict ediyoruz

Y_pred = model.predict(x_val)

# 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# 

Y_true = np.argmax(y_val,axis = 1) 

# 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 



# ve çizdirelim

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
# Yanlış tahmin edilen bazı örnekleri görelim





errors = (Y_pred_classes - Y_true != 0)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

X_val_errors = x_val[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    n = 0

    nrows = 2

    ncols = 3

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1



Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)



true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))



delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors



sorted_dela_errors = np.argsort(delta_pred_true_errors)



most_important_errors = sorted_dela_errors[-6:]



display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
# test edelim

results = model.predict(test)



results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)