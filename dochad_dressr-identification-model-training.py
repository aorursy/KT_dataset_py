# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
num_classes = 10
epochs = 5
from sklearn.model_selection import train_test_split
import os
print(os.listdir("../input/"))
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/fashion-mnist_train.csv")
test = pd.read_csv("../input/fashion-mnist_test.csv")
test2 = pd.read_csv("../input/fashion-mnist_test.csv")
train.head()
train_data = np.array(train, dtype = 'float32')
test_data = np.array(test, dtype='float32')

x_train = train_data[:,1:]/255

y_train = train_data[:,0]

x_test= test_data[:,1:]/255

y_test=test_data[:,0]
x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 12345)
image = x_train[200,:].reshape((28,28))
plt.imshow(image)
plt.show()
image_rows = 28
image_cols = 28
batch_size = 512

image_shape = (image_rows,image_cols,1) # Defined the shape of the image as 3d with rows and columns and 1 for the 3d visualisation
x_train = x_train.reshape(x_train.shape[0],*image_shape)/255
x_test = x_test.reshape(x_test.shape[0],*image_shape)/255
x_validate = x_validate.reshape(x_validate.shape[0],*image_shape)/255
cnn_model = Sequential()
cnn_model.add(Conv2D(20, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(image_rows, image_cols, 1))) #1 here shows the channel, since this was a greyscale, it had only 1, if it was a colored image, it would be 3.#
cnn_model.add(Dropout(0.2))
cnn_model.add(Flatten())
cnn_model.add(Dense(50, activation='relu'))
cnn_model.add(Dense(10, activation='softmax'))
cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])
history = cnn_model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=50,
    verbose=1,
    validation_data=(x_validate,y_validate),
)
score = cnn_model.evaluate(x_test,y_test,verbose=0)
print('Test Loss : {:.4f}'.format(score[0]))
print('Test Accuracy : {:.4f}'.format(score[1]))
import matplotlib.pyplot as plt

%matplotlib inline

accuracy = history.history['acc']

val_accuracy = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')

plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')

plt.title('Training and Validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
#get the predictions for the test data

predicted_classes = cnn_model.predict_classes(x_test)

#get the indices to be plotted

y_true = test.iloc[:, 0]

correct = np.nonzero(predicted_classes==y_true)[0]

incorrect = np.nonzero(predicted_classes!=y_true)[0]

from sklearn.metrics import classification_report

target_names = ["Class {}".format(i) for i in range(num_classes)]

print(classification_report(y_true, predicted_classes, target_names=target_names))
cnn_model.save("Dressr_output.h5")
cnn_model.save("Dress_output.json")
