# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
num_classes = 10
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv',sep=',')
test_df = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv',sep=',')
train_df.head()
train_data = np.array(train_df, dtype = 'float32')
test_data = np.array(test_df, dtype = 'float32')
x_train = train_data[:,1:]/255
x_test = test_data[:,1:]/255

y_train = train_data[:,0]
y_test = test_data[:,0]
x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 210196)
image = x_train[4500,:].reshape((28,28))
plt.imshow(image)
plt.show()
image_rows = 28
image_cols = 28
batch_size = 512
image_shape = (image_rows,image_cols,1)
x_train = x_train.reshape(x_train.shape[0],*image_shape)
x_test = x_test.reshape(x_test.shape[0],*image_shape)
x_validate = x_validate.reshape(x_validate.shape[0],*image_shape)
cnn_model = Sequential([
    Conv2D(filters = 32,kernel_size = 3,activation="relu",input_shape = image_shape),
    MaxPooling2D(pool_size  = 2),
    Dropout(0.2),
    Flatten(),
    Dense(32,activation = "relu"),
    Dense(10,activation = "softmax")
])
cnn_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(lr = 0.001),metrics = ['accuracy'])
history = cnn_model.fit(
    x_train,
    y_train,
    batch_size = batch_size,
    epochs = 50,
    verbose = 1,
    validation_data = (x_validate, y_validate)
)
score = cnn_model.evaluate(x_test,y_test,verbose = 0)
print('Test Loss : {:.4f}'.format(score[0]))
print('Test Accuracy {:.4f}'.format(score[1]))
import matplotlib.pyplot as plt
%matplotlib inline
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss,'bo', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.legend()
plt.show
predicted_classes = cnn_model.predict_classes(x_test)
y_true = test_df.iloc[:,0]
correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]
from sklearn.metrics import classification_report
target_names = ["Class {}". format(i) for i in range(num_classes)]
print(classification_report(y_true, predicted_classes, target_names = target_names))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[correct].reshape(28,28),cmap = 'gray', interpolation = 'none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_true[correct]))
    plt.tight_layout()
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[incorrect].reshape(28,28),cmap = 'gray', interpolation = 'none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_true[incorrect]))
    plt.tight_layout()