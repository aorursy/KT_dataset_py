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
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import keras

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard

num_classes = 10

train_df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv',sep=',')

test_df= pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv', sep = ',')

print("Train size:{}\nTest size:{}".format(train_df.shape, test_df.shape))
train_df.head()
train_data = np.array(train_df, dtype = 'float32')

test_data = np.array(test_df, dtype='float32')

x_train = train_data[:,1:]/255



y_train = train_data[:,0]



x_test= test_data[:,1:]/255



y_test=test_data[:,0]
x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 12345)
image = x_train[22,:].reshape((28,28))

plt.imshow(image)

plt.show()

image_rows = 28



image_cols = 28



batch_size = 512



image_shape = (image_rows,image_cols,1) # Defined the shape of the image as 3d with rows and columns and 1 for the 3d visualisation
x_train = x_train.reshape(x_train.shape[0],*image_shape)

x_test = x_test.reshape(x_test.shape[0],*image_shape)

x_validate = x_validate.reshape(x_validate.shape[0],*image_shape)
cnn_model = Sequential([

    Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = image_shape),

    MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14

    Dropout(0.5),

    Flatten(), # flatten out the layers

    Dense(32,activation='relu'),

    Dense(10,activation = 'softmax')

    

])
cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])
history = cnn_model.fit(

    x_train,

    y_train,

    batch_size=batch_size,

    epochs=17,

    verbose=1,

    validation_data=(x_validate,y_validate),

)
score = cnn_model.evaluate(x_test,y_test,verbose=0)

print('Test Loss : {:.4f}'.format(score[0]))

print('Test Accuracy : {:.4f}'.format(score[1]))
plt.figure(1)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Complexity Graph:  Training vs. Validation Loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validate'], loc='upper right')



plt.figure(2)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy Graph:  Training vs. Validation accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validate'], loc='upper right')

plt.show()


predicted_classes = cnn_model.predict_classes(x_test)



#get the indices to be plotted



y_true = test_df.iloc[:, 0]



correct = np.nonzero(predicted_classes==y_true)[0]



incorrect = np.nonzero(predicted_classes!=y_true)[0]



from sklearn.metrics import classification_report



target_names = ["Class {}".format(i) for i in range(num_classes)]



print(classification_report(y_true, predicted_classes, target_names=target_names))
classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num_rows = 5

num_cols = 10

sample_size = num_rows * num_cols

indices = np.arange(sample_size)

x_pred = x_test[indices,:,:]

predictions = cnn_model.predict(x_pred)

x_pred = np.squeeze(x_test[indices,:,:])

y_pred = np.argmax(predictions,axis=1)



num_images = num_rows*num_cols

plt.figure(figsize=(num_cols*2, num_rows*2))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)

for i in range(num_images):

  plt.subplot(num_rows, num_cols, i+1)

  plt.imshow(x_pred[i])

  plt.title(classes[y_pred[i]])

  # plt.subplot(num_rows, 2*num_cols, 2*i+2)

  # plot_value_array(i, predictions, test_labels)

plt.show()