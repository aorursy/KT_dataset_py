# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
df1 = pd.read_csv('/kaggle/input/fashion-mnist_train.csv',sep=',')
df2 = pd.read_csv('/kaggle/input/fashion-mnist_test.csv', sep = ',')
## Display dataframes
df1.head()
df2.head()
x_train = df1.iloc[:,1:]
y_train = df1.iloc[:,0]
x_test= df2.iloc[:,1:]
y_test=df2.iloc[:,0]

import numpy as np
x_train = np.array(x_train, dtype = 'float32')
x_test = np.array(x_test, dtype ='float32')
y_train = np.array(y_train, dtype = 'float32')
y_test = np.array(y_test, dtype ='float32')
x_train.max()
x_train.min()
x_train = x_train[:,:]/255
x_train
x_test = x_test[:,:]/255
x_test
from sklearn.cross_validation import train_test_split
x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 12345)
import matplotlib.pyplot as plt
%matplotlib inline
image1 = x_train[10,:].reshape((28,28))
plt.imshow(image1)
plt.show()
image2 = x_train[100,:].reshape((28,28))
plt.imshow(image2)
plt.show()
image_rows = 28
image_cols = 28
batch_size = 512
image_shape = (image_rows,image_cols,1)
x_train = x_train.reshape(x_train.shape[0],*image_shape)
x_test = x_test.reshape(x_test.shape[0],*image_shape)
x_validate = x_validate.reshape(x_validate.shape[0],*image_shape)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,Adam
#### Define the model
cnn_model = Sequential([
    Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = image_shape),
    MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14
    Dropout(0.2),
    Flatten(), # flatten out the layers
    Dense(32,activation='relu'),
    Dense(10,activation = 'softmax')
    
])


cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])
history = cnn_model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=100,
    verbose=1,
    validation_data=(x_validate,y_validate),
)
score = cnn_model.evaluate(x_test,y_test,verbose=0)
print('Test Loss :',score[0])
print('Test Accuracy : ',score[1])
#get the predictions for the test data
predicted = cnn_model.predict_classes(x_test)
predicted
