import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))
train_df = pd.read_csv('../input/fashion-mnist_train.csv')
test_df = pd.read_csv('../input/fashion-mnist_test.csv')
train_df.head()

train_data = np.array(train_df,dtype = 'float32')
test_data = np.array(test_df,dtype = 'float32')

x_train = train_data[:,1:] / 255
y_train = train_data[:,0]

x_test = test_data[:,1:] / 255
y_test = test_data[:,0]


x_train , x_validate , y_train , y_validate = train_test_split(
  x_train,y_train,test_size=0.2,random_state=12345, 
)
image = x_train[500,:].reshape((28,28))

plt.imshow(image)
plt.show()
im_rows = 28
im_cols = 28

batch_size = 512
im_shape = (im_rows,im_cols,1)

x_train = x_train.reshape(x_train.shape[0],*im_shape)
x_test = x_test.reshape(x_test.shape[0],*im_shape)
x_validate = x_validate.reshape(x_validate.shape[0],*im_shape)

print('x_train shape {}'.format(x_train.shape))
print('x_test shape {}'.format(x_test.shape))
print('x_validate shape {}'.format(x_validate.shape))
cnn_model = Sequential([
    Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=im_shape),
    MaxPooling2D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(32,activation='relu'),
    Dense(10,activation='softmax')
])
cnn_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(lr=0.001),
    metrics=['accuracy']
)
cnn_model.fit(
    x_train,y_train,batch_size=batch_size,
    epochs=10,verbose=1,
    validation_data=(x_validate,y_validate)
)
score = cnn_model.evaluate(x_test,y_test,verbose=0)

print('test loss: {:.4f}'.format(score[0]))
print('test accu: {:.4f}'.format(score[1]))
