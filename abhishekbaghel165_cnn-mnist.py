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
import keras as K

from keras.layers import Input,Dense,Dropout,Conv2D,MaxPooling2D,ZeroPadding2D,BatchNormalization,Activation,Flatten

from keras.models import Model,Sequential

from keras.utils import to_categorical



from sklearn.model_selection import train_test_split
df_train = pd.read_csv('../input/fashion-mnist_train.csv')

df_test = pd.read_csv('../input/fashion-mnist_test.csv')



train_x = np.array(df_train.iloc[:, 1:])

train_y = to_categorical(np.array(df_train.iloc[:, 0]))



X_train, X_val, y_train, y_val  = train_test_split(train_x,train_y, test_size = 0.2)



X_test = np.array(df_test.iloc[:, 1:])

y_test = to_categorical(np.array(df_test.iloc[:, 0]))



X_train = X_train.reshape((-1, 28,28,1))

X_val = X_val.reshape((-1, 28,28,1))

X_test = X_test.reshape((-1,28,28,1))



X_train = X_train.astype('float32')

X_val = X_val.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255

X_val /=255
epochs = 20

batch_size = 64

input_shape = [28,28,1]
model = Sequential()

model.add(Conv2D(24,kernel_size = (5,5), padding = 'same', activation = 'relu',input_shape = (28,28,1), name = 'conv0'))

model.add(MaxPooling2D())

model.add(Conv2D(48,kernel_size = (5,5), padding = 'same', activation = 'relu', name = 'conv1'))

model.add(MaxPooling2D())    

model.add(Flatten())

model.add(Dense(250,activation = 'relu',name = 'fc0'))

model.add(Dropout(0.5))

model.add(Dense(10,activation = 'softmax',name = 'fc1'))

    

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  
hist = model.fit(X_train,y_train,epochs = epochs,verbose = 1, validation_data = (X_val,y_val))
import matplotlib.pyplot as plt

%matplotlib inline



accuracy = hist.history['acc']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'r-', label = "Training accuracy")

plt.title("Training accuracy")

plt.legend()

plt.show()

plt.figure()

val_acc = hist.history['val_acc']

plt.plot(epochs,val_acc, 'b-', label = 'Validation accuracy')

plt.legend()

plt.show()
acc = model.evaluate(X_test,y_test)

print("Test accuracy is : " + str(acc[1]))

print("Test loss is : " + str(acc[0]))
#Visualizations
example = X_test[1367]

plt.imshow(example.reshape((28,28)), cmap = 'viridis' ,interpolation = 'none')
print(type(model.layers[1]))

print(type(MaxPooling2D()))
#Activation of the first hidden layer

activations = [layer.output for layer in model.layers]



activation_model = Model(inputs = model.input, outputs = activations)



activation_tensors = activation_model.predict(example.reshape(1,28,28,1))
plt.imshow(activation_tensors[0][:,:,:,8].reshape(28,28), cmap = 'viridis')