# Import libraries necessary for this project

# Input data files are available in the "../input/" directory.
import numpy  as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
%matplotlib inline

from keras.utils.np_utils    import to_categorical
from keras.models            import Sequential
from keras.layers            import Dense
from keras.layers            import Dropout
from keras.layers            import Conv2D
from keras.layers            import Activation
from keras.layers            import MaxPooling2D
from keras.layers            import Flatten
from sklearn.model_selection import train_test_split
from matplotlib.pyplot       import imshow
from PIL                     import Image

print("Input data : ",os.listdir("../input"))

print("\nImporting ✓\n")
# Load the MNIST dataset

train_data = pd.read_csv('../input/train.csv')
test_data  = pd.read_csv('../input/test.csv' )

y_train = train_data["label"]
x_train = train_data.drop(labels=["label"], axis=1) 

print("\nLoading ✓\n")
print("There are", x_train.shape[0] , "training examples.")
print("And", test_data.shape[0], "testing examples.")
print(x_train.shape)
print("\nThe classes we have are :",np.unique(y_train))

%matplotlib inline
# everytime you run you'll get another data point 
random = np.random.randint(0, 42000)
print ("\nThe class you got is",y_train[random])
imshow(x_train.iloc[random].values.reshape((28, 28)))
print("\nExploring ✓\n")
#  Normalizing the data to be in range from 0 to 1 instead of 0 to 255
x_train   =   x_train.values.reshape(-1, 28, 28, 1) / 255 
test_data = test_data.values.reshape(-1, 28, 28, 1) / 255
print(x_train.shape)
print(test_data.shape)
# Splitting the output to 10 distinct calsses
y_train = to_categorical(y_train, num_classes = 10)
print(y_train.shape)

print("\nReshaping ✓\n")
model = Sequential()

model.add(Conv2D(128, padding='same',data_format='channels_last', kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, padding='same', kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, padding='same', kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, padding='same', kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())

# Output layer
model.add(Dense(units=10, activation='softmax'))

# compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\nCompiling ✓\n")
model.fit(x = x_train, y = y_train, epochs=32, verbose=2)
print("\nTraining ✓\n")
res = model.predict(test_data)
res = np.argmax(res,axis = 1)
res = pd.Series(res, name="Label")

submission = pd.concat([pd.Series(range(1 ,28001) ,name = "ImageId"),   res],axis = 1)
submission.to_csv("cnn_mnist_class.csv",index=False)

print("\nSubmission ✓\n")
