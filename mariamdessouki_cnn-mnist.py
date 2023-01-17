
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
y_train = train['label']
x_train = train.drop(labels='label',axis=1)


x_train /= 255
test /= 255
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train,10)
from sklearn.model_selection import train_test_split
#90% of the data is used for training and 10% is used for validation
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train, test_size = 0.1, random_state=15)

from matplotlib import pyplot as plt
f = plt.figure()
f.add_subplot(1,3,1)
plt.imshow(x_train[0][:,:,0])
f.add_subplot(1,3,2)
plt.imshow(x_train[1000][:,:,0])
f.add_subplot(1,3,3)
plt.imshow(x_train[10000][:,:,0])
print("The below figure shows 3 random training examples from the mnist dataset")
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5), 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3), 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


from keras.optimizers import RMSprop
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer = optimizer,loss = "categorical_crossentropy", metrics=["accuracy"])
from keras.callbacks import ReduceLROnPlateau
lrr = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
model.fit(x_train, y_train, batch_size = 100, epochs = 30,    
          validation_data = (x_test, y_test), verbose = 2)
results = model.predict(test)

#print(results)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)