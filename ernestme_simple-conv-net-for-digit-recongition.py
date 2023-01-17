import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Lambda, Reshape, MaxPooling2D
from keras.optimizers import SGD, Adam


import pandas as pd
import numpy as np

import matplotlib.pyplot as pyplot
%matplotlib inline

train_data = pd.read_csv("../data/train.csv");
test_data = pd.read_csv("../data/test.csv");




# papare the data
X_train = train_data.iloc[:,1:].values.astype("float32")
y_train = train_data.iloc[:,0].values.astype("int")
X_test = test_data.values.astype("float32")



# reshape imge to 28*28
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1 )
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# zero-mean
X_mean = X_train.mean().astype('float32')
X_std = X_train.std().astype('float32')

X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# convert the y annotation to binary array
y_train = keras.utils.to_categorical(y_train, 10);
# visualize
for i in range(9):
    pyplot.subplot(3,3,i+1)
    pyplot.imshow(X_train[i,:,:].reshape(28,28));
    pyplot.title(np.argmax(y_train[i]))
# model
model = Sequential()
model.add(Conv2D(64, (3,3),input_shape=(28,28,1),activation='relu', bias_initializer='RandomNormal'))
model.add(Conv2D(64, (3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(64, (3,3),activation='relu'))
model.add(Conv2D(64, (3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
keras.utils.print_summary(model)
# train
model.fit(X_train,y_train,epochs=50, batch_size = 64);
# predict
results = model.predict(X_test)
results = np.argmax(results,1)
output = pd.Series(results,name="Label")
output = pd.concat([pd.Series(range(1,len(results)+1),name="ImageID"), output], axis = 1)
print(output)
output.to_csv('../results/predict.csv', index=False)


