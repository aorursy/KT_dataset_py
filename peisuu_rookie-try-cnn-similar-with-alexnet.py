import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

from sklearn.preprocessing import OneHotEncoder
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape, test.shape
train.head(5)
test.head(5)
y_train = train['label']
x_train = train.drop('label', axis=1)
y_train.shape, x_train.shape
x_train.head(5)
y_train.value_counts()
x_train.isnull().any().describe()
test.isnull().any().describe()
x_train = x_train/255.0
test = test/255.0
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
test.shape
# Use the Scikit to make one-hot encoding label
y_train = pd.DataFrame(data=y_train)
one_hot = OneHotEncoder(handle_unknown='ignore')
one_hot.fit(y_train.values)
y_train = one_hot.transform(y_train.values).toarray()
y_train, y_train.shape
from sklearn.model_selection import train_test_split
random_seed = 3
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed)
x_train.shape, x_val.shape, y_train.shape, y_val.shape
g = plt.imshow(x_train[0][:,:,0])
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

model = Sequential([
    Conv2D(filters = 64, input_shape=(28,28,1), kernel_size=(3,3), strides=(1,1), padding='valid'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'),
    BatchNormalization(),
    
    Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='valid'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'),
    BatchNormalization(),
    
    Conv2D(filters=192, kernel_size=(1,1), strides=(1,1), padding='valid'),
    Activation('relu'),
    BatchNormalization(),
    
    Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='valid'),
    Activation('relu'),
    BatchNormalization(),

    Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='valid'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'),
    BatchNormalization(),
    
    Flatten(),
    Dense(2048),
    Activation('relu'),
    Dropout(0.4),
    BatchNormalization(),
    
    Dense(2048),
    Activation('relu'),
    Dropout(0.4),
    BatchNormalization(),

    Dense(800),
    Activation('relu'),
    Dropout(0.4),
    BatchNormalization(),
    
    Dense(10),
    Activation('softmax'),
])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=200, validation_data=(x_val,y_val), epochs = 10)
results = model.predict(test)
results = np.argmax(results, axis=1)
# select the indix with the maximum probability
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_submission1.csv",index=False)