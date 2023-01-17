import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
training_set = pd.read_csv('../input/train.csv')
print(training_set.shape)
from collections import Counter
Unique = Counter(training_set['label'])
sns.countplot(training_set['label'])
test_set = pd.read_csv('../input/test.csv')
print(test_set.shape)

training_set.head()
a_train = training_set.iloc[:, 1:].values
b_train = training_set.iloc[:, 0].values
a_test = test_set.values
plt.figure(figsize = (10, 8))
a, b = 9, 3
for i in range(27):
    plt.subplot(b, a, i+1)
    plt.imshow(a_train[i].reshape((28, 28)))
plt.show()
a_train = a_train/255.0
a_test = a_test/255.0
b_train
A_train = a_train.reshape(a_train.shape[0], 28, 28, 1)
A_test = a_test.reshape(a_test.shape[0], 28, 28, 1)
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
batch_size = 32
epochs = 100
num_classes = 10
input_shape = (28, 28, 1)
from sklearn.model_selection import train_test_split
b_train = keras.utils.to_categorical(b_train, num_classes)
A_train, A_val, B_train, B_val = train_test_split(A_train, b_train, test_size = 0.2)
Digit = Sequential()

Digit.add(Conv2D(32, (3, 3), input_shape = input_shape, activation = 'relu', kernel_initializer = 'uniform'))
Digit.add(MaxPooling2D(pool_size = (2, 2)))
Digit.add(Dropout(rate = 0.1))

Digit.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'uniform'))
Digit.add(MaxPooling2D(pool_size = (2, 2)))
Digit.add(Dropout(rate = 0.1))

Digit.add(Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'uniform'))
Digit.add(MaxPooling2D(pool_size = (2, 2)))
Digit.add(Dropout(rate = 0.1))

Digit.add(Flatten())
Digit.add(Dense(units = 128, activation = 'relu'))
Digit.add(BatchNormalization())
Digit.add(Dropout(rate = 0.1))
Digit.add(Dense(num_classes, activation = 'sigmoid'))
Digit.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor ='val_loss', 
                                            patience = 10, 
                                            verbose= 0, 
                                            factor = 0.1, 
                                            min_lr = 0.0001)
IDG = ImageDataGenerator(rotation_range=15, #  rotate images in the range (degrees, 0 to 180)
                         zoom_range = 0.1, #  zoom image 
                         width_shift_range=0.1,  #  shift images horizontally (fraction of total width)
                         height_shift_range=0.1)
Digit.summary()
IDG.fit(A_train)
h = Digit.fit_generator(IDG.flow(A_train, B_train, batch_size = batch_size),
                                 epochs = epochs, validation_data = (A_val, B_val),
                                                                     steps_per_epoch = a_train.shape[0] // batch_size,
                                                                     callbacks = [learning_rate_reduction],)

loss, accuracy = Digit.evaluate(A_val, B_val)
print('Final loss : {0:.6f},  final accuracy : {1:.6f}'.format(loss, accuracy))
result = Digit.predict_classes(A_test)
my_submission = pd.DataFrame(data = {'ImageId' : range(1, result.shape[0] + 1), 'Label' : result})
my_submission.to_csv('my_submission.csv', index = None)