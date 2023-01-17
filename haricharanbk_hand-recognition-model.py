import numpy as np 

import pandas as pd 

from sklearn.model_selection import train_test_split

import cv2 as cv

import matplotlib.pyplot as plt

from keras.utils import to_categorical



from keras.models import Sequential, load_model

from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, AveragePooling2D



import os

# os.listdir('/kaggle/input/processed-data-of-handgestures')
data = pd.read_feather('/kaggle/input/processed-data-of-handgestures/processed_data.feather')

# tot_x, tot_y = data.iloc[:, 1], data.iloc[:, 0]

# print(data.shape)

tot_x, tot_y = np.reshape( np.array(data.iloc[:, 1:]), (data.shape[0], 256, 256, 1)), data.iloc[:, 0]

print(tot_x.shape)

print(tot_y.shape)
x, x_test, y, y_test = train_test_split(tot_x, tot_y, train_size = .9)

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = .8)
print(x_train.shape, y_train.shape)

print(x_val.shape, y_val.shape)

print(x_test.shape, y_test.shape)
del(data, x, y, tot_x, tot_y)
y_train = to_categorical(y_train)

y_val = to_categorical(y_val)

y_test = to_categorical(y_test)
print(y_train[:10])

print(y_val[:10])

print(y_test[:10])
model = Sequential()



# model.add(Conv2D(4, kernel_size=(128, 128), activation='relu', input_shape= (256, 256, 1) ))



# model.add( Flatten() )

# model.add( Dropout(0.5) )

# model.add( Dense(4, activation = 'softmax') )



model.add(Conv2D(32, (5, 5), activation='relu', input_shape = (256, 256, 1)))

model.add(AveragePooling2D(3,3))

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(AveragePooling2D(3,3))

model.add(Conv2D(128, (3,3), strides = (2,2), activation='relu'))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), strides= (1,1), activation = 'relu', padding = 'same'))

model.add(Conv2D(256, (3,3), activation = 'relu'))

model.add(AveragePooling2D(3,3))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(Dropout(0.25))

model.add(Dense(4, activation = 'softmax'))

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5)
# df = pd.DataFrame()

# for i in range(10):

#     df = df.append(pd.Series([*(range(100))]), ignore_index=True)

# print(df)

preds = model.predict(x_test)
# decoding from one-hot encoding

predicted = np.argmax(preds, axis = 1)

actual = np.argmax(y_test, axis = 1)
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 



print('confusion matrix:\n', confusion_matrix(predicted, actual))

print()

print('accuracy: ' , accuracy_score(predicted, actual))

print()

print('classification report:\n', classification_report(predicted, actual))
model.save('HandRecognitionModel_acc_98.h5')
x = load_model('HandRecognitionModel_acc_98.h5')