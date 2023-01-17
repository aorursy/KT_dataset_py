# 2nd attempt at the MNIST challenge. Previously, I tried a WRN, which turned out to be much too complex 

# for the dataset and extremely overfit. This time, a simpler CNN will suffice.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import keras

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/train.csv')

df.head(5)
df.values.shape
x = df.drop(['label'],1)

x.head()
y = df['label']

y.head()
from keras.utils import to_categorical

X = x.values.reshape(x.values.shape[0],28,28,1)

Y = to_categorical(y.values,10)

print(Y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.15)



print(x_train.shape,x_test.shape)
# Define model

from keras import layers, models



network_input = layers.Input((28,28,1))

conv1 = layers.Conv2D(16, kernel_size=(4,4))(network_input)

b1 = layers.BatchNormalization()(conv1)

a1 = layers.Activation('relu')(b1)

d1 = layers.Dropout(0.1)(a1)

conv2 = layers.Conv2D(32, kernel_size=(4,4), strides = 2)(d1)

b2 = layers.BatchNormalization()(conv2)

a2 = layers.Activation('relu')(b2)

d2 = layers.Dropout(0.1)(a2)

conv3 = layers.Conv2D(64, kernel_size=(4,4), strides = 2)(d2)

b3 = layers.BatchNormalization()(conv3)

a3 = layers.Activation('relu')(b3)

drop = layers.Dropout(0.2)(a3)

flat = layers.Flatten()(drop)

output = layers.Dense(10, activation='softmax')(flat)



model = models.Model(network_input, output)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

model.summary()



from keras.callbacks import *

callbacks = [

    EarlyStopping(monitor='val_loss', patience=2, verbose=0),

    ModelCheckpoint('mnist.h5', monitor='val_loss', save_best_only=True, verbose=0),

]

model.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_test,y_test),callbacks=callbacks)
model = models.load_model('mnist.h5')
# Load test values and make predictions

predict_x = pd.read_csv('/kaggle/input/test.csv')

predict_x = predict_x.values.reshape(predict_x.values.shape[0],28,28,1)

print(predict_x.shape)

predictions = np.argmax(model.predict(predict_x),axis=1)

print(predictions.shape)
# Visualize predictions

import random

import matplotlib.pyplot as plt

choice = random.randint(0,predict_x.shape[0])

plt.imshow(predict_x[choice][:,:,0],cmap='gray')

print("Prediction: ",predictions[choice])
# Encode our data into comma separated values for submission

text = ["ImageId,Label\n"]

for index, prediction in enumerate(predictions):

    string = str(index+1)+','+str(prediction)

    if index != 27999: string += "\n" # We do this to make sure there's not an empty row at the end.

    text.append(string)

with open("submission.csv","w+") as writer:

    writer.writelines(text)