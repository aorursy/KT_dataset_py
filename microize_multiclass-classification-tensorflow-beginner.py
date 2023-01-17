import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import Adam 

from tensorflow.keras.datasets.mnist import load_data



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
(X_train, y_train) , (X_test, y_test)=load_data()

print("Total images in Train Dataset :",len(X_train))

print("Total images in Test Dataset :",len(X_test))
X_train[0]
plt.matshow(X_train[0])
y_train[0]
num_rows, num_cols = 2, 5

f,ax=plt.subplots(num_rows, num_cols, figsize=(12,5),

                     gridspec_kw={'wspace':0.03, 'hspace':0.01}, 

                     squeeze=True)



for r in range(num_rows):

    for c in range(num_cols):

      

        image_index = r * 5 + c

        ax[r,c].axis("off")

        ax[r,c].imshow( X_train[image_index], cmap='gray')

        ax[r,c].set_title('No. %d' % y_train[image_index])

plt.show()

plt.close()
X_train = X_train / 255

X_test = X_test / 255
X_train.shape
X_train_flattened = X_train.reshape(len(X_train), 28*28)

X_test_flattened = X_test.reshape(len(X_test), 28*28)
X_train_flattened.shape
model=Sequential()

model.add(Dense(10,activation='sigmoid',input_shape=(784,)))

model.summary()
model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train_flattened,y_train,epochs=5)

model.evaluate(X_test_flattened,y_test)
y_predicted = model.predict(X_test_flattened)

y_predicted[0]
np.argmax(y_predicted[0])
plt.matshow(X_test[0])
model=Sequential()

model.add(Dense(100,activation='relu',input_shape=(784,)))

model.add(Dense(100,activation='relu'))

model.add(Dense(10,activation='sigmoid'))



model.summary()
model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train_flattened, y_train, batch_size= 128,epochs=5)

model.evaluate(X_test_flattened,y_test)
model.save('keras_mnist.h5')