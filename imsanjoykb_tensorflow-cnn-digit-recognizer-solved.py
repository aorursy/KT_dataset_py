# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#Training data
train_data = pd.read_csv('../input/digit-recognizer/train.csv') # Import the dataset
train_y = train_data["label"] # Create label vector
train_data.drop(["label"], axis=1, inplace=True) # Remove the label vector from the pixel column matrix
train_X = train_data
train_X = train_X.values.reshape(-1, 28, 28, 1)
train_y = train_y.values
train_y = tf.keras.utils.to_categorical(train_y)
 
train_X = train_X/255.00 # Normalization
#Test data
test_X = pd.read_csv('../input/digit-recognizer/test.csv')
test_X = test_X.values.reshape(-1,28,28,1)
test_X = test_X / 255.0 # Normalization
model = tf.keras.Sequential([
tf.keras.layers.Conv2D(32, kernel_size = (5,5), padding = 'same', activation ='relu', input_shape = (28,28,1)),   
tf.keras.layers.Dropout(0.8),
tf.keras.layers.Conv2D(32, kernel_size = (5,5), padding = 'same', activation ='relu'),
tf.keras.layers.Dropout(0.8),    
tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
tf.keras.layers.Conv2D(32, kernel_size = (5,5), padding = 'same', activation ='relu', input_shape = (28,28,1)),
tf.keras.layers.Conv2D(32, kernel_size = (5,5), padding = 'same', activation ='relu'),
tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(1024, activation = "relu"),
tf.keras.layers.Dense(256, activation = "relu"),
tf.keras.layers.Dense(10, activation = "softmax")
    ])
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40, zoom_range=0.2)# Create Data Augmentation (DA) Iterator
datagen.fit(train_X) #Our DA Iterator is trained with the image data, to calculate internal statistics
# Let's add callbacks, which adjust learning rate
ln_fc = lambda x: 1e-3 * 0.985 ** x
lrng_rt = tf.keras.callbacks.LearningRateScheduler(ln_fc)
# Fit our CNN-model using the DA Iterator using Flow-method, which feeds batches of augmented data:
digitizer = model.fit_generator(datagen.flow(train_X, train_y, batch_size=1024), epochs=80, callbacks=[lrng_rt]) 
predictions = model.predict(test_X)
predictions[358]
pred = np.argmax(predictions, axis=1)

plt.imshow(test_X[358][:,:,0],cmap='gray')
plt.show()

pred[358]

pred_digits = pd.DataFrame({'ImageId': range(1,len(test_X)+1) ,'Label':pred })
pred_digits.to_csv("pre_digits.csv",index=False)
predictions = model.predict(test_X)
predictions[364]
pred = np.argmax(predictions, axis=1)

plt.imshow(test_X[364][:,:,0],cmap='gray')
plt.show()

pred[364]

pred_digits = pd.DataFrame({'ImageId': range(1,len(test_X)+1) ,'Label':pred })
pred_digits.to_csv("pre_digits.csv",index=False)
