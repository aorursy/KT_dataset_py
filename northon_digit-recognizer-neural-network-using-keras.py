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
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
import pandas as pd
input_file = ("../input/train.csv")

df_train=pd.read_csv(input_file)
df_train.shape # (42000, 785)
df_train.head()
import pandas as pd
input_file = ("../input/test.csv")

df_test=pd.read_csv(input_file)
df_test.shape # gives (28000, 784)
df_test.head() #gives a dataframe
from sklearn.model_selection import train_test_split # the needed split-function imported from scikit-learn

train_set, test_set = train_test_split(df_train, test_size=0.20, random_state=42)

X_train_set = train_set.drop(['label'], axis=1) #Dropping 'label', the predicted variable 
y_train_set = train_set['label'] # keeping 'label', the predicted variable 

X_test_set = test_set.drop(['label'], axis=1)
y_test_set = test_set['label']
df_train_label_array = y_train_set.as_matrix() #creates a numpy array of the df
df_train_image_array = X_train_set.as_matrix() #creates a numpy array of the df

df_test_image_array = X_test_set.as_matrix()
df_test_label_array = y_test_set.as_matrix()
from keras import backend as K

if K.image_data_format() == 'channels_first':
    train_images = df_train_image_array.reshape(df_train_image_array.shape[0], 1, 28, 28)
    test_images =  df_test_image_array.reshape(df_test_image_array.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    train_images = df_train_image_array.reshape(df_train_image_array.shape[0], 28, 28, 1)
    test_images = df_test_image_array.reshape(df_test_image_array.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255
test_images.shape
train_labels = keras.utils.to_categorical(df_train_label_array, 10)
test_labels = keras.utils.to_categorical(df_test_label_array, 10)
test_labels
import matplotlib.pyplot as plt

def display_sample(num):
    #Print the one-hot array of this sample's label 
    print(train_labels[num])  
    #Print the label converted back to a number
    label = train_labels[num].argmax(axis=0)
    #Reshape the 768 values to a 28x28 image
    image = train_images[num].reshape([28,28])
    plt.title('Sample: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
    
display_sample(1111) #the 1111th image in the Training set
display_sample(2222) #the 2222nd image in the Training set
display_sample(3333) #the 3333rd image in the Training set
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# 64 3x3 kernels
model.add(Conv2D(64, (3, 3), activation='relu'))
# Reduce by taking the max of each 2x2 block
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout to avoid overfitting
model.add(Dropout(0.25))
# Flatten the results to one dimension for passing into our final layer
model.add(Flatten())
# A hidden layer to learn with
model.add(Dense(128, activation='relu'))
# Another dropout
model.add(Dropout(0.5))
# Final categorization from 0-9 with softmax
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(train_images, train_labels,
                    batch_size=32,
                    epochs=10,
                    verbose=2,
                    validation_data=(test_images, test_labels))
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
df_test_RESULTS = df_test.as_matrix()
df_test_RESULTS.shape
testX = df_test_RESULTS.reshape(df_test_RESULTS.shape[0], 28, 28, 1)
testX = testX.astype(float)
testX /= 255.0
testX.shape
predictions = model.predict_classes(testX, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)
