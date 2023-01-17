import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam,RMSprop
train_data = pd.read_csv("../input/digit-recognizer/train.csv")  #read the training data
train_data.head()       #see the data
train_data.info()           #see how many entries and type of data
data_y = train_data['label']                              # get dependent variable 
train_data.drop(['label'],axis=1,inplace=True)             # drop it from original dataframe
data_x = train_data                                        # store all independent variables separately

data_y.head()
data_x
data_x = data_x.values.reshape(-1,28,28,1)                     # reshape to proper shape
data_y = data_y.values

from keras.utils.np_utils import to_categorical               # turn individual numbers into categorical data
data_y = to_categorical(data_y)                               # Ex : [2] -> [0,0,1,0,0,0,0,0,0,0]
data_x = data_x / 255.0                        # since values are between 0 and 255, divide by 255 to make them between 0 to 1, easier for processing
model = tf.keras.models.Sequential([
        
        tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),       # adding convolution layer with input size (28,28,1) , 1 means the images are in greyscale not rgb
        tf.keras.layers.MaxPooling2D(2,2),                                              # adding pooling layer
    
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
    
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),                                              # total 3 layers added
    
        tf.keras.layers.Flatten(),                                                      # flatten will flatten the input (28,28,1) to a single array
        tf.keras.layers.Dense(128,activation='relu'),                                   # hidden layer with 128 units
        tf.keras.layers.Dense(10,activation='softmax')                                  # output layer with 10 units, each representing the corresponding output
])
model.summary()
model.compile(RMSprop(lr=0.001),loss="categorical_crossentropy",metrics=['accuracy'])          
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x,data_y,test_size=0.2,random_state=0)         # splitting data into train and test for validation later on
datagen = ImageDataGenerator()

train_datagen = datagen.flow(train_x,train_y,batch_size=50)                        # flow training data in batches of 50 to the model
#validation_datagen = datagen.flow(test_x,test_y,batch_size=50)                     # flow validation data in batches of 50
model.fit(train_datagen,
          #validation_data = validation_datagen,
          steps_per_epoch=500,
          epochs=40
         )                                                                       # fit the model with training and validation generators
test = pd.read_csv("../input/digit-recognizer/test.csv")
test = test.values.reshape(-1,28,28,1)
test = test/255.0
preds = model.predict(test)                    # use model to predict for all test values
preds
submission = np.argmax(preds,axis=1)                               # since results are in categorical form, choose the highest in each and store them
submission
test.shape
my_submission = pd.DataFrame({'ImageId': range(1,len(test)+1) ,'Label':submission })               # make dataframe with the column headders and predicted values

my_submission.to_csv("results.csv",index=False)
my_submission.head()
my_submission
