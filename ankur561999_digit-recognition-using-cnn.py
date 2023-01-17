# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import keras

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

from keras.optimizers import SGD
train_path = "/kaggle/input/digit-recognizer/train.csv"

test_path = "/kaggle/input/digit-recognizer/test.csv"

sample_path = "/kaggle/input/digit-recognizer/sample_submission.csv"



train_df = pd.read_csv(train_path)

test_df = pd.read_csv(test_path) 

sample = pd.read_csv(sample_path) 



# view first 5 rows of the dataset

train_df.head()
# view first 5 rows of sample_submission

sample.head() 
# view first 5 rows of test dataset

test_df.head() 
n_height = 28 # height of the image

n_width = 28  # width of the image

n_depth = 1   # depth of the image
# define the input variable X for training set

X = train_df.drop(["label"], axis=1)



# reshape the input variable X to (m, 28,28,1)

X = np.array(X).reshape(-1, n_height,n_width,n_depth)



# define the target variable Y for training set

Y = train_df['label']



# convert the output variable Y to one hot vector

n_classes = np.max(Y) + 1



Y = np.eye(n_classes)[Y]



# check shape of X and Y

print("X.shape: ",X.shape)

print("Y.shape: ",Y.shape)
# define X_test

X_test = test_df



# reshape the X_test

X_test = np.array(X_test).reshape(-1, n_height, n_width, n_depth) 

X_test.shape
# define the training and validation set



# training set

X_train = X[0:37000]

Y_train = Y[0:37000]



# validation set

X_val = X[37000:]

Y_val = Y[37000:]



# print shape of training and valudation set

print("X_train.shape: {0}    Y_train.shape: {1}".format(X_train.shape, Y_train.shape))

print("X_val.shape:   {0}    Y_val.shape:   {1}".format(X_val.shape, Y_val.shape))
# define number of filters 

n_filters = [32, 64]



# define the hyper-parameters

learning_rate = 0.001

n_epochs = 5

batch_size = 64
model = Sequential()



# first convolutional layer with 32 filters of size (3,3)

model.add(Conv2D(filters = n_filters[0], input_shape = (n_width, n_height, n_depth), kernel_size=3,

                padding= 'SAME', activation='relu'))



assert(model.output_shape == (None, 28,28,32))



# first pooling layer with region of size 2x2 and stride 2

model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))



assert(model.output_shape == (None, 14,14,32))



# second convolutional layer with 64 filters of size (3,3)

model.add(Conv2D(filters = n_filters[1], input_shape = (n_width, n_height, n_depth), kernel_size=3,

                padding= 'SAME', activation='relu'))



assert(model.output_shape == (None, 14,14,64))



# second pooling layer with region of size 2x2 and stride 2

model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))



assert(model.output_shape == (None, 7,7,64))



# flatten the layer 

model.add(Flatten())



assert(model.output_shape == (None, 7*7*64))



# fully-connected layer with 1024 neurons

model.add(Dense(units = 1024, activation='relu'))



assert(model.output_shape == (None, 1024))



# another fully-connected layer with 512 neurons

model.add(Dense(units = 512, activation='relu'))



assert(model.output_shape == (None, 512))



# finally the output layer with softmax activation

model.add(Dense(units = 10, activation='softmax'))



# print model summary

model.summary()
# compile the model

model.compile(loss='categorical_crossentropy', 

             optimizer = SGD(lr = learning_rate),

             metrics = ['accuracy'])



# train the model 

model.fit(X_train, Y_train, batch_size = 1,

         epochs = 5)



# evaluate the model

score = model.evaluate(X_val, Y_val)



print("validation loss: ",score[0])

print("validation accuracy: ",score[1])
# save the model

model.save("/kaggle/working/digit_recognizer_model")
# load the model

model = keras.models.load_model("/kaggle/working/digit_recognizer_model")
# make predictions

y_pred = model.predict(X_test) 
preds = keras.backend.argmax(y_pred)

preds = np.array(preds)
imageid = np.arange(1,28001) 

print("preds.shape: ", preds.shape) 

print("imageid.shape: ", imageid.shape) 
arr = np.vstack((imageid, preds)). T 

arr.shape
# create submission dataframe

submission = pd.DataFrame(data=arr, columns=["ImageId", "Label"]) 

submission.shape
submission.tail() 
submission.to_csv("/kaggle/working/digit_recognizer_submission.csv")
from IPython.display import FileLink

FileLink("/kaggle/working/digit_recognizer_submission.csv") 