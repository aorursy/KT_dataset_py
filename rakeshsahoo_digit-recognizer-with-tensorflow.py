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
#importimg packages.

import numpy as np

import matplotlib.pyplot as plt #Graph

from keras.models import Sequential #ANN Architecture

from keras.layers import Dense #The layers in ANN

from keras.utils import to_categorical
#Lets have a look at the Train and Test Data

data = pd.read_csv("../input/digit-recognizer/train.csv")

test_data = pd.read_csv("../input/digit-recognizer/test.csv")

sample_submission=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

data.head()
data.columns
y_train= data.iloc[:,0:1]

x_train= data.iloc[:,1:]

x_test = test_data

X_train = x_train.to_numpy()

Y_train = y_train.to_numpy()

X_test = x_test.to_numpy()

print(X_train.shape,Y_train.shape,X_test.shape)
# Normalize the Images,Normal the pixel values from [0,255] tp

# [-0.5,0.5] to make our network easier to train

X_train = (X_train/255)-0.5

X_test = (X_test/255)-0.5

# Flatten the image, flatten each 28*28 images to a 28*28=784 dimentional vector to pass in to the neaural network

X_train = X_train.reshape((-1,784))

tX_test = X_test.reshape((-1,784))

# Print the shape

print(X_train.shape)

print(X_test.shape)
# Build the model

# Total 3 layers,2 layers with 64 neurons and the relu function

# 1 layer with 10 neurons and softmax function

model=Sequential()

model.add( Dense(64,activation= "relu", input_dim=784))

model.add( Dense(64,activation= "relu"))

model.add( Dense(10,activation= "softmax"))
# Compile the model

# The loss function measures how well the model did on training , and then trais to improve on it using the optimizer

model.compile(

 optimizer = "adam",

 loss = "categorical_crossentropy",

 metrics = ["accuracy"]

)
# Train the model

model.fit(

 X_train,

 to_categorical(Y_train),

 epochs= 50, # the number of itteration over the entair dataset to train on

 batch_size = 32 # the number of sample per gradiant updte for training. 

)
predictions = model.predict_classes(test_data)

#print our model prediction

#print(np.argmax(predictions, axis = 0))

#print(predictions.shape)
sample_submission.head()

submission=pd.DataFrame({'ImageId': sample_submission.ImageId,'Label':predictions})

submission.to_csv('/kaggle/working/submission.csv',index=False)

check=pd.read_csv('/kaggle/working/submission.csv')

check.head()
print('Prediction: ', predictions[0:10])
for i in range (0,10):

    first_image=X_test[i]

    first_image=np.array(first_image,dtype= "float")

    pixels= first_image.reshape((28,28))

    plt.imshow(pixels)

    plt.show()