import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import time



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten , Activation, Convolution2D

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
y_train  = train.label

X_train = train.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,test_size=0.2)  # For seperating train and test data.
X_train = X_train.values

y_train = y_train.values

X_test = X_test.values

y_test = y_test.values
print(X_train.shape)

print(X_test.shape)

print(X_train[0].shape) # We need to do a reshape to be able to display the image 
# Since we know that the square root of 784 is 28, we change the shape as follows

#plot the first image in the dataset

plt.imshow(X_train[0].reshape(28,28))
#reshape data to fit model (we should reshape them to make sure that all our data has the same dimention and is suitable for the training phase)

X_train = X_train.reshape(33600,28,28,1)

X_test = X_test.reshape(8400,28,28,1)
#one-hot encode target column

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

y_train[0]
#creating our model

model = Sequential()

#add model layers

model.add(Conv2D(64, kernel_size=3, input_shape=(28,28,1)))

model.add(Activation("relu"))

model.add(Conv2D(32, kernel_size=3))

model.add(Activation("relu"))

model.add(Flatten())

model.add(Dense(10))

model.add(Activation("softmax"))
#compile model using accuracy to measure model performance

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
start_time = time.time()

#train the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

print("Finised in : ", time.time()-start_time, "s")
#predict first 3 images in the test set

model.predict(X_test[:3])
#actual results for first 3 images in test set

y_test[:3]
Data = test.values

Data.shape
Data = Data.reshape(28000,28,28,1)
predicted_results = model.predict(Data)
print(predicted_results[0])

np.argmax(predicted_results[0])  #Getting the maximum values' index from softmax prediction's output

# max(predicted_results[0]).index
prediction_list = []

for i in range(0,len(Data)):

    prediction_list.append(np.argmax(predicted_results[i]))
prediction_list
labels = ['ImageId','Label']

index = list(range(1,len(Data)+1))

toDF = {'ImageId':index, 'Label':prediction_list}
Predicted_df=pd.DataFrame(toDF,index=None)
Predicted_df.head()
sample = pd.read_csv('../input/sample_submission.csv')
sample.head()
Predicted_df.to_csv('submission.csv',index=False)