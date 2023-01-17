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
# importing some useful libs for the idea

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import keras 

from keras.models import Sequential

from keras.layers import Dense,Flatten, Conv2D, MaxPooling2D, Dropout
# Taking training and testing data

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test =  pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
# getting the dimensions of train data set

print(train.shape)

# as image is in 28x28 pixels we have 784 colums of pixel values and 1 column of extra class label in data
# getting the dimensions of test data set

print(test.shape)

# as image is in 28x28 pixels we have 784 colums of pixel values and 28000 samples for the testing purpose
train_df = np.array(train.drop("label",axis=1))

test_df = np.array(test)
# KFold - optional could perform kfold operation for better accuracy

# and training and testing of data

num_classes = 10

img_size = 28

X_train, X_test, Y_train, Y_test = train_test_split(train_df,train.loc[:,"label"],test_size=0.30,random_state=0)



X_train = X_train.reshape(X_train.shape[0],img_size,img_size,1)

X_test = X_test.reshape(X_test.shape[0],img_size,img_size,1)



Y_train = keras.utils.to_categorical(Y_train, num_classes)

Y_test = keras.utils.to_categorical(Y_test, num_classes)



print("X_train.shape :",X_train.shape)

print("X_test.shape :",X_test.shape)

print("Y_train.shape :",Y_train.shape)

print("Y_test.shape :",Y_test.shape)
# much more conv2d layer and maxpooling layer could be added but would may also reduce the accuracy

model = Sequential()

model.add(Conv2D(64, kernel_size=(5,5),activation="relu",input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(16, kernel_size=(4,4),activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(256,activation = "relu"))

model.add(Dense(128,activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

model.summary()
epochs = 50

batch_size = 100 



model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size)
loss, acc = model.evaluate(X_test,Y_test)

print("loss :",loss)

print("accuracy :",acc)
import seaborn as sns

Y_pred = model.predict(X_test)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(Y_test,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 



# plot the confusion matrix

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Reds",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
# output generation step for submission

#set ids as ImageId and predict label

test_df = test_df.reshape(test.shape[0],img_size,img_size,1)

sId = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

ids = sId.drop("Label",axis=1)

predict = model.predict(test_df)

predict = np.argmax(predict,axis = 1) 



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'ImageId' : ids.ImageId, 'Label': predict})

output.to_csv('submission.csv', index=False)