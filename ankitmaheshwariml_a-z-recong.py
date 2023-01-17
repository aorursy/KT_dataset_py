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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

#address = "./A_Z Handwritten Data.csv"

#df = pd.read_csv(address,header=None)

from keras import models,layers
df = pd.read_csv("../input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv",header = None)

df.head(5)
print("Unique value of alphabets is",len(df[0].unique()))

print("Total sample is", df.shape[0] , "one sample having" , df.shape[1] ,"values")
print("Lets suffles the data")

df = df.sample(frac = 1) 

df.head()
train_data = df.iloc[:,1:].values

train_target = df[0].values

print(train_target.shape,train_data.shape)
x_train = train_data[:100000]

x_test = train_data[100000:120000,:]

x_val = train_data[120000:130000,:]

print(x_train.shape,x_test.shape,x_val.shape)



y_train = train_target[:100000]

y_test = train_target[100000:120000]

y_val = train_target[120000:130000]

print(y_train.shape,y_test.shape,y_val.shape)

print("Lets see some image")

plt.figure(figsize = (12,10))

row, colums = 4, 5

for i in range(0,20):

    plt.subplot(colums, row, i+1)

    plt.imshow(x_train[i].reshape(28,28))

plt.show()
def one_hot_code(sequences,dimension = 26):

    result = np.zeros((len(sequences),dimension))

    for i,label in enumerate(sequences):

        result[i,label] =1

    return result

print("Doing one hot encoding for target date")

y_train = one_hot_code(y_train)

y_test = one_hot_code(y_test)

y_val = one_hot_code(y_val)

print(y_train.shape,y_test.shape,y_val.shape)
print("Prining shape and type for ready data ")

print(x_train.shape,x_test.shape,x_val.shape)

print(y_train.shape,y_test.shape,y_val.shape)

print("Data Type")

print(type(x_train),type(y_train))
print("opting 96 layer network with relu and softmax function")

model = models.Sequential()

model.add(layers.Dense(96,activation= 'relu',input_shape = (784,)))

model.add(layers.Dense(96,activation= 'relu'))

model.add(layers.Dense(26,activation='softmax'))

model.compile(optimizer='rmsprop',

             loss='categorical_crossentropy',

             metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=512,epochs=20,validation_data=(x_val,y_val))
val_loss = history.history['val_loss']

val_accuracy = history.history['val_accuracy']

loss = history.history['loss']

accuracy = history.history['accuracy']
plt.clf()

epoch = range(1,len(loss) + 1)

plt.plot(epoch,loss,'bo',label = 'Loss')

plt.plot(epoch,val_loss,label = 'Validation_loss')

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.show()
plt.plot(epoch,accuracy,'bo',label = 'Accuracy')

plt.plot(epoch,val_accuracy,label = "Validation Accuray")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
print("Lets see how model is working")

result = model.evaluate(x_test,y_test)



print("loss for the model is",result[0]," and accuracy is ",result[1])
print("lets evaluate model on rest of the data")

x_final = train_data[130000:]

y_final = train_target[130000:]

y_final = one_hot_code(y_final)

print(x_final.shape,y_final.shape)

result = model.evaluate(x_final,y_final)

print("loss for the model is",result[0] * 100," and accuracy is ",result[1] * 100)
print("I used only 100000 samples for traiing to avoid overfitting and model showing accuracy above  90%")
print("lets try this with convolutional layer")
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation = 'relu',input_shape =(28,28,1) ))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation="relu"))

model.add(layers.Flatten())

model.add(layers.Dense(64,activation = 'relu'))

model.add(layers.Dense(26,activation ='softmax'))

model.summary()
x_train = x_train.reshape(100000,28,28,1)

x_train = x_train.astype('float32') /255

x_val = x_val.reshape(10000,28,28,1)

x_val = x_val.astype('float32') /255

x_test = x_test.reshape(20000,28,28,1)

x_test = x_test.astype('float32') /255
print("Prining shape and type for ready data ")

print(x_train.shape,x_test.shape,x_val.shape)

print(y_train.shape,y_test.shape,y_val.shape)

print("Data Type")

print(type(x_train),type(y_train))
model.compile(optimizer='rmsprop',

             loss = 'categorical_crossentropy',

             metrics = ['accuracy'])
history = model.fit(x_train,y_train,batch_size=512,epochs=10,validation_data=(x_val,y_val))
test_loss,test_acc = model.evaluate(x_test,y_test)
print("Accuracy for the model on test dats is ",test_acc)