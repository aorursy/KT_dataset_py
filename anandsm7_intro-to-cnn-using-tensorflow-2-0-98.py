#Lets import the needed Libraries

#Kaggle comes with the latest tensorflow 2.0 version so i don't have to install it :p

import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

import itertools

tf.__version__
#Now lets load the dataset

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
#Now we have loaded the data now lets see its content

train.head()
#Now we can see that the first column labels is the target and rest are the features

X_train = train.iloc[:,1:]

y_train = train.iloc[:,0]

#Same for the test set

X_test = test.iloc[:,:]

train.shape
#Now lets just visualize the MNIST class count tp check imbalance

g = sns.countplot(y_train)
#We can see that all the classes are around similar range.That's what we needed 

#Now lets the normalize the data to make the pixels within (0 - 1) range

X_train = X_train / 255.0

X_test = X_test / 255.0
#In the above section[24] we could see that dataset shape (42000, 785),

#but since we are using Convolution our model expects an input in the shape (height x width x depth)

#Since we are using a black and white image the depth/colour channel should be 1.

#So lets reshape the inputs

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

X_train.shape
#Now lets encode the labels since that what our loss function expects

y_train = tf.keras.utils.to_categorical(y_train)

y_train.shape
#Now lets split our dataset for train and validation.90% for training and 10% for validation

random_seed = 42

X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.1)

input_shape = (28,28,1)
#Now let us take a look at sample MNIST data

plt.imshow(X_train[0].reshape(28,28),cmap=plt.set_cmap('gray'))
#Building the model

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32,kernel_size = (3,3),activation='relu',input_shape=input_shape))

model.add(tf.keras.layers.Conv2D(64,kernel_size = (3,3),activation='relu'))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128,activation='relu'))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(10,activation='softmax'))
#Lets checkout the model summary

model.summary()
#Looks okay :) 

#Now lets compile the model

model.compile(optimizer='adam',metrics=['acc'],loss='categorical_crossentropy')
epochs = 10

batch_size = 32

history = model.fit(X_train,y_train,

                   epochs=epochs,

                   validation_data=(X_val,y_val),

                   batch_size=batch_size,

                   verbose=1).history
#Now we can plot how our accuracy and loss went

loss = history['loss']

val_loss = history['val_loss']

epochs = range(1, len(loss)+ 1)

print(loss)

print(epochs)

line1 = plt.plot(epochs,loss,label="Validation/Test loss")

line2 = plt.plot(epochs,val_loss,label="Training loss")

plt.setp(line1,linewidth=2.0,marker='+',markersize="10.0")

plt.setp(line2,linewidth=2.0,marker='4',markersize="10.0")

plt.title("Model Loss")

plt.ylabel("Epocs")

plt.xlabel("Loss")

plt.grid(True)

plt.legend()

plt.show()



acc = history['acc']

val_acc = history['val_acc']

line1 = plt.plot(epochs,acc,label="Validation/Test Accuracy")

line2 = plt.plot(epochs,val_acc,label="Training Accuracy")

plt.setp(line1,linewidth=2.0,marker='+',markersize="10.0")

plt.setp(line2,linewidth=2.0,marker='4',markersize="10.0")

plt.title("Model Accuracy")

plt.ylabel("Epocs")

plt.xlabel("Accuracy")

plt.grid(True)

plt.legend()

plt.show()
#Check model score

score = model.evaluate(X_val,y_val)

print(f"loss :{score[0]}")

print(f"Accuracy :{score[1]}")
#lets plot a random image from the test set

rand = np.random.randint(0,len(X_test))

plt.imshow(X_test[rand].reshape(28,28))
# 7 thats my favourite number :)

#Now lets see how the model predict

result = model.predict_classes(X_test[rand].reshape(-1,28,28,1))[0]

print(f"Predicted Result : {result}")