import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# data visualization

import matplotlib.pyplot as plt 

import seaborn as sns



# Model Selection

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



# Model Libraries

import keras 

from keras.models import Sequential

from keras.layers import Dense, Dropout,Flatten, Conv2D, MaxPooling2D



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test =  pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
img_size = 28



train_piksel = np.array(train.drop("label",axis=1))

test_piksel = np.array(test)

test_piksel = test_piksel.reshape(test.shape[0],img_size,img_size,1)



plt.subplot(1,2,1)

plt.imshow(train_piksel[1].reshape(img_size, img_size))

plt.axis('off')

plt.subplot(1, 2, 2)

plt.imshow(train_piksel[15].reshape(img_size, img_size))

plt.axis('off')

plt.show()
num_classes = 10



X_train, X_test, Y_train, Y_test = train_test_split(train_piksel,train.loc[:,"label"],test_size=0.3)



X_train = X_train.reshape(X_train.shape[0],img_size,img_size,1)

X_test = X_test.reshape(X_test.shape[0],img_size,img_size,1)



Y_train = keras.utils.to_categorical(Y_train, num_classes)

Y_test = keras.utils.to_categorical(Y_test, num_classes)



print("X_train.shape :",X_train.shape)

print("X_test.shape :",X_test.shape)

print("Y_train.shape :",Y_train.shape)

print("Y_test.shape :",Y_test.shape)
model = Sequential()

model.add(Conv2D(6, kernel_size=(5,5),activation="relu",input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(16, kernel_size=(5,5),activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(256,activation = "relu"))

model.add(Dense(128,activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

model.summary()
epochs = 20

batch_size = 86



model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size)
loss, acc = model.evaluate(X_test,Y_test)

print("loss :",loss)

print("accuracy :",acc)
Y_pred = model.predict(X_test)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(Y_test,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 



# plot the confusion matrix

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
#set ids as ImageId and predict label 

sId = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

ids = sId.drop("Label",axis=1)

predict = model.predict(test_piksel)

predict = np.argmax(predict,axis = 1) 



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'ImageId' : ids.ImageId, 'Label': predict})

output.to_csv('submission.csv', index=False)