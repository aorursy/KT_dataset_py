import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt 

%matplotlib inline

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
train= pd.read_csv("../input/digit-recognizer/train.csv")

test= pd.read_csv("../input/digit-recognizer/test.csv")

print("number of training images: %d"%train.shape[0])

print ("number of pixels : %d"%train.shape[1])

train.head()
train.isnull().any().describe()

#thers is not any null values!!
test.isnull().any().describe()

#thers is not any null values!!
x_train=train.drop(['label'],axis=1)

y_train= train['label']
img_train_arr = x_train.to_numpy(dtype=np.int32)
x_train_show = img_train_arr[::,::].reshape(-1,28,28)

fig = plt.figure(figsize=(10,10))

for i in range (25):

    fig.add_subplot(5 , 5 ,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(x_train_show[i] ,cmap=plt.cm.binary)

    plt.title(y_train[i])
x_train= img_train_arr[::,::].reshape(-1,28,28,1)
x_train=x_train/255
y_train= to_categorical(y_train)
X_train,X_val,Y_train ,Y_val = train_test_split(x_train,y_train, test_size=0.2 ,random_state=42)
X_train.shape
model=Sequential()

model.add(Conv2D(32,kernel_size=(3,3),input_shape=(28,28,1), activation='relu',padding='same'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=(3,3), activation= 'relu',padding='same'))

model.add(Dropout(0.2))





model.add(Conv2D(64,kernel_size=(3,3), activation='relu', padding='same'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3), activation='relu', padding='same'))

model.add(Dropout(0.2))





model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(256,activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(64,activation ='relu'))

model.add(Dropout(0.2))

model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=15, validation_data=(X_val, Y_val))
print(history.history.keys())

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.show()
img_test_arr = test.to_numpy(dtype=np.int32)
test=img_test_arr[::,::].reshape(-1,28,28,1)
test=test/255
pred=model.predict(test)

pred = np.argmax(pred,axis = 1)
ids=range(1,len(pred)+1)

submission=pd.DataFrame(columns=['ImageId','Label'])

submission['ImageId']=ids

submission['Label']=pred

submission.to_csv("submission_result.csv", index=False)