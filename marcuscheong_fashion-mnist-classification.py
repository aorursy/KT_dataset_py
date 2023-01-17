import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import cv2
train=pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

train.head()
sns.countplot(train['label'])
plt.figure(figsize=(15,15))

for n in range(1,11):

    for data in train.values:

        label=data[0]

        if(label==n-1):

            img=np.array(data[1:]).reshape(28,28)

            plt.subplot(5,2,n)

            plt.title(n-1)

            plt.axis('off')

            plt.imshow(img,cmap='gray')

            break
from sklearn.preprocessing import OneHotEncoder



X=((train.drop(columns=['label'])).values/255).reshape(-1,28,28,1)

y=train['label']



ohe=OneHotEncoder()

y=ohe.fit_transform(y.values.reshape(-1,1))

y=y.todense()
from sklearn.model_selection import train_test_split



xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2)
import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization



model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))

model.add(BatchNormalization())



model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

    

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(10, activation='softmax'))



model.summary()
model.compile(optimizer='adam',

             loss='categorical_crossentropy',

             metrics=['accuracy'])



earlystop=keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)



history=model.fit(xtrain, ytrain, validation_split=0.2, epochs=50,batch_size=100, callbacks=[earlystop])
# Accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Training accuracy vs Validation accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.show()



# Loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Training loss vs Validation loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.show()
score=model.evaluate(xtest,ytest)