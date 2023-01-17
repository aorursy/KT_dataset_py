import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense , Dropout ,Flatten, Conv2D , MaxPool2D

from keras.utils import to_categorical

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_dataset=pd.read_csv("../input/digit-recognizer/train.csv")

train_dataset.head()
x=train_dataset.drop(["label"],axis=1).values.reshape(42000,28,28,1)

y=train_dataset.label.values.reshape(-1,1)
print("x shape :",x.shape)

print("y shape :",y.shape)
img=train_dataset.iloc[0,1:].values

img_size=28

plt.imshow(img.reshape(img_size,img_size))

plt.axis("off")

plt.title(train_dataset.iloc[0,0])

plt.show()
sns.countplot(train_dataset.label)

plt.xlabel("numbers")

plt.ylabel("count")

plt.show()
y=to_categorical(y,num_classes=10)

print("new y shape :",y.shape)
x=x/255.0
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.1,random_state=2)

print("x_train shape :",x_train.shape)

print("x_val shape :",x_val.shape)

print("y_train shape :",y_train.shape)

print("y_val shape :",y_val.shape)
model = Sequential()



model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
datagen = ImageDataGenerator(

        rotation_range=0.01,  

        zoom_range = 0.01, 

        width_shift_range=0.5,  

        height_shift_range=0.5, 

        ) 



datagen.fit(x_train)
epochs=100

batch_size=250
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_val,y_val), steps_per_epoch=x_train.shape[0] // batch_size)
print(history.history.keys())
plt.figure(figsize=(10,10))

plt.plot(history.history["loss"],color="green",label="Train Loss")

plt.plot(history.history["val_loss"],color="blue",label="Val Loss")

plt.legend()

plt.title("Train and Validation Loss Plot")

plt.show()
plt.figure(figsize=(10,10))

plt.plot(history.history["accuracy"],color="red",label="Train Accuracy")

plt.plot(history.history["val_accuracy"],color="cyan",label="Val Accuracy")

plt.legend()

plt.title("Train and Validation Accuracy Plot")

plt.show()
y_prediction=model.predict(x_val)
y_prediction_int=np.argmax(y_prediction,axis=1)

y_true=np.argmax(y_val,axis=1)
print("y prediction int shape :",y_prediction_int.shape)

print("y true shape :",y_true.shape)
y_prediction_int
cfm=confusion_matrix(y_true,y_prediction_int)
f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(cfm,annot=True,cmap="coolwarm",linewidths=1,linecolor="black",fmt=".1f",ax=ax)

plt.title("Evaluation Error Plot")

plt.show()
prediction=model.predict(x_val[1].reshape(1,28,28,1))
prediction=np.argmax(prediction,axis=1)

prediction
y_val[1]