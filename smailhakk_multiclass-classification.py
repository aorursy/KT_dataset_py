# Importing modules 

import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import cv2



from keras.utils import to_categorical

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout

from keras.models import Sequential

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator





from sklearn.model_selection import train_test_split



np.random.seed(1)
train_path = '../input/fruit-images-for-object-detection/train_zip/train'

train_image = []

train_labels = []



for filenames in os.listdir('../input/fruit-images-for-object-detection/train_zip/train'):

    if(filenames.split('.')[1]=='jpg'):

       img = cv2.imread(os.path.join(train_path,filenames))

       train_labels.append(filenames.split('_')[0])

    

       img = cv2.resize(img, (200, 200))

   

       train_image.append(img)



train_labels = pd.get_dummies(train_labels).values

train_image = np.array(train_image)

x_train, x_val, y_train, y_val = train_test_split(train_image,train_labels,random_state=1)
print(len(train_labels))
print(train_labels)
test_images = []

test_labels = []

test_path = '../input/fruit-images-for-object-detection/test_zip/test'

for filenames in os.listdir('../input/fruit-images-for-object-detection/test_zip/test'):

    if(filenames.split(".")[1]=='jpg'):

        test_labels.append(filenames.split("_")[0])

        img = cv2.imread(os.path.join(test_path,filenames))

        img = cv2.resize(img,(200,200))

        test_images.append(img)



test_images = np.array(test_images)

    
model = Sequential()

model.add(Conv2D(filters=20,kernel_size=(3,3),activation='tanh',input_shape=(200,200,3,)))

model.add(MaxPool2D(2,2))

model.add(Conv2D(filters=30,kernel_size=(3,3),activation='tanh'))

model.add(MaxPool2D(2,2))

model.add(Conv2D(filters=40,kernel_size =(3,3),activation ='tanh'))

model.add(MaxPool2D(2,2))

model.add(Conv2D(filters=50,kernel_size =(3,3),activation ='tanh'))

model.add(MaxPool2D(2,2))



model.add(Flatten())

model.add(Dense(20,activation='tanh'))

model.add(Dropout(0.50))

model.add(Dense(4,activation='softmax'))
optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.999)

model.compile(optimizer = optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

epochs=2

batch_size=200
# Model Summary

model.summary()
#Training the model

history = model.fit(x_train,y_train,epochs=10,batch_size=50,validation_data=(x_val,y_val))



# Visualizing Training data

print(train_labels[0])

plt.imshow(train_image[0])
# Visualizing Training data

# Visualizing Training data

print(train_labels[0])

plt.imshow(train_image[0])

print(train_labels[4])

plt.imshow(train_image[4])
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# Evaluating model on validation data

evaluate = model.evaluate(x_val,y_val)

print(evaluate)
# Testing predictions and the actual label

checkImage = test_images[0:1]

checklabel = test_labels[0:1]



predict = model.predict(np.array(checkImage))



output = { 0:'apple',1:'banana',2:'mixed',3:'orange'}



print("Actual :- ",checklabel)

print("Predicted :- ",output[np.argmax(predict)])