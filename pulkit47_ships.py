import numpy as np

import pandas as pd 

import os



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_json('../input/shipsnet.json')

df.head()
type(df['data'][0])
img_rows=80

img_cols=80

img_channels=3
x=[]

for image in df['data']:

    image=np.array(image)

    image=image.reshape((3, 6400)).T.reshape((80,80,3))

    x.append(image)

x=np.array(x)

y=np.array(df['labels'])
image_shape=(80,80,3)
print(x.shape)

print(y.shape)
#creating a list of random indices from the training dataset

from random import sample

plot_num_images=6

num_imgs=x.shape[0]

indices=sample(range(0,num_imgs+1),plot_num_images)

indices
import matplotlib.pyplot as plt

%matplotlib inline 
i=0

for index in indices:

    plt.subplot(2,3,i+1)

    img=x[index]

    plt.imshow(img,cmap="hot")

    class_label=y[index]

    plt.title('Class Label: {}'.format(class_label))

    i+=1

plt.tight_layout()

plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=7)
x_train.shape
x_train=x_train.astype('float32')

x_test=x_test.astype('float32')
x_train/=255.0

x_test/=255.0
num_classes=2
from keras.utils import to_categorical

y_train=to_categorical(y_train)

y_test=to_categorical(y_test)
#Using Keras

from keras.models import Sequential

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers import Dense, Dropout
#create model

model=Sequential()



model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=image_shape))

#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.20))



model.add(Conv2D(64, (3, 3), activation='relu'))

#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.20))



model.add(Conv2D(64, (3, 3), activation='relu'))

#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.20))



model.add(Flatten())

model.add(Dense(64,activation='relu'))

model.add(Dropout(0.20))

model.add(Dense(num_classes, activation='softmax'))



#compile model

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



#fit the model

history= model.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=10,batch_size=32,verbose=1)
model.summary()
# list all data in history

print(history.history.keys())
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='best')

plt.show()
# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='best')

plt.show()
#create model

model=Sequential()



model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=image_shape))

#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.20))



model.add(Conv2D(64, (3, 3), activation='relu'))

#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.20))



model.add(Conv2D(64, (3, 3), activation='relu'))

#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.20))



model.add(Flatten())

model.add(Dense(64,activation='relu'))

model.add(Dropout(0.20))

model.add(Dense(num_classes, activation='softmax'))



#compile model

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])



#fit the model

history= model.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=10,batch_size=32,verbose=1)
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='best')

plt.show()
# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='best')

plt.show()