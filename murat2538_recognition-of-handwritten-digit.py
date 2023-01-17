#import Library
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import models
from keras import layers
import numpy as np

import warnings
warnings.filterwarnings("ignore")
#preparing and load data,seperate data
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
#ploting data
digit1=train_images[25]
digit1=digit1.reshape(28,28)
plt.imshow(digit1,plt.cm.binary)
plt.show()
#prepare data for train and test model
train_images=train_images.reshape((60000,28,28,1))
train_images=train_images.astype("float32")/255

test_images=test_images.reshape((10000,28,28,1))
test_images=test_images.astype("float32")/255

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)
#create model
model=models.Sequential()
model.add(layers.Conv2D(32,(5,5),activation="relu",input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2),strides=2))
model.add(layers.Conv2D(32,(5,5),activation="relu"))
model.add(layers.MaxPooling2D((2,2),strides=2))
model.add(layers.Conv2D(64,(4,4),activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(256,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))
#show model
model.summary()
#compile model
model.compile(optimizer="rmsprop",loss="categorical_crossentropy",
              metrics=["accuracy"])
history=model.fit(train_images,train_labels,epochs=8,batch_size=40)
#if you want to save model you run this code
#model.save('my_model_digit.h5')
test_loss,test_acc=model.evaluate(test_images,test_labels)
print("test_acc: ",test_acc)
print("test_loss",test_loss)
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,5))
ax1.plot(history.history['accuracy'])
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'test'], loc='upper left')

ax2.plot(history.history['loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'test'], loc='upper left');

f,ax=plt.subplots(1,5,figsize=(20,10))
index=100
for i in range(0,5):
    digit=test_images[index]
    label=np.argmax(test_labels[index])
    score=model.predict_proba(digit.reshape(1,28,28,1)).max()
    predict=model.predict_classes(digit.reshape(1,28,28,1))
    ax[i].imshow(digit.reshape(28,28),plt.cm.binary)
    ax[i].set_title("label:{}\npredict:{},score:{:.6f}".format(label,predict,score))
    index+=1
f,ax=plt.subplots(1,5,figsize=(20,10))
index=5000
for i in range(0,5):
    digit=test_images[index]
    label=np.argmax(test_labels[index])
    score=model.predict_proba(digit.reshape(1,28,28,1)).max()
    predict=model.predict_classes(digit.reshape(1,28,28,1))
    ax[i].imshow(digit.reshape(28,28),plt.cm.binary)
    ax[i].set_title("label:{}\npredict:{},score:{:.6f}".format(label,predict,score))
    index+=1
f,ax=plt.subplots(1,5,figsize=(20,10))
index=2658
for i in range(0,5):
    digit=test_images[index]
    label=np.argmax(test_labels[index])
    score=model.predict_proba(digit.reshape(1,28,28,1)).max()
    predict=model.predict_classes(digit.reshape(1,28,28,1))
    ax[i].imshow(digit.reshape(28,28),plt.cm.binary)
    ax[i].set_title("label:{}\npredict:{},score:{:.6f}".format(label,predict,score))
    index+=5