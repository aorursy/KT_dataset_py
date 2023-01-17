import tensorflow as tf
import keras
from keras.callbacks import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
traindata = pd.read_csv('./../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')
traindata.head(4)
traindata.shape
f = plt.figure(figsize=(20,6))
ax = f.add_subplot(161)
ax2 = f.add_subplot(162)
ax3 = f.add_subplot(163)
ax4 = f.add_subplot(164)
ax5 = f.add_subplot(165)
ax6 = f.add_subplot(166)
ax.imshow(traindata.iloc[0].values[1:].reshape(28,28))
ax2.imshow(traindata.iloc[5].values[1:].reshape(28,28))
ax3.imshow(traindata.iloc[20].values[1:].reshape(28,28))
ax4.imshow(traindata.iloc[456].values[1:].reshape(28,28))
ax5.imshow(traindata.iloc[999].values[1:].reshape(28,28))
ax6.imshow(traindata.iloc[1500].values[1:].reshape(28,28))
plt.show()
trainlabel=traindata['label'].values
traindata.drop('label',inplace=True,axis=1)
trainimages = traindata.values
#reshape it to (28,28,1)-> (height,width,channels)
trainimages=trainimages.reshape(-1,28,28,1)
testdata = pd.read_csv('./../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')
testlabel=testdata['label'].values
testdata.drop('label',inplace=True,axis=1)
testimages = testdata.values
testimages=testimages.reshape(-1,28,28,1)
from keras.preprocessing.image import ImageDataGenerator
traingen=ImageDataGenerator(rotation_range=20,
                            zoom_range=0.1,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            shear_range=0.1,
                            horizontal_flip=True,
                            rescale=1/255.0,#normalising the data
                            validation_split=0.2 #train_val split
                            )
traindata_generator = traingen.flow(trainimages,trainlabel,subset='training')
validationdata_generator = traingen.flow(trainimages,trainlabel,subset='validation')
testgen=ImageDataGenerator(rescale=1/255.0)
testdata_generator = testgen.flow(testimages,testlabel)
model=Sequential([])

model.add(Conv2D(64,(3,3),activation="relu",input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation="relu",input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256,activation="relu"))
model.add(BatchNormalization())
model.add(Dense(26,activation="softmax"))


model.summary()
model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
# Define a Callback class that stops training once accuracy reaches 99.5%
class myCallback(Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy')>0.995):
      print("\nReached 99.5% accuracy so cancelling training!")
      self.model.stop_training = True
callback=myCallback()
dynamicrate = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
history=model.fit(traindata_generator,epochs=50,validation_data=validationdata_generator,callbacks=[callback,dynamicrate])
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
loss,accuracy = model.evaluate_generator(testdata_generator)
print("test accuracy: "+ str(accuracy*100))
