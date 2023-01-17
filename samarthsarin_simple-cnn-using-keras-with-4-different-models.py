import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Convolution2D,Dense,Flatten,MaxPooling2D,BatchNormalization,Dropout

from keras.utils import to_categorical

from keras.utils.vis_utils import plot_model

%matplotlib inline
df = pd.read_csv('../input/train.csv')
df.head()
x = df.drop('label',1)

y = df['label']

y = to_categorical(y)
x = x/255.0
x = x.values.reshape(42000,28,28,1)
x.shape
plt.imshow(x[3].reshape(28,28),cmap = 'gray')
model1 = Sequential()

model1.add(Convolution2D(32,(5,5),activation='relu',input_shape = (28,28,1),padding='same'))

model1.add(MaxPooling2D(2,2))

model1.add(Convolution2D(32,(5,5),activation='relu',padding='same'))

model1.add(MaxPooling2D(2,2))

model1.add(Convolution2D(64,(3,3),activation='relu',padding = 'same'))

model1.add(MaxPooling2D(2,2))

model1.add(Convolution2D(64,(3,3),activation = 'relu',padding = 'same'))

model1.add(Flatten())

model1.add(Dense(256,activation='relu'))

model1.add(Dropout(0.2))

model1.add(Dense(10,activation = 'softmax'))
model1.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
model1.summary()
history = model1.fit(x,y,batch_size=32,epochs=10,validation_split=0.1)
values = history.history

validation_acc = values['val_acc']

training_acc = values['acc']

validation_loss = values['val_loss']

training_loss = values['loss']

epochs = range(10)
plt.plot(epochs,validation_acc,label = 'Validation Accuracy')

plt.plot(epochs,training_acc,label = 'Training Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
plt.plot(epochs,validation_loss,label = 'Validation Loss')

plt.plot(epochs,training_loss,label = 'Training Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
model2 = Sequential()

model2.add(Convolution2D(32,(3,3),activation='relu',input_shape = (28,28,1)))

model2.add(Convolution2D(32,(3,3),activation='relu'))

model2.add(Convolution2D(64,(3,3),activation='relu'))

model2.add(Convolution2D(64,(3,3),activation = 'relu'))

model2.add(Flatten())

model2.add(Dense(256,activation='relu'))

model2.add(Dropout(0.2))

model2.add(Dense(10,activation = 'softmax'))
model2.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
model2.summary()
history = model2.fit(x,y,batch_size=32,epochs=10,validation_split=0.1)
values = history.history

validation_acc = values['val_acc']

training_acc = values['acc']

validation_loss = values['val_loss']

training_loss = values['loss']

epochs = range(10)
plt.plot(epochs,validation_acc,label = 'Validation Accuracy')

plt.plot(epochs,training_acc,label = 'Training Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
plt.plot(epochs,validation_loss,label = 'Validation Loss')

plt.plot(epochs,training_loss,label = 'Training Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
model3 = Sequential()

model3.add(Convolution2D(32,(3,3),activation='relu',input_shape = (28,28,1)))

model3.add(MaxPooling2D(2,2))

model3.add(Convolution2D(64,(3,3),activation='relu'))

model3.add(Flatten())

model3.add(Dense(128,activation='relu'))

model3.add(Dropout(0.2))

model3.add(Dense(10,activation = 'softmax'))
model3.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
model3.summary()
history = model3.fit(x,y,batch_size=32,epochs=10,validation_split=0.1)
values = history.history

validation_acc = values['val_acc']

training_acc = values['acc']

validation_loss = values['val_loss']

training_loss = values['loss']

epochs = range(10)
plt.plot(epochs,validation_acc,label = 'Validation Accuracy')

plt.plot(epochs,training_acc,label = 'Training Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
plt.plot(epochs,validation_loss,label = 'Validation Loss')

plt.plot(epochs,training_loss,label = 'Training Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
model4 = Sequential()

model4.add(Convolution2D(32,(5,5),activation='tanh',input_shape = (28,28,1),padding='same'))

model4.add(MaxPooling2D(2,2))

model4.add(Convolution2D(32,(5,5),activation='tanh',padding='same'))

model4.add(MaxPooling2D(2,2))

model4.add(Convolution2D(64,(3,3),activation='tanh',padding = 'same'))

model4.add(MaxPooling2D(2,2))

model4.add(Convolution2D(64,(3,3),activation = 'tanh',padding = 'same'))

model4.add(Flatten())

model4.add(Dense(256,activation='relu'))

model4.add(Dropout(0.2))

model4.add(Dense(10,activation = 'softmax'))
model4.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
model4.summary()
history = model4.fit(x,y,batch_size=32,epochs=10,validation_split=0.1)
values = history.history

validation_acc = values['val_acc']

training_acc = values['acc']

validation_loss = values['val_loss']

training_loss = values['loss']

epochs = range(10)
plt.plot(epochs,validation_acc,label = 'Validation Accuracy')

plt.plot(epochs,training_acc,label = 'Training Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
plt.plot(epochs,validation_loss,label = 'Validation Loss')

plt.plot(epochs,training_loss,label = 'Training Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
plot_model(model1,to_file='model1.png',show_layer_names=True,show_shapes=True)

plot_model(model2,to_file='model2.png',show_layer_names=True,show_shapes=True)

plot_model(model3,to_file='model3.png',show_layer_names=True,show_shapes=True)

plot_model(model4,to_file='model4.png',show_layer_names=True,show_shapes=True)
test = pd.read_csv('../input/test.csv')
test.head()
sub = pd.read_csv('../input/sample_submission.csv')
sub.head()
test = test/255.0
test.shape
test = test.values.reshape(28000,28,28,1)
predict_model1 = model1.predict_classes(test)
predict_model2 = model2.predict_classes(test)
predict_model3 = model3.predict_classes(test)
predict_model4 = model4.predict_classes(test)
final_prediction = 0.25*predict_model1+0.25*predict_model2+0.25*predict_model3+0.25*predict_model4
final_prediction = np.round(final_prediction)

answer = []

for i in range(len(final_prediction)):

    answer.append(int(final_prediction[i]))
predict = pd.Series(answer,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predict],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)
predict = pd.Series(predict_model1,name="Label")

submission_1 = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predict],axis = 1)

submission_1.to_csv("submission1.csv",index=False)
predict = pd.Series(predict_model2,name="Label")

submission_2 = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predict],axis = 1)

submission_2.to_csv("submission2.csv",index=False)
predict = pd.Series(predict_model3,name="Label")

submission_3 = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predict],axis = 1)

submission_3.to_csv("submission3.csv",index=False)
predict = pd.Series(predict_model4,name="Label")

submission_4 = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predict],axis = 1)

submission_4.to_csv("submission4.csv",index=False)
submission['Label'] = int(submission['Label'])