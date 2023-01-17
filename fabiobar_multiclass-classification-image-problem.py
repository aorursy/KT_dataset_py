from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
(train_data,train_labels),(test_data,test_labels)=mnist.load_data()


plt.imshow(train_data[0],cmap=plt.cm.binary)
plt.show()
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)
train_data=train_data.reshape(60000,28*28)
train_data = train_data.astype('float32') / 255

test_data=test_data.reshape(10000,28*28)
test_data = test_data.astype('float32') / 255
partial_x_train=train_data[:50000]
x_val=train_data[50000:]

partial_y_train=train_labels[:50000]
y_val=train_labels[50000:]
model=models.Sequential()
model.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history=model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_val,y_val))
results=model.evaluate(test_data,test_labels)
print(results)
