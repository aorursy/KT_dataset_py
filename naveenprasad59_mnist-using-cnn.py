import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
(X_train,y_train),(X_test,y_test) = mnist.load_data()
show = np.random.randint(0,60001)
plt.imshow(X_train[show],cmap='gray')
plt.title(f'Label :{y_train[show]}')
plt.show()
print('Shape of Training Data Before :',X_train.shape)
print('Shape of Testing Data Before :',X_test.shape)
X_train = np.expand_dims(X_train,axis=3)
X_test = np.expand_dims(X_test,axis=3)
print('Shape of Training Data After ',X_train.shape)
print('Shape of Testing Data After:',X_test.shape)
print('Max pixel value before Scaling:',np.max(X_train))
X_train = X_train/255.0
X_test  = X_test/255.0
print('Max pixel value after scaling:',np.max(X_train))

y_train[0]
model = tf.keras.models.Sequential([
            Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
            MaxPool2D(2,2),
            Conv2D(64,(3,3),activation='relu'),
            MaxPool2D(2,2),
            Flatten(),
            Dense(256,activation='relu'),
            Dense(128,activation='relu'),
            Dense(64,activation='relu'),
            Dense(10,activation='softmax'),

])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(X_train,y_train,epochs=10,batch_size=32,validation_data=(X_test,y_test))
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
epochs=range(len(train_acc))
plt.plot(epochs,train_acc,'r', label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, train_loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
