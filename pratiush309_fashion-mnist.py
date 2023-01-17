import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils import to_categorical
import matplotlib.pyplot as plt
%matplotlib inline
from keras.datasets import fashion_mnist
(train_x,train_y),(test_x,test_y)= fashion_mnist.load_data()
print("The Fashion MNIST train dataset has {0} rows and {1} coluns".format(train_x.shape[0],train_x.shape[1]))
print('Train shape :' ,train_x.shape,train_y.shape)
print('Test shape: ',test_x.shape,test_y.shape)
plt.figure(figsize=[5,5])
plt.subplot(121)
plt.imshow(train_x[0,:,:])
plt.title('Ground truth :{0}'.format(train_y[0]))
plt.subplot(122)
plt.imshow(train_x[11,:,:])
plt.title('Ground Truth : {0}'.format(train_y[11]))
print(train_x[0,:,:])
train_x = train_x.reshape(-1,28,28,1)
test_x= test_x.reshape(-1,28,28,1)
print("Train shape ",train_x.shape)
print("Test shape ",test_x.shape)
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_x = train_x / 255.
test_x = test_x / 255.
# Change the labels from categorical to one-hot encoding
train_y_one_hot = to_categorical(train_y)
test_y_one_hot = to_categorical(test_y)

# Display the change for category label using one-hot encoding
print('Original label:', train_y[0])
print('After conversion to one-hot:', train_y_one_hot[0])
from sklearn.model_selection import train_test_split
t_x,val_x,t_y,val_y= train_test_split(train_x,train_y_one_hot,test_size=0.2,random_state=13)
t_x.shape,val_x.shape,t_y.shape,val_y.shape
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
batch_size = 64
epochs = 20
num_classes = 10
model= Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='linear',input_shape=(28,28,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(64,kernel_size=(3,3),activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Conv2D(128,kernel_size=(3,3),activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Flatten())
model.add(Dense(128,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()
model_train = model.fit(t_x,t_y,batch_size=batch_size,epochs=epochs,validation_data=(val_x,val_y),verbose=1)
t= test_x[0]
t= np.expand_dims(t,axis=0)
t.shape
np.argmax(np.round(model.predict(t)))
# To predict on single image 
# https://stackoverflow.com/questions/43017017/keras-model-predict-for-a-single-image
np.argmax(np.round(model.predict(np.expand_dims(test_x[11],axis=0))))
