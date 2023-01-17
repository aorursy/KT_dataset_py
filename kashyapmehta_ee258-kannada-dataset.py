import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow as tf # Import tensorflow library

from tensorflow import keras # Import Keras Library
train=pd.read_csv("../input/Kannada-MNIST/train.csv")
test=pd.read_csv("../input/Kannada-MNIST/test.csv")
train.head()
test.head()
Y_train = train['label']

X_train = train.drop(columns=['label'])

X_test = test.drop(columns=['id'])
print("x_train shape:", X_train.shape, "y_train shape:", Y_train.shape) 
#X_train=np.array(X_train)

#Y_train=np.array(Y_train)

#X_test=np.array(X_test)
X_train.values[100]
plt.imshow(X_train.values[100].reshape(28,28), cmap = plt.cm.binary, interpolation = 'nearest') #plt.axis("off")

plt.show()
digit_train, counts_train = np.unique(Y_train, return_counts = True)
plt.bar(digit_train,counts_train,width =0.6)

plt.title('Distribution of Y_train')

plt.xlabel('Digit Number')

plt.ylabel('Counts')

plt.show()
#Using Standardization Scaler method

from sklearn.preprocessing import StandardScaler 

scaler=StandardScaler()

#from sklearn.preprocessing import MinMaxScaler

#scaler=MinMaxScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))



#X_train_scaled=scaler.fit_transform(X_train.values)

X_test_scaled=pd.DataFrame(scaler.transform(X_test.values.reshape(len(X_test),784)))
type(X_train_scaled)
fig,ax = plt.subplots(1,2)

ax[0].imshow(X_train.values[100].reshape(28,28), cmap = plt.cm.binary, interpolation = 'nearest') #plt.axis("off")

ax[0].set_title('Unscaled')

ax[1].imshow(X_train_scaled.values[100].reshape(28,28), cmap = plt.cm.binary, interpolation = 'nearest') #plt.axis("off")

ax[1].set_title('Scaled')

plt.show()
fig_object, ax_object = plt.subplots(1, 10, figsize=(12,5))

ax_object = ax_object.reshape(10,)

    

for i in range(len(ax_object)):

    ax = ax_object[i]

    idx=np.argwhere(Y_train.values==i)[0]

    ax.imshow(X_train.values[idx].reshape(28,28), cmap = plt.cm.binary, interpolation = 'nearest')

    ax.set_xlabel(Y_train.values[i])

    ax.set_title(i)

    

plt.show()



fig_object, ax_object = plt.subplots(1, 10, figsize=(12,5))

ax_object = ax_object.reshape(10,)

      

for i in range(len(ax_object)):

    ax = ax_object[i]

    idx=np.argwhere(Y_train==i)[0]

    ax.imshow(X_train_scaled.values[idx].reshape(28,28), cmap = plt.cm.binary, interpolation = 'nearest')

    ax.set_xlabel(Y_train[i])

    plt.xlabel(Y_train[i])

    ax.set_title(i)       

plt.show()
from sklearn.model_selection import train_test_split

X_train,X_val,Y_train,Y_val = train_test_split(X_train_scaled,Y_train,test_size=0.25)
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1
import keras

from matplotlib import pyplot as plt

import numpy as np

import gzip

%matplotlib inline

from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D

from keras.models import Model

from keras.optimizers import RMSprop
X_train = X_train.reshape(-1, 28,28, 1)

X_test = X_test.reshape(-1, 28,28, 1)

X_train.shape, X_test.shape
batch_size = 128

epochs = 128

inChannel = 1

x, y = 28, 28

input_img = Input(shape = (x, y, inChannel))
def autoencoder(input_img):

    #encoder

    #input = 28 x 28 x 1 (wide and thin)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)



    #decoder

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128

    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128

    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64

    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1

    return decoded
autoencoder = Model(input_img, autoencoder(input_img))

autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoencoder.summary()
autoencoder_train = autoencoder.fit(X_train, Y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_val, Y_val))
X_train.shape
X_val.shape
import keras

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

#Import libraries to build the model
model=Sequential()
model.add(Convolution2D(64,3,3,input_shape=(28,28,1),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128,3,3,input_shape=(28,28,1),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(output_dim=256,activation='relu'))

model.add(Dense(output_dim=128,activation='relu'))

model.add(Dense(output_dim=10,activation='sigmoid'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
X_train=X_train.reshape(-1,28,28,1)
epochs=30



# fits the model on batches with real-time data augmentation:

history=model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=64, validation_data=(X_val.values.reshape(len(X_val),28,28,1), Y_val))
test_loss, test_acc = model.evaluate(X_val.values.reshape(len(X_val),28,28,1),  Y_val,verbose=2)

print('\nTest accuracy:', test_acc)
plt.plot(range(epochs),history.history['accuracy'])

plt.xlabel('No. of Epochs')

plt.ylabel('Accuracy')

plt.title('Training Accuracy')

plt.show()

plt.plot(range(epochs),history.history['loss'])

plt.xlabel('No. of Epochs')

plt.ylabel('Loss Value')

plt.title('Loss Function')

plt.show()
predict = model.predict(X_val.values.reshape(len(X_val),28,28,1))
Y_pred=[]

for i in range(len(predict)):

    Y_pred.append(np.argmax(predict[i,:]))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_val,Y_pred)

print(cm)
acc=0

for j in range(len(cm)):

    acc=acc+cm[j,j]

print(acc/15000*100)
submission=pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")
test_predict = model.predict(X_test_scaled.values.reshape(len(X_test),28,28,1))

Y_predict=np.argmax(test_predict,axis=1)

submission['label']=Y_predict

submission['id']=range(0,len(X_test_scaled))
submission.to_csv('submission.csv',index=False)
from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D

from keras import backend as K



# create the base pre-trained model

base_model = InceptionV3(weights='imagenet', include_top=False)



# add a global spatial average pooling layer

x = base_model.output

x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 200 classes

predictions = Dense(200, activation='softmax')(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)



# first: train only the top layers (which were randomly initialized)

# i.e. freeze all convolutional InceptionV3 layers

for layer in base_model.layers:

    layer.trainable = False



# compile the model (should be done *after* setting layers to non-trainable)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')



# train the model on the new data for a few epochs

model.fit_generator(...)



# at this point, the top layers are well trained and we can start fine-tuning

# convolutional layers from inception V3. We will freeze the bottom N layers

# and train the remaining top layers.



# let's visualize layer names and layer indices to see how many layers

# we should freeze:

for i, layer in enumerate(base_model.layers):

    print(i, layer.name)



# we chose to train the top 2 inception blocks, i.e. we will freeze

# the first 249 layers and unfreeze the rest:

for layer in model.layers[:249]:

    layer.trainable = False

for layer in model.layers[249:]:

    layer.trainable = True



# we need to recompile the model for these modifications to take effect

# we use SGD with a low learning rate

from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')



# we train our model again (this time fine-tuning the top 2 inception blocks

# alongside the top Dense layers

model.fit_generator(...)
shape_x = 28

shape_y = 28

nRows,nCols,nDims = X_train_scaled.shape[1:]

input_shape = (nRows, nCols, nDims)

classes = np.unique(Y_train)

nClasses = len(classes)