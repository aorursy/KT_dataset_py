## load the libraries 
from keras.layers import Dense, Input, Conv2D, LSTM, MaxPool2D, UpSampling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from numpy import argmax, array_equal
import matplotlib.pyplot as plt
from keras.models import Model
from imgaug import augmenters
from random import randint
import pandas as pd
import numpy as np
#read training dasatet
train = pd.read_csv('../input/fashion-mnist_train.csv');

#separate features and target
train_x = train[list(train.columns)[1:]].values;
train_y = train['label'].values;

#normalize
train_x = train_x / 255;

#training and validation sets
train_x, val_x, train_y, val_y = train_test_split(train_x,train_y,test_size = 0.2);

#reshape the inputs
train_x = train_x.reshape(-1,784);
val_x = val_x.reshape(-1,784);
train_x.shape
val_x.shape
#input layer
input_layer = Input(shape=(784,));

#encoding layer, 3 layers with 1500,1000,500 hidden units/nuerons
encode_layer1 = Dense(1500,activation = 'relu')(input_layer);
encode_layer2 = Dense(1000,activation = 'relu')(encode_layer1);
encode_layer3 = Dense(500,activation = 'relu')(encode_layer2);

#latent view space, with 10 nodes
latent_view = Dense(10,activation = 'relu')(encode_layer3);

#decoding layer, 3 layers with 500,1000,1500 hidden units/nuerons
decode_layer1 = Dense(500,activation = 'relu')(latent_view);
decode_layer2 = Dense(1000,activation = 'relu')(decode_layer1);
decode_layer3 = Dense(500,activation = 'relu')(decode_layer2);

#output layer with same nodes i.e. 784 as input layer with linear activation
output_layer = Dense(784)(decode_layer3);

#model
model = Model(input_layer,output_layer)

#summry of the model
model.summary()
#compile with Adamoptimizer and MSE loss metric
model.compile(optimizer='adam',loss='mse');

#early stopping
early_stopping = EarlyStopping(monitor = 'val_loss',min_delta = 0, patience = 10,verbose = 1,mode = 'auto');

#train the model
model.fit(train_x,train_x,epochs  = 20,batch_size = 2048,validation_data=(val_x,val_x),callbacks = [early_stopping]);

#predict on val data
pred = model.predict(val_x);

#plot images from input data
from PIL import Image 
f, ax = plt.subplots(1,5)
f.set_size_inches(80, 40)
for i in range(5):
    ax[i].imshow(val_x[i].reshape(28, 28))
plt.show()

f, ax = plt.subplots(1,5)
f.set_size_inches(80, 40)
for i in range(5):
    ax[i].imshow(pred[i].reshape(28, 28))
plt.show()
#recreate train and validation data
train_x = train[list(train.columns)[1:]].values;
trani_x, val_x = train_test_split(train_x, test_size = 0.2);

#normalize data
train_x = train_x/255;
val_x = val_x/255;
#reshape input into 28*28 matrix for the conv layers
train_x = train_x.reshape(-1,28,28,1);
val_x = val_x.reshape(-1,28,28,1);
noise = augmenters.SaltAndPepper(0.1);
seq_object = augmenters.Sequential([noise]);

train_xn = seq_object.augment_images(train_x * 255)/255;
val_xn = seq_object.augment_images(val_x * 255)/255;
f, ax = plt.subplots(1,5)
f.set_size_inches(80, 40)
for i in range(5,10):
    ax[i-5].imshow(train_x[i].reshape(28, 28))
plt.show()
f, ax = plt.subplots(1,5)
f.set_size_inches(80, 40)
for i in range(5,10):
    ax[i-5].imshow(train_xn[i].reshape(28, 28))
plt.show()
#input layer
input_layer = Input(shape = (28,28,1));

#encoding layers
encoding_layer1 = Conv2D(64,(3,3),activation = 'elu',padding = 'same')(input_layer);
encoding_layer2 = MaxPool2D((2,2),padding = 'same')(encoding_layer1);
encoding_layer3 = Conv2D(32,(3,3),activation = 'elu',padding = 'same')(encoding_layer2);
encoding_layer4 = MaxPool2D((2,2),padding = 'same')(encoding_layer3);
encoding_layer5 = Conv2D(16,(3,3),activation = 'elu',padding = 'same')(encoding_layer4);

#latent view
latent_view = MaxPool2D((2,2),padding = 'same')(encoding_layer5);

#decoding layers
decoding_layer1 = Conv2D(16,(3,3),activation = 'elu',padding = 'same')(latent_view);
decoding_layer2 = UpSampling2D((2,2))(decoding_layer1);
decoding_layer3 = Conv2D(32,(3,3),activation = 'elu',padding = 'same')(decoding_layer2);
decoding_layer4 = UpSampling2D((2,2))(decoding_layer3);
decoding_layer5 = Conv2D(64,(3,3),activation = 'elu',padding = 'valid')(decoding_layer4);
decoding_layer6 = UpSampling2D((2,2))(decoding_layer5);

#output layer
output_layer = Conv2D(1,(3,3),padding = 'same')(decoding_layer6);

model = Model(input_layer,output_layer);
#summary
model.summary();
model.compile(optimizer = 'adam',loss = 'mse');
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=5, mode='auto')
history = model.fit(train_xn, train_x, epochs=10, batch_size=2048, validation_data=(val_xn, val_x), callbacks=[early_stopping])
preds  = model.predict(val_xn[:10]);
f, ax = plt.subplots(1,5)
f.set_size_inches(80, 40)
for i in range(5,10):
    ax[i-5].imshow(preds[i].reshape(28, 28))
plt.show()