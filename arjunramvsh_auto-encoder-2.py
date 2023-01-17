import numpy as np
import matplotlib.pyplot as plt 
%matplotlib inline

from keras.datasets import cifar10 
(train,_) , (test,_) = cifar10.load_data()
print(train.shape[1:],test.shape)
#normalizr the data sets 
train_norm = train.astype('float32')/255
test_norm = test.astype('float32')/255
#adding noise to the images 
def add_noise_and_clip_data(data):
   noise = np.random.normal(loc=0.0, scale=0.3, size=data.shape)
   data = data + noise
   data = np.clip(data, 0., 1.)
   return data
train_noise = add_noise_and_clip_data(train_norm)
test_noise = add_noise_and_clip_data(test_norm)
index = 12
plt.subplot(1,2,1)
plt.imshow(train[index])
plt.title('Original image')
plt.subplot(1,2,2)
plt.imshow(train_noise[index])
plt.title('Image with noise')
plt.show()
from keras.layers import Conv2D, Input, Dense, Reshape, Conv2DTranspose, Activation, BatchNormalization, ReLU,Concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
#encoder function block
def encode(x, filters, kernel_size, strides=2):
    x = Conv2D(filters=filters, kernel_size=kernel_size,
                  strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x
#decoder function block 
def decode(x,filters,kernel_size,strides=2):
    x = Conv2DTranspose(filters=filters,kernel_size = kernel_size,
                           strides = strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x 
# the convolutional layers (4) with downsampling and (1) without
def denoising_autoencoder():
    # the encoder
    inputs = Input(shape=train.shape[1:],name="inputs")
    conv1 = encode(inputs,32,3)
    conv2 = encode(conv1,64,3)
    conv3 = encode(conv2,128,3)
    conv4 = encode(conv3,256,3)
    conv5 = encode(conv4,256,3,1)
    
    # the decoder
    dconv1 = decode(conv5,256,3)
    merge1 = Concatenate()([dconv1,conv3])
    dconv2 = decode(merge1,128,3)
    merge2 = Concatenate()([dconv2,conv2])
    dconv3 = decode(merge2,64,3)
    merge3 = Concatenate()([dconv3,conv1])
    dconv4 = decode(merge3,32,3)
    
    dconv5 = Conv2DTranspose(filters=3,kernel_size=3,
                                padding='same')(dconv4)
    output = Activation('sigmoid',name='output')(dconv5)
    # dae - denoising auto encoder
    return Model(inputs,output,name='dae')
model = denoising_autoencoder()
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
checkpoint = ModelCheckpoint('model1.h5', verbose=1, save_best_only=True, save_weights_only=True)
model.summary()
model.fit(train_noise, train_norm, validation_data = (test_noise,test_norm),
             epochs = 40, batch_size=128,
             callbacks = [checkpoint])
predicted = model.predict(test_noise)

idx = 12
plt.subplot(1,3,1)
plt.imshow(test[idx])
plt.title('original')
plt.subplot(1,3,2)
plt.imshow(test_noise[idx])
plt.title('noisy')
plt.subplot(1,3,3)
plt.imshow(predicted[idx])
plt.title('denoised')
plt.show()
from random import randint

fig,ax = plt.subplots(4,3,figsize=(11,11))

for i in range(4):
    idx = randint(0,10000)
#     plt.subplot(2,3,i+1,figsize=(7,7))
    ax[i,0].imshow(test[idx])
    ax[i,0].title.set_text('original')
#     fig.title('original')
#     plt.subplot(2,3,i+2,figsize=(7,7))
    ax[i,1].imshow(test_noise[idx])
    ax[i,1].title.set_text('noisy')
#     fig.title('noisy')
#     plt.subplot(2,3,i+3,figsize=(7,7))
    ax[i,2].imshow(predicted[idx])
    ax[i,2].title.set_text('denoised')
#     fig.title('denoised')
plt.show()
scores = model.evaluate(test_noise,test_norm,verbose=1)
print(model.metrics_names,scores)
