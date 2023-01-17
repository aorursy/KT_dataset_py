import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
IMAGE_LIB = '../input/2d_images/'
MASK_LIB = '../input/2d_masks/'
IMG_HEIGHT, IMG_WIDTH = 32, 32
SEED=42
all_images = [x for x in sorted(os.listdir(IMAGE_LIB)) if x[-4:] == '.tif']

x_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(IMAGE_LIB + name, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    x_data[i] = im

y_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(MASK_LIB + name, cv2.IMREAD_UNCHANGED).astype('float32')/255.
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    y_data[i] = im
fig, ax = plt.subplots(1,2, figsize = (8,4))
ax[0].imshow(x_data[0], cmap='gray')
ax[1].imshow(y_data[0], cmap='gray')
plt.show()
x_data = x_data[:,:,:,np.newaxis]
y_data = y_data[:,:,:,np.newaxis]
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 0.5)
print("Size of test set = ", len(x_val))
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def gdice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    A = K.sum(y_true_f)
    B = K.sum(y_pred_f)
    
    TP = K.sum(y_true_f * y_pred_f)
    two_TP = 2. * TP
    
    FN = A - TP
    FP = B - TP
    FP_1 = FP + (K.square(FP) / ( TP + FN + K.epsilon() ))
    
    G_DICE = (two_TP + K.epsilon()) / (two_TP + FN + FP_1 + K.epsilon())
    return (G_DICE)

def gdice_coef_loss(y_true, y_pred):
    return -gdice_coef(y_true, y_pred)

def sgdice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    A = K.sum(y_true_f)
    B = K.sum(y_pred_f)
    
    TP = K.sum(y_true_f * y_pred_f)
    two_TP = 2. * TP
    
    FN = A - TP
    FP = B - TP
    FP_1 = FP + (K.square(FP) / ( TP + FN - FP + K.epsilon() ))
    
    SG_DICE = (two_TP + K.epsilon()) / (two_TP + FN + FP_1 + K.epsilon())
    return (SG_DICE)

def sgdice_coef_loss(y_true, y_pred):
    return -sgdice_coef(y_true, y_pred)

def point_wise_binary_cross_entropy (y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y = K.sum(y_true_f)
    y_hat = K.sum(y_pred_f)
    beta = 0.71
    PWBCE = K.sum(-beta * (y * K.log(y_hat)) - ((1 - beta) * (1 - y) * K.log(1 - y_hat)) )
    return PWBCE

def pwbce_loss(y_true, y_pred):
    return -point_wise_binary_cross_entropy(y_true, y_pred)

input_layer = Input(shape=x_train.shape[1:])
c1 = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
l = MaxPool2D(strides=(2,2))(c1)
c2 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c2)
c3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c3)
c4 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(c4), c3], axis=-1)
l = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(l), c2], axis=-1)
l = Conv2D(filters=24, kernel_size=(2,2), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(l), c1], axis=-1)
l = Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding='same')(l)
l = Conv2D(filters=64, kernel_size=(1,1), activation='relu')(l)
l = Dropout(0.5)(l)
output_layer = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l)
                                                         
model_a = Model(input_layer, output_layer)
model_b = Model(input_layer, output_layer)
model_c = Model(input_layer, output_layer)
model_a.summary()
model_b.summary()
model_c.summary()
from keras.utils.vis_utils import plot_model
plot_model(model_a, to_file='model_a_plot.png', show_shapes=True, show_layer_names=True)
plot_model(model_b, to_file='model_b_plot.png', show_shapes=True, show_layer_names=True)
plot_model(model_c, to_file='model_c_plot.png', show_shapes=True, show_layer_names=True)
def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch
image_batch, mask_batch = next(my_generator(x_train, y_train, 8))
fix, ax = plt.subplots(8,2, figsize=(8,20))
for i in range(8):
    ax[i,0].imshow(image_batch[i,:,:,0])
    ax[i,1].imshow(mask_batch[i,:,:,0])
plt.show()
#model.compile(optimizer=Adam(2e-4), loss=dice_coef_loss, metrics=[dice_coef, gdice_coef])
#model.compile(optimizer=Adam(2e-4), loss=gdice_coef_loss, metrics=[dice_coef, gdice_coef])
#model.compile(optimizer=Adam(2e-4), loss=pwbce_loss, metrics=[dice_coef, gdice_coef])
model_a.compile(optimizer=Adam(2e-4), loss=dice_coef_loss, metrics=[dice_coef, gdice_coef, sgdice_coef])
model_b.compile(optimizer=Adam(2e-4), loss=gdice_coef_loss, metrics=[dice_coef, gdice_coef, sgdice_coef])
model_c.compile(optimizer=Adam(2e-4), loss=sgdice_coef_loss, metrics=[dice_coef, gdice_coef, sgdice_coef])
weight_saver_a = ModelCheckpoint('lung_a.h5', monitor='val_dice_coef', 
                                              save_best_only=True, save_weights_only=True)
weight_saver_b = ModelCheckpoint('lung_b.h5', monitor='val_gdice_coef', 
                                              save_best_only=True, save_weights_only=True)
weight_saver_c = ModelCheckpoint('lung_c.h5', monitor='val_sgdice_coef', 
                                              save_best_only=True, save_weights_only=True)
annealer_a = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)
annealer_b = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)
annealer_c = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)
hist_a = model_a.fit_generator(my_generator(x_train, y_train, 8),
                               steps_per_epoch = 200,
                               validation_data = (x_val, y_val),
                               epochs=50, verbose=2,
                               callbacks = [weight_saver_a, annealer_a])
hist_b = model_b.fit_generator(my_generator(x_train, y_train, 8),
                               steps_per_epoch = 200,
                               validation_data = (x_val, y_val),
                               epochs=50, verbose=2,
                               callbacks = [weight_saver_b, annealer_b])
hist_c = model_c.fit_generator(my_generator(x_train, y_train, 8),
                               steps_per_epoch = 200,
                               validation_data = (x_val, y_val),
                               epochs=50, verbose=2,
                               callbacks = [weight_saver_c, annealer_c])
loss_values_a = hist_a.history['loss']
loss_values_b = hist_b.history['loss']
loss_values_c = hist_c.history['loss']

dice_values = hist_a.history['dice_coef']
gdice_values = hist_b.history['gdice_coef']
sgdice_values = hist_c.history['sgdice_coef']

export_data = pd.DataFrame()
export_data['loss_values_a'] = loss_values_a
export_data['loss_values_b'] = loss_values_b
export_data['loss_values_c'] = loss_values_c
export_data['dice_coef'] = dice_values
export_data['gdice_coef'] = gdice_values
export_data['sgdice_coef'] = gdice_values

export_data.to_csv("outputData.csv", sep = ";")

model_a.load_weights('lung_a.h5')
model_b.load_weights('lung_b.h5')
model_c.load_weights('lung_c.h5')
plt.plot(hist_a.history['loss'], color='b')
plt.plot(hist_a.history['val_loss'], color='r')
plt.show()
plt.plot(hist_a.history['dice_coef'], color='b')
plt.plot(hist_a.history['gdice_coef'], color='g')
plt.plot(hist_a.history['sgdice_coef'], color='r')
plt.show()

plt.plot(hist_b.history['loss'], color='b')
plt.plot(hist_b.history['val_loss'], color='r')
plt.show()
plt.plot(hist_b.history['dice_coef'], color='b')
plt.plot(hist_b.history['sgdice_coef'], color='g')
plt.plot(hist_b.history['sgdice_coef'], color='r')
plt.show()

plt.plot(hist_c.history['loss'], color='b')
plt.plot(hist_c.history['val_loss'], color='r')
plt.show()
plt.plot(hist_c.history['dice_coef'], color='b')
plt.plot(hist_c.history['sgdice_coef'], color='g')
plt.plot(hist_c.history['sgdice_coef'], color='r')
plt.show()
plt.imshow(model_a.predict(x_train[0].reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0], cmap='gray')
plt.imshow(model_b.predict(x_train[0].reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0], cmap='gray')
plt.imshow(model_c.predict(x_train[0].reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0], cmap='gray')
y_hat_a = model_a.predict(x_val)
y_hat_b = model_b.predict(x_val)
y_hat_c = model_c.predict(x_val)

fig, ax = plt.subplots(1,3,figsize=(12,6))
ax[0].imshow(x_val[0,:,:,0], cmap='gray')
ax[1].imshow(y_val[0,:,:,0])
ax[2].imshow(y_hat_a[0,:,:,0])

fig, ax = plt.subplots(1,3,figsize=(12,6))
ax[0].imshow(x_val[0,:,:,0], cmap='gray')
ax[1].imshow(y_val[0,:,:,0])
ax[2].imshow(y_hat_b[0,:,:,0])

fig, ax = plt.subplots(1,3,figsize=(12,6))
ax[0].imshow(x_val[0,:,:,0], cmap='gray')
ax[1].imshow(y_val[0,:,:,0])
ax[2].imshow(y_hat_c[0,:,:,0])
sum(y_hat_a == y_hat_b)
sum(y_hat_a == y_hat_c)
sum(y_hat_b == y_hat_c)
print("Analysis of learning method: a.")
print("____________________________________________")

y_hat = y_hat_a #or _b, or _c

# Absolute Difference
dif = abs(y_hat - y_val).sum()
print("Diferença absoluta = ", dif)
# dice = 10316.324

# Real and Predicted values
print("Soma y_previsto = ", y_hat.sum())
print("Soma y_real = ", y_val.sum())

# MSE
b = (1/len(y_val))
MSE = (y_val - y_hat)**2
MSE = b * MSE.sum()
print("Mean Squared Error = ", MSE)

# RMSE
import math 
RMSE = math.sqrt(MSE)
print("Root Mean Squared Error = ", RMSE)

# Normalized RMSE
NRMSE = RMSE / y_val.mean()
print("Normalized Root Mean Squared Error = ", NRMSE)

# Mean of y
print("Mean of y = ", y_val.mean())

print(y_val.ndim)
print(y_val.size)
print(y_val.shape)
print(y_hat.ndim)
print(y_hat.size)
print(y_hat.shape)
print("Analysis of learning method: b.")
print("____________________________________________")

y_hat = y_hat_b #or _b, or _c

# Absolute Difference
dif = abs(y_hat - y_val).sum()
print("Diferença absoluta = ", dif)
# dice = 10316.324

# Real and Predicted values
print("Soma y_previsto = ", y_hat.sum())
print("Soma y_real = ", y_val.sum())

# MSE
b = (1/len(y_val))
MSE = (y_val - y_hat)**2
MSE = b * MSE.sum()
print("Mean Squared Error = ", MSE)

# RMSE
import math 
RMSE = math.sqrt(MSE)
print("Root Mean Squared Error = ", RMSE)

# Normalized RMSE
NRMSE = RMSE / y_val.mean()
print("Normalized Root Mean Squared Error = ", NRMSE)

# Mean of y
print("Mean of y = ", y_val.mean())

print(y_val.ndim)
print(y_val.size)
print(y_val.shape)
print(y_hat.ndim)
print(y_hat.size)
print(y_hat.shape)
print("Analysis of learning method: c.")
print("____________________________________________")

y_hat = y_hat_c #or _b, or _c

# Absolute Difference
dif = abs(y_hat - y_val).sum()
print("Diferença absoluta = ", dif)
# dice = 10316.324

# Real and Predicted values
print("Soma y_previsto = ", y_hat.sum())
print("Soma y_real = ", y_val.sum())

# MSE
b = (1/len(y_val))
MSE = (y_val - y_hat)**2
MSE = b * MSE.sum()
print("Mean Squared Error = ", MSE)

# RMSE
import math 
RMSE = math.sqrt(MSE)
print("Root Mean Squared Error = ", RMSE)

# Normalized RMSE
NRMSE = RMSE / y_val.mean()
print("Normalized Root Mean Squared Error = ", NRMSE)

# Mean of y
print("Mean of y = ", y_val.mean())

print(y_val.ndim)
print(y_val.size)
print(y_val.shape)
print(y_hat.ndim)
print(y_hat.size)
print(y_hat.shape)