import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
plt.style.use("ggplot")
%matplotlib inline

import cv2
from tqdm import tqdm_notebook, tnrange
from glob import glob
from itertools import chain
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


from tensorflow.keras import backend as K
train_files = []
mask_files = glob('../input/lgg-mri-segmentation/kaggle_3m/*/*_mask*')

for i in mask_files:
    train_files.append(i.replace('_mask',''))
train_files[0], mask_files[0]
img = cv2.imread(train_files[0])
plt.imshow(img)
msk = cv2.imread(mask_files[0])
plt.imshow(msk, alpha=0.4)
plt.grid(False)
#split ratio
split=0.1
total_size = len(train_files)
valid_size = int(split * total_size)
test_size = int(split * total_size)

#getting the val data
train_x, valid_x = train_test_split(train_files, test_size=valid_size, random_state=42)
train_y, valid_y = train_test_split(mask_files, test_size=valid_size, random_state=42)

#getting the test data
train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)
#number of train, valid and test data
print(len(train_x), len(valid_x), len(test_x))
train_x[0], train_y[0]
fig=plt.figure(figsize=(10,10))

fig.add_subplot(1,2,1)
img = cv2.imread(train_x[0])
plt.imshow(img)
plt.grid(False)
fig.add_subplot(1,2,2)
msk = cv2.imread(train_y[0])
plt.imshow(msk)
plt.grid(False)
valid_x[0], valid_y[0]
fig=plt.figure(figsize=(10,10))

fig.add_subplot(1,2,1)
img = cv2.imread(valid_x[0])
plt.imshow(img)
plt.grid(False)
fig.add_subplot(1,2,2)
msk = cv2.imread(valid_y[0])
plt.imshow(msk)
plt.grid(False)
test_x[0], test_y[0]
fig=plt.figure(figsize=(10,10))

fig.add_subplot(1,2,1)
img = cv2.imread(test_x[0])
plt.imshow(img)
plt.grid(False)
fig.add_subplot(1,2,2)
msk = cv2.imread(test_y[0])
plt.imshow(msk)
plt.grid(False)
smooth=100

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac
#installing tensorflow_io to decode the tiff images
!pip install tensorflow-io
import tensorflow_io as tfio
def parse_function(filename, label):
    '''
    This function reads and decodes the image and its corresponding mask
    and perform operations like normamlization, converting to float32,
    resizing the image and return the image and the mask.
    Arguments
    -----------------------
    filename - the image path
    label    - the mask path
    ------------------------
    '''

    image_string = tf.io.read_file(filename)
    label_string = tf.io.read_file(label)

    image = tfio.experimental.image.decode_tiff(image_string)
    label = tfio.experimental.image.decode_tiff(label_string)

    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.image.convert_image_dtype(label, tf.float32)

    resized_image = tf.image.resize(image, [256, 256])
    resized_label = tf.image.resize(label, [256, 256])
    if tf.random.uniform(()) > 0.5:
        resized_image = tf.image.flip_left_right(resized_image)
        resized_label = tf.image.flip_left_right(resized_label)

    resized_image = tf.convert_to_tensor(resized_image[:,:,:3])
    resized_label = tf.convert_to_tensor(resized_label[:,:,:1])
    
    return resized_image, resized_label
#creating train dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.shuffle(len(train_x))
train_dataset = train_dataset.map(parse_function, num_parallel_calls=4)
train_dataset = train_dataset.batch(32)
train_dataset = train_dataset.prefetch(1)
#creating validaion dataset
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_x, valid_y))
valid_dataset = valid_dataset.shuffle(len(valid_x))
valid_dataset = valid_dataset.map(parse_function, num_parallel_calls=4)
valid_dataset = valid_dataset.batch(32)
valid_dataset = valid_dataset.prefetch(1)
#creating test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_dataset = test_dataset.shuffle(len(test_x))
test_dataset = test_dataset.map(parse_function, num_parallel_calls=4)
test_dataset = test_dataset.batch(32)
test_dataset = test_dataset.prefetch(1)
#let's check the shape of our data
for x, y in train_dataset.take(1):
  print('Train Image shape:', x.shape)
  print('Train Mask shape:', y.shape)
def conv(x, filters, kernel_size=(3, 3), strides=1):
    ''' 
    This function will create the ConvBlock and return it
    Arguments:
    -----------------------------
    x - previous 
    filters - value for the filters
    kernel_size - size of the kernels default value is (3,3)
    strides - size of the strids default value is 1
    ------------------------------
    '''

    conv = keras.layers.BatchNormalization()(x)
    conv = keras.layers.Activation('relu')(conv)
    conv = keras.layers.Conv2D(filters, kernel_size, padding='same', strides=strides)(conv)
    return conv

def res(x, filters, strides=1):
    '''
    This function will create the residual block and returns it.
    Arguments:
    ----------------------
    x - previous 
    filters - size of the filters
    strides - size of the strides default value 1
    ----------------------
    '''

    r = conv(x, filters, kernel_size=(3, 3), strides=strides)
    r = conv(r, filters, kernel_size=(3, 3), strides=1)
    
    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding="same", strides=strides)(x)
    shortcut = keras.layers.BatchNormalization()(shortcut)

    output = keras.layers.Add()([shortcut, r])
    return output
def ResUNet():
    '''
    This function will create the ResUNet model and returns it.
    '''

    inputs = keras.layers.Input((256, 256, 3))
    enc0 = inputs
    c = keras.layers.Conv2D(16, (3, 3), padding="same", strides=1)(enc0)
    c2 = conv(c, 16, kernel_size=(3, 3), strides=1)
    s = keras.layers.Conv2D(16, kernel_size=(1, 1), padding="same", strides=1)(enc0)
    s2 = keras.layers.BatchNormalization()(s)
    
    enc1 = keras.layers.Add()([c2, s2])
    enc2 = res(enc1, 32, strides=2)
    enc3 = res(enc2, 64, strides=2)
    enc4 = res(enc3, 128, strides=2)
    enc5 = res(enc4, 256, strides=2)
    
    b0 = conv(enc5, 256, strides=1)
    b1 = conv(b0, 256, strides=1)
    
    upsamp1 = keras.layers.UpSampling2D((2, 2))(b1)
    upsamp1 = keras.layers.Concatenate()([upsamp1, enc4])
    dec1 = res(upsamp1, 256)
    
    upsamp2 = keras.layers.UpSampling2D((2, 2))(dec1)
    upsamp2 = keras.layers.Concatenate()([upsamp2, enc3])
    dec2 = res(upsamp2, 128)
    
    upsamp3 = keras.layers.UpSampling2D((2, 2))(dec2)
    upsamp3 = keras.layers.Concatenate()([upsamp3, enc2])
    dec3 = res(upsamp3, 64)
    
    upsamp4 = keras.layers.UpSampling2D((2, 2))(dec3)
    upsamp4 = keras.layers.Concatenate()([upsamp4, enc1])
    dec4 = res(upsamp4, 32)
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(dec4)
    model = keras.models.Model(inputs, outputs)
    return model
model = ResUNet()
model.summary()
#set the hyper-parameters
im_width = 256
im_height = 256
epochs = 100
batch = 32
learning_rate = 1e-4
#defining callbacks

#early stopping to stop the training when the val_loss stops improving for 11 epochs
es = EarlyStopping(monitor='val_loss', patience=11)

#reduce the learning rate once the val_loss stops improving after certain epochs
rlr =  ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 5,
                        verbose = 10, mode = "auto", min_delta = 1e-04, cooldown = 0,
                        min_lr = 1e-5)

#save the best model
#we'll save the model in the tensorflow's saved_model format
model_filename = "test-Epoch-{epoch:02d}"
checkpoint_path = os.path.join('models/', model_filename)
cpt = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
#creating the model
model = ResUNet()

#initializing the optimizer and compiling the model
opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, \
           decay=learning_rate / epochs, amsgrad=False)
model.compile(optimizer=opt, loss=dice_coef_loss, metrics=["binary_accuracy", iou, dice_coef])

#defining the training steps
train_steps = len(train_x)//batch
valid_steps = len(valid_x)//batch
if len(train_x) % batch != 0:
    train_steps += 1
if len(valid_x) % batch != 0:
    valid_steps += 1

#make a list of callbacks
callbacks = [cpt, es, rlr]

#fit the model on the train data and validate on the validation split
history = model.fit(train_dataset,
    validation_data=valid_dataset,
    epochs=epochs,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    callbacks=callbacks)
#saving the history just in case if we need it later
pd.DataFrame.from_dict(history.history).to_csv('history.csv',index=False)
hist = history.history

list_traindice = hist['dice_coef']
list_testdice = hist['val_dice_coef']

list_trainjaccard = hist['iou']
list_testjaccard = hist['val_iou']

list_trainloss = hist['loss']
list_testloss = hist['val_loss']

plt.figure(1)
plt.plot(list_testloss, 'b-', label='valid')
plt.plot(list_trainloss,'r-', label='train')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend()
plt.title('loss', fontsize = 15)

plt.figure(2)
plt.plot(list_traindice, 'r-', label='train')
plt.plot(list_testdice, 'b-', label='valid')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.legend()
plt.title('accuracy', fontsize = 15)
plt.show()
loaded = tf.saved_model.load("./models/test-Epoch-51")
print(list(loaded.signatures.keys()))
infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)
#visualize the predicted and ground truth mask
#set the image width and height
im_width = 256
im_height = 256

#set random seed
np.random.seed(1435)

#visualize the output for 15 images
for i in range(15):
    
    index=np.random.randint(1, len(test_x))
    
    img = tf.keras.preprocessing.image.load_img(test_x[index], target_size=[im_width, im_height])
    img = tf.keras.preprocessing.image.img_to_array(img) / 255
    
    img = img[tf.newaxis,...]
    img = tf.cast(img, tf.float32)
    
    pred = infer(tf.constant(img))[model.output_names[0]]
    
    #plot the original image
    plt.figure(figsize=(12,12))
    plt.subplot(1,3,1)
    plt.imshow(tf.squeeze(img))
    plt.grid(False)
    plt.title('Original Image')

    #plot the original mask
    plt.subplot(1,3,2)
    msk = tf.keras.preprocessing.image.load_img(test_y[index], target_size=[im_width,im_height])
    plt.imshow(msk)
    plt.grid(False)
    plt.title('Original Mask')
    
    #plot the predicted mask
    plt.subplot(1,3,3)
    plt.imshow(tf.squeeze(pred) > .5)
    plt.grid(False)
    plt.title('Prediction')
    plt.show()

