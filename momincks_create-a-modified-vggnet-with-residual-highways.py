import os
import time
import glob
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow_addons.layers import WeightNormalization
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
image_path = '../input/stanford-dogs-dataset/images/Images'
categories_count = len(os.listdir(image_path))
print('## %s Dog categories in this dataset.' %categories_count)
# print('\nFiles Directory:')
    
# for root,dirs,files in os.walk(image_path):
#     print('root:',root)
#     print('dirs:',dirs)
#     print('files:',files)
# Image classification for some categories only.

###################
num_of_categories = 10
image_size = 256
###################

target_files = os.listdir(image_path)[:num_of_categories]

target_categories = []
for i in target_files:
    x = i.split('-')[1]
    target_categories.append(x)

print(target_categories)

def read_images(target_files):
    image_set = []
    label_set = []
    breed_ID = 0
    for each_file in target_files:
        for image in os.listdir(image_path + '/' + each_file):            
            image = cv2.imread(image_path + '/' + each_file + '/' + image, cv2.IMREAD_COLOR)
            image = cv2.resize(image,(image_size,image_size))
            image = image.astype('float32')
            for channel in range(3):
                image[:,:,channel] = StandardScaler().fit_transform(image[:,:,channel])
            image_set.append(image)
            label_set.append(breed_ID)
        breed_ID += 1
    image_set = np.array(image_set)
    label_set = np.array(label_set)
    label_set = keras.utils.to_categorical(label_set)
    np.save('image_set_%s.npy'%image_size,image_set)
    np.save('label_set_%s.npy'%image_size,label_set)

# read_images(target_files)
path = '../input/finding-what-dog-is-it-with-a-modified-vggnet/'     #input folder
path = ''                                                            #output folder
image_set = np.load(glob.glob(path+'image*.npy')[0])
label_set = np.load(glob.glob(path+'label*.npy')[0])
def split(image_set, label_set):
    x_train, x_test, y_train, y_test = train_test_split(image_set, label_set, train_size = 0.8, random_state = np.random)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = split(image_set, label_set)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
def datagen():
  
    datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False, samplewise_center=False,
        featurewise_std_normalization=False, samplewise_std_normalization=False,
        zca_whitening=False, zca_epsilon=1e-06, rotation_range=20, width_shift_range=0.1,
        height_shift_range=0.1, brightness_range=None, shear_range=0.1, zoom_range=0.1,
        channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=True,
        vertical_flip=False, rescale=None, preprocessing_function=None,
        data_format='channels_last', validation_split=0.0)
    
    return datagen

train_datagen = datagen()
train_datagen.fit(x_train)
###################
filters = 32
kernel_size = (3,3)
stride = (1,1)
pool_size = (3,3)
###################
def conv2D(x,filters=filters,kernel=kernel_size,stride=stride,pad='same',activate=True,WN=False):
    if activate:
        x = keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=stride,padding=pad,activation='selu')(x)
    else:
        x = keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=stride,padding=pad,activation=None)(x)
    if WN:
        tfa.addons.WeightNormalization(x)
    return x

def selu(x):
    x = tf.nn.selu(x)
    return x

def maxpool2D(x,pool_size=pool_size,stride=stride,pad='same'):
    x = keras.layers.MaxPool2D(pool_size=pool_size,strides=stride,padding=pad)(x)
    return x

def BN(x):
    x = keras.layers.BatchNormalization()(x)
    return x

def concat(x): # input as list
    x = keras.layers.Concatenate()(x)
    return x

def res_add(raw_x,transformed_x,keep_scale):
    x = keras.layers.Add()([raw_x*keep_scale,transformed_x*(1-keep_scale)])
    return x

def stem(x):
    x = conv2D(x,32,(3,3),(2,2))
    x = conv2D(x,64,(3,3))
    x = maxpool2D(x,(3,3),(2,2))
    x = conv2D(x,96,(3,3))
    x = conv2D(x,128,(3,3))
    x = maxpool2D(x,(3,3),(2,2))
    x_1 = conv2D(x,128,(1,1))
    x_1 = conv2D(x_1,128,(3,3))
    x = res_add(x,x_1,0.3)
    x = conv2D(x,256,(1,1))
    x = maxpool2D(x,(3,3),(2,2))
    x_1 = conv2D(x,256,(1,1))
    x_1 = conv2D(x_1,256,(3,3))
    x = res_add(x,x_1,0.3)
    x = conv2D(x,512,(1,1))
    x_1 = conv2D(x,512,(1,1))
    x_1 = conv2D(x_1,512,(3,3))
    x = res_add(x,x_1,0.3)
    x = conv2D(x,768,(1,1))
    x = maxpool2D(x,(3,3),(2,2))
    x_1 = conv2D(x,768,(1,1))
    x_1 = conv2D(x_1,768,(3,3))
    x = res_add(x,x_1,0.3)
    return x

def FC(x):
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512,activation='selu')(x)
    x = keras.layers.Dense(64,activation='selu')(x)
    x = keras.layers.Dense(num_of_categories, activation='softmax')(x)
    return x

def GAP(x):
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(num_of_categories, activation='softmax')(x)
    return x

inputs = keras.Input(shape=(image_size,image_size,3))
x = stem(inputs)
outputs = GAP(x)

model = keras.Model(inputs,outputs)
print(model.summary())
###################
total_epoch = 50
lr_init = 0.00005
batch_size = 8
###################

def scheduler(epoch):
    epoch += 1
    lr = lr_init
    threshold = 10
    depre = 0.95**(epoch-threshold)
    if epoch <= threshold:
        return lr_init
    elif lr > lr_init/20:
        lr = lr_init * depre
        return lr
    else:
        return lr

scheduler = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
# adding an earlystop so training process will stop in advance if the metrics (accuracy) doesn't improve for 10 epochs consecutively
earlystop = keras.callbacks.EarlyStopping(monitor='val_acc',mode='max',verbose=1,patience=10,restore_best_weights=True)
%%time

loss = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.Nadam(learning_rate=lr_init)
metrics = [tf.keras.metrics.CategoricalAccuracy(name='acc')]
model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
use_data_gen = True

if use_data_gen:
    history = model.fit(train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=(len(x_train)/batch_size), epochs=total_epoch, callbacks=[scheduler,earlystop],
        validation_data=(x_test, y_test), workers=0, use_multiprocessing=True, shuffle=True)
else:
    history = model.fit(x_train, y_train, epochs=total_epoch, batch_size=batch_size,
        callbacks=[scheduler,earlystop], validation_data=(x_test, y_test), shuffle=True)

model.save('model.h5')
pred = model.predict(x_test)
pred = np.argmax(pred,axis=1)
y_test = np.argmax(y_test,axis=1)
report = sklearn.metrics.classification_report(y_test,pred,target_names=target_categories)
print(report)
plt.title('model accuracy')
plt.plot(history.history['acc'],label='train accuracy')
plt.plot(history.history['val_acc'],label='test accuracy')
plt.legend()
plt.show()
plt.title('model loss')
plt.plot(history.history['loss'],label='train loss')
plt.plot(history.history['val_loss'],label='test loss')
plt.legend()
plt.show()