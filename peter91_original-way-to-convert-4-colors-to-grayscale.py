import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm_notebook
import os

data_dir = "../input/"

w_size = 1024
o_size = 512

train = pd.read_csv(data_dir+"/train.csv")
print(train.head())

train_dataset_info = []

for name, labels in zip(train['Id'], train['Target'].str.split(' ')):
        lb = np.zeros(28, dtype='int')
        for label in labels:
            lb[int(label) ] = 1
        train_dataset_info.append({
        'path':os.path.join(data_dir+"train/", name),
        'labels':lb})
train_dataset_info = np.array(train_dataset_info)
train_num = train_dataset_info.shape[0]
print (" images num ", train_num)
idx = np.zeros((28),dtype='int')
tst = np.zeros((train_num),dtype='int')[:]>0
for k in range(train_num):
    for i in range(28):
        if idx[i]<100 and train_dataset_info[k]["labels"][i] > 0:
            idx += train_dataset_info[k]["labels"]
            tst[k] = True
            break
w_train_dataset_info = train_dataset_info[tst]
w_num = w_train_dataset_info.shape[0]
#w_num, idx
num_classes = 28
w_imgs = np.zeros((w_num,w_size,w_size,1), dtype='float32')
w_class = np.zeros((w_num,num_classes), dtype='float32')

for k in tqdm_notebook(range(w_num)):
    red =   np.array(Image.open(w_train_dataset_info[k]["path"]+"_red.png"   ))
    green = np.array(Image.open(w_train_dataset_info[k]["path"]+"_green.png" ))
    blue =  np.array(Image.open(w_train_dataset_info[k]["path"]+"_blue.png"  ))
    yellow =np.array(Image.open(w_train_dataset_info[k]["path"]+"_yellow.png"))
    w_imgs[k,::2,::2,0]  = red/255.
    w_imgs[k,1::2,::2,0] = blue/255.
    w_imgs[k,::2,1::2,0] = green/255.
    w_imgs[k,1::2,1::2,0]= yellow/255.
    w_class[k] = w_train_dataset_info[k]["labels"]

from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, Activation, Add
from keras.layers import Dense, Flatten, BatchNormalization, AveragePooling2D
from keras.losses import binary_crossentropy
import tensorflow as tf
#import keras as keras
from keras import backend as K

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
'''
Thanks Iafoss.
pretrained ResNet34 with RGBY
https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
'''
gamma = 2.0
epsilon = K.epsilon()
def focal_loss(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    y_true = tf.convert_to_tensor(y_true, np.float32)
    pt = y_pred * y_true + (1-y_pred) * (1-y_true)
    pt = K.clip(pt, epsilon, 1-epsilon)
    CE = -K.log(pt)
    FL = K.pow(1-pt, gamma) * CE
    loss = K.sum(FL, axis=1)
    #return binary_crossentropy(y_true, y_pred) + loss
    return loss

from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({'focal_loss': focal_loss })
get_custom_objects().update({'f1': f1 })

def build_model(input_layer, start_neurons):

    conv1 = Conv2D(start_neurons * 1, (14, 14), strides=(2, 2), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

    drop = Dropout(0.5)(convm)

    flat = Flatten()(drop)
#    hidden = Dense(128, activation='relu')(flat)
#    drop = Dropout(0.5)(hidden)
    output_layer = Dense(num_classes, activation='sigmoid')(flat)
    
    return output_layer
input_layer = Input((w_size, w_size, 1))
output_layer = build_model(input_layer, 8)
model = Model(input_layer, output_layer)
model.compile(loss="focal_loss", optimizer=Adam(lr=1e-3), metrics=["binary_accuracy","f1"])
# model.save_weights('./init.weights')

model.summary()
early_stopping = EarlyStopping(monitor='f1', mode = 'max',patience=10, verbose=1)
#model_checkpoint = ModelCheckpoint("./keras_1-28_200.model",monitor='val_f1', 
#                               mode = 'max', save_best_only=True, verbose=1)
#reduce_lr = ReduceLROnPlateau(monitor='val_f1', 
#                              mode = 'max',
#                              factor=0.2, 
#                              patience=5, min_lr=0.00001, 
#                              verbose=1)

history = model.fit(w_imgs,w_class,
                        epochs=20,
                        batch_size=32,
                        callbacks=[early_stopping],
                        verbose=2)

