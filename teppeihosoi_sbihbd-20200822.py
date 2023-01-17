import glob
import os
import random

import cv2
from math import ceil
import tensorflow as tf
import numpy as np
import pandas as pd
import pandas_profiling as pdp
from keras import models, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# 乱数シード固定
seed_everything(9999)

def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=ceil(len(img_matrix_list) / ncols), ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()
    
    
# def load_images(df,inputPath,size,roomType):
#     images = []
#     for i in df['id']:
#         basePath = os.path.sep.join([inputPath, "{}_{}*".format(i,roomType)])
#         housePaths = sorted(list(glob.glob(basePath)))
#         for housePath in housePaths:
#             image = cv2.imread(housePath)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             image = cv2.resize(image, (size, size))
#         images.append(image)
#     return np.array(images) / 255.0

def load_images(df,inputPath, size = 128):
    images = []
    for i in df['id']:
        basePath = os.path.sep.join([inputPath, "{}_*".format(i)])
        housePaths = sorted(list(glob.glob(basePath)))
        outputImage = np.zeros((size, size, 3), dtype="uint8")
        inputImages = []
        for housePath in housePaths:
            image = cv2.imread(housePath)
            image = cv2.resize(image, (size//2, size//2))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputImages.append(image)
        outputImage[0:64, 0:64] = inputImages[0]
        outputImage[0:64, 64:128] = inputImages[1]
        outputImage[64:128, 64:128] = inputImages[2]
        outputImage[64:128, 0:64] = inputImages[3]
        images.append(outputImage)

    return np.array(images) / 255.0

def make_mlp(num_cols):
    """
    演習:Dropoutを変更してみてください
    """
    model = models.Sequential()
    model.add(layers.Dense(units=256, input_shape = (num_cols,), 
                    kernel_initializer='he_normal',activation='relu'))    
    # model.add(Dropout(0.2))
    model.add(layers.Dense(units=128,  kernel_initializer='he_normal',activation='relu'))
    # model.add(Dropout(0.2))
    model.add(layers.Dense(units=64, kernel_initializer='he_normal', activation='relu'))
    model.add(layers.Dense(units=32, kernel_initializer='he_normal', activation='relu'))     
    # model.add(Dropout(0.2))
    # model.add(Dense(1, activation='linear'))
    # model.compile(loss='mape', optimizer='adam', metrics=['mape']) 
    return model

def vgg16(shape,is_finetune):
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(shape,shape,3))
    conv_base.trainable = True
    set_trainable = False
    if is_finetune:
        for layer in conv_base.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(units=256, activation='relu'))  
    model.add(layers.Dense(units=32, activation='relu'))
    return model

df = pd.read_csv("../input/4th-datarobot-ai-academy-deep-learning/train.csv")
num_cols = ['bedrooms', 'bathrooms', 'area']
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df = pd.concat([df,pd.get_dummies(df['zipcode'],prefix="zipcode")],axis=1)
df.shape
new_ids = np.random.choice(df['id'],size=3,replace=False)
img_matrix_list = []
title_list = []
for new_id in new_ids:
    for img in sorted(glob.glob('../input/4th-datarobot-ai-academy-deep-learning/images/train_images/{}_*.jpg'.format(new_id))):
        title_list.append(img.split("/")[-1])
        img_matrix_list.append(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
plot_multiple_img(img_matrix_list,title_list=title_list, ncols=4)
size = 128
inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/train_images/'
train_images = load_images(df,inputPath, size)
train_images.shape

train_x, valid_x, train_images_x, valid_images_x = train_test_split(df, train_images, test_size=0.2)

train_y = train_x['price'].values
valid_y = valid_x['price'].values

train_x = train_x.drop(columns=['id', 'zipcode', 'price']).values
valid_x = valid_x.drop(columns=['id', 'zipcode', 'price']).values
# display(train_images_x.shape, train_x.shape, train_y.shape)
# display(valid_images_x.shape, valid_x.shape, valid_y.shape)

mlp = make_mlp(train_x.shape[1])
cnn = vgg16(shape = size, is_finetune=True)

combinedInput = layers.concatenate([mlp.output, cnn.output])
x = layers.Dense(32, activation="relu")(combinedInput)
x = layers.Dense(1, activation="linear")(x)

model = models.Model(inputs=[mlp.input, cnn.input], outputs=x)
model.summary()
plot_model(model)

model.compile(loss='mape', optimizer="Adam", metrics=['mape']) 
filepath = "double_inputs_best_model.hdf5" 
es = EarlyStopping(patience=5, mode='min', verbose=1) 
checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1,  mode='min')
history = model.fit([train_x, train_images_x], train_y,
          validation_data=([valid_x, valid_images_x], valid_y),
          epochs=200,batch_size=8,
          callbacks=[es, checkpoint, reduce_lr_loss])

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'bo' ,label = 'training loss')
plt.plot(epochs, val_loss, 'b' , label= 'validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

epochs = np.argmin(history.history['val_loss'])
mlp = make_mlp(train_x.shape[1])
cnn = vgg16(shape = size, is_finetune=True)

combinedInput = layers.concatenate([mlp.output, cnn.output])
x = layers.Dense(32, activation="relu")(combinedInput)
x = layers.Dense(1, activation="linear")(x)

model = models.Model(inputs=[mlp.input, cnn.input], outputs=x)
model.compile(loss='mape', optimizer="Adam", metrics=['mape']) 
history = model.fit([train_x, train_images_x], train_y,epochs=epochs)
test = pd.read_csv("../input/4th-datarobot-ai-academy-deep-learning/test.csv")
scaler = MinMaxScaler()
test[num_cols] = scaler.fit_transform(test[num_cols])
test = pd.concat([test,pd.get_dummies(test['zipcode'],prefix="zipcode")],axis=1)
test.shape
unexist_col = [i for i in df.columns if i not in test.columns ]

del unexist_col[unexist_col.index('price')]

for col  in unexist_col:
    test[col] = 0
    
test = test.drop(columns = [i for i in test.columns if i not in  df.columns])
test.shape
inputPath = '../input/4th-datarobot-ai-academy-deep-learning/images/test_images/'
test_images = load_images(test,inputPath, size)
test_images.shape

test_x = test.drop(columns=['id', 'zipcode']).values
test_yhat = model.predict([test_x, test_images])

submit = test[['id']].copy()

submit['price'] = test_yhat
submit.to_csv("submission_sbihbd_20200822_2nd.csv", index=False)
