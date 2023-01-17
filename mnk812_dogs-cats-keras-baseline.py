import pandas as pd

import numpy as np

from PIL import Image

from tqdm import tqdm

import os

import cv2



from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization, Dense, Activation, GlobalAveragePooling2D

from keras.models import Model,Sequential

from keras.regularizers import l2

from keras.preprocessing.image import load_img,img_to_array

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint

from keras.layers import AveragePooling2D,Flatten,add,Input,MaxPooling2D,ZeroPadding2D
#制作训练集的图片id和标签

training_dir = '../input/train/'

testing_dir = '../input/test/'



train_files = os.listdir(training_dir)

test_files = os.listdir(testing_dir)



train_labels = []

for file in train_files:

    train_labels.append(file.split(".")[0])

    

df_train = pd.DataFrame({"id": train_files, "label": train_labels})

df_train.head()
df_test = pd.DataFrame({"id": test_files})

df_test["label"] = ["cat"]*(len(test_files))

df_test.head()
#制作keras数据生成器

classes = ['cat', 'dog']



def get_data(batch_size=32, target_size=(96,96), class_mode="categorical", training_dir=training_dir,

             testing_dir=testing_dir, classes=classes, df_train=df_train, df_test=df_test):

    

    train_datagen = ImageDataGenerator(horizontal_flip=True, shear_range=0.2,zoom_range=0.2,

        rescale=1.0/255,validation_split=0.25)

    test_datagen = ImageDataGenerator(rescale=1.0/255)

    

    train_generator = train_datagen.flow_from_dataframe(df_train, training_dir, x_col='id', y_col='label', 

        has_ext=True, target_size=target_size, classes = classes, class_mode=class_mode, 

        batch_size=batch_size, shuffle=True, seed=42,subset='training')

    

    validation_generator = train_datagen.flow_from_dataframe(df_train, training_dir, x_col='id', y_col='label', 

        has_ext=True, target_size=target_size, classes = classes, class_mode=class_mode, 

        batch_size=batch_size, shuffle=True, seed=42, subset='validation')



    test_generator = test_datagen.flow_from_dataframe(df_test, testing_dir, x_col='id', y_col='label', 

        has_ext=True, target_size=target_size, classes = classes, class_mode=class_mode, 

        batch_size=batch_size, shuffle=False)

    

    steps_per_epoch = len(train_generator)

    validation_steps = len(validation_generator)

    

    return train_generator, validation_generator, test_generator,  steps_per_epoch, validation_steps
#接下来定义网络结构，这里仿照resnet的结构进行定义。

#先定义卷积-BN结构

def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same'):

    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu')(x)

    x = BatchNormalization(axis=3)(x)

    return x



#定义resnet的残差结构，其中with_conv_shortcut参数是使用卷积防止通道不一致。

def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):

    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')

    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')

    if with_conv_shortcut:

        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)

        x = add([x, shortcut])

        return x

    else:

        x = add([x, inpt])

        return x

    

    #定义resnet网络结构

def Resnet():

    inpt = Input(shape=(299,299,3))

    x = ZeroPadding2D((3, 3))(inpt)

    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # (56,56,64)

    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))

    # (28,28,128)

    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)

    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))

    # (14,14,256)

    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)

    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))

    # (7,7,512)

    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)

    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))

    x = AveragePooling2D(pool_size=(7, 7))(x)

    x = Flatten()(x)

    x = Dense(2, activation='softmax')(x)

 

    model = Model(inpt,x)

    #model.summary()

    return model
#读取数据

batch_size = 32

target_size = (299, 299)

train_generator, validation_generator, test_generator, steps_per_epoch, validation_steps = get_data(batch_size=batch_size, target_size=target_size, classes=classes, df_test=df_test)

#建立模型

model = Resnet()

optimizer = Adam(0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'] )

checkpoint = ModelCheckpoint('model.hdf5', monitor='val_acc', save_best_only=True)

callbacks = [checkpoint]

#开始训练

history = model.fit_generator(

    train_generator,

    steps_per_epoch=steps_per_epoch,

    epochs=3,

    verbose=1,

    callbacks=callbacks,

    validation_data=validation_generator,

    validation_steps=validation_steps)
def generate_result(model, test_generator, nsteps=len(test_generator)):

    y_preds = model.predict_generator(test_generator, steps=nsteps, verbose=1) 

    return y_preds, y_preds[:,1]



y_preds_all, y_preds = generate_result(model, test_generator)       
df_test = pd.DataFrame({"id": test_generator.filenames, "label": y_preds})

df_test['id'] = df_test['id'].map(lambda x: x.split('.')[0])

df_test['id'] = df_test['id'].astype(int)

df_test = df_test.sort_values('id')

df_test.to_csv('submission.csv', index=False)

df_test.head()