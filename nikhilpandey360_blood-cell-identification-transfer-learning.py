

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

import os

import tqdm

print(os.listdir("../input/dataset2-master/dataset2-master/images/TRAIN/"))



img_path = "../input/dataset2-master/dataset2-master/images/TRAIN/"



# Any results you write to the current directory are saved as output.
class_ls = os.listdir(img_path)

img_ls = []

label_ls = []

dim = (128,128)

for i,j in enumerate(class_ls):

    sub_path = os.path.join(img_path,j)

    

    for img in tqdm.tqdm(os.listdir(sub_path)):

        if img.split(".")[1]=="jpeg":

            im = cv2.resize(cv2.imread(os.path.join(sub_path,img))/255,dim).astype(np.float32)

            img_ls.append(im)

            label_ls.append(int(i))

        

        

        
images = np.array(img_ls)

labels = np.array(label_ls)





plt.figure(figsize=(20,15))



for i in range(0,9,3):

    plt.subplot(3,3,i+1)

    plt.imshow(img_ls[i])

    plt.xlabel(label_ls[i])

    

    

    plt.subplot(3,3,i+2)

    plt.imshow(img_ls[i+1])

    plt.xlabel(label_ls[i+1])

    

    

    plt.subplot(3,3,i+3)

    plt.imshow(img_ls[i+2])

    plt.xlabel(label_ls[i+2])

    
import keras

y = keras.utils.to_categorical(labels, num_classes=None, dtype='float32')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images,y,test_size=0.1)
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path="{}_weights.best.hdf5".format('tb_detector')



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)





reduceLROnPlat = ReduceLROnPlateau(monitor='val_acc', factor=0.8, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)



early = EarlyStopping(monitor="val_acc", 

                      mode="min", 

                      patience=15)



callbacks_list = [checkpoint, early, reduceLROnPlat]
from keras.applications.vgg16 import VGG16

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda

from keras.models import Model

in_lay = Input(X_train.shape[1:])

base_pretrained_model = VGG16(input_shape =  X_train.shape[1:], 

                              include_top = False, weights = 'imagenet')

base_pretrained_model.trainable = False

pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]

pt_features = base_pretrained_model(in_lay)

from keras.layers import BatchNormalization

bn_features = BatchNormalization()(pt_features)



# here we do an attention mechanism to turn pixels in the GAP on an off



attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(bn_features)

attn_layer = Conv2D(20, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)

attn_layer = Conv2D(1, kernel_size = (1,1), padding = 'valid', activation = 'sigmoid')(attn_layer)



# fan it out to all of the channels

up_c2_w = np.ones((1, 1, 1, pt_depth))

up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same',  activation = 'linear', use_bias = False, weights = [up_c2_w])

up_c2.trainable = False

attn_layer = up_c2(attn_layer)



mask_features = multiply([attn_layer, bn_features])

gap_features = GlobalAveragePooling2D()(mask_features)

gap_mask = GlobalAveragePooling2D()(attn_layer)

# to account for missing values from the attention model

gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])

gap_dr = Dropout(0.25)(gap)

dr_steps = Dropout(0.65)(Dense(512, activation = 'elu')(gap_dr))

out_layer = Dense(4, activation = 'softmax')(dr_steps)

tb_model = Model(inputs = [in_lay], outputs = [out_layer])



tb_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',

                           metrics = ['accuracy'])
loss_history = tb_model.fit(X_train,y_train,

                                      validation_data =(X_test,y_test), 

                                  epochs = 25, 

                                  callbacks = callbacks_list)



# load the best version of the model

tb_model.load_weights(weight_path)

tb_model.save('full_tb_model.h5')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))

ax1.plot(loss_history.history['loss'], '-', label = 'Loss')

ax1.plot(loss_history.history['val_loss'], '-', label = 'Validation Loss')

ax1.legend()



ax2.plot(100*np.array(loss_history.history['acc']), '-', 

         label = 'Accuracy')

ax2.plot(100*np.array(loss_history.history['val_acc']), '-',

         label = 'Validation Accuracy')

ax2.legend()
pred_Y = tb_model.predict(X_test , verbose = True)
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

dic = {}

for i in tqdm.tqdm(range(10000)):

    dic[0.0001*i] = round(accuracy_score(y_test,pred_Y>(0.0001*i)),3)
accuracy_score(y_test,pred_Y>max(dic, key=dic.get))