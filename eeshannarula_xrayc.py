import numpy as np

import cv2

import os



from keras.models import Sequential, Model

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D, Activation

from keras.layers import GlobalMaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import Concatenate

from keras.models import Model

from keras.optimizers import Adam, SGD, RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
def getLabel(l):

    if (l == 'NORMAL'):

        return [1,0]

    elif (l == 'PNEUMONIA'):

        return [0,1]

    else:

        return None



def loadPaths(type):

    imgs = []

    labels = []

    for t in os.listdir(os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray',type)):

        if t != '.DS_Store':

            for i in os.listdir(os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray',type,t)):

                if i != '.DS_Store':

                   imgs.append(os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray',type,t,i))

                   if getLabel(t) != None:

                      labels.append(getLabel(t))





            # imgs.append(cv2.imread('image',-1))

    return imgs,labels



def loadData():

    data = []

    lbls = []

    cats = ['test','train','val']

    for i in cats:

        pth,labels = loadPaths(i)

        array = []

        for j in pth:

            if not '.DS_Store' in j:

               img = cv2.resize(cv2.imread(j,1),(224,224))

               array.append(img)

        data.append(np.array(array))

        lbls.append(np.array(labels))

        print(i + 'done')

    return data,lbls

def build_model():

    input_img = Input(shape=(224,224,3), name='ImageInput')

    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_1')(input_img)

    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_2')(x)

    x = MaxPooling2D((2,2), name='pool1')(x)



    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1')(x)

    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2')(x)

    x = MaxPooling2D((2,2), name='pool2')(x)



    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)

    x = BatchNormalization(name='bn1')(x)

    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)

    x = BatchNormalization(name='bn2')(x)

    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)

    x = MaxPooling2D((2,2), name='pool3')(x)



    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)

    x = BatchNormalization(name='bn3')(x)

    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)

    x = BatchNormalization(name='bn4')(x)

    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)

    x = MaxPooling2D((2,2), name='pool4')(x)



    x = Flatten(name='flatten')(x)

    x = Dense(1024, activation='relu', name='fc1')(x)

    x = Dropout(0.7, name='dropout1')(x)

    x = Dense(512, activation='relu', name='fc2')(x)

    x = Dropout(0.5, name='dropout2')(x)

    x = Dense(2, activation='softmax', name='fc3')(x)



    model = Model(inputs=input_img, outputs=x)

    return model
[testx,trainx,valx],[testy,trainy,valy] = loadData()
model =  build_model()



opt = Adam(lr=0.0001, decay=1e-5)

es = EarlyStopping(patience=5)

chkpt = ModelCheckpoint(filepath='output', save_best_only=True, save_weights_only=True)

model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)



batch_size = 16

nb_epochs = 5



nb_train_steps = trainx.shape[0]//batch_size

nb_val_steps = valx.shape[0]//batch_size

model.fit(trainx,trainy,epochs = nb_epochs,validation_data=(valx, valy),callbacks=[es, chkpt])

model.save('model.h5')
score, acc = model.evaluate(testx, testy)

print('Test score:', score)

print('Test accuracy:', acc)