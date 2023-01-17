import os

os.chdir('../input')
get_ipython().system_raw("../input/train.zip")

get_ipython().system_raw("../input/test.zip")
os.listdir()
import numpy as np

import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm
mapping_label = {0:'cat',1:'dog'}
def img_label(img):

    word_label=img.split('.')[0]

    if word_label=='cat':

        return 0

    elif word_label=='dog':

        return 1
def create_train_data(path):

    X,Y=[],[]

    my_dir = os.listdir(path)

    for img in tqdm(my_dir):

        label=img_label(img)

        img_data = cv2.imread('./'+ path +'/'+img)

        interpolate = cv2.INTER_AREA

        if(img_data.shape[0]<80 or img_data.shape[1]<80):

            interpolate = cv2.INTER_LINEAR

        img_data = cv2.resize(img_data, (80, 80),interpolation=interpolate)

        X.append(img_data)

        Y.append(label)

    return X,Y
X_train,Y_train=create_train_data('train')
X_train  = np.asarray(X_train)

Y_train = np.asarray(Y_train)
def create_test_data(path):

    X=[]

    my_dir = sorted(list(map(lambda x:int(x[:-4]),os.listdir(path))))

    for img in tqdm(my_dir):

        img_data = cv2.imread('./'+ path +'/'+ str(img) + '.jpg')

        interpolate = cv2.INTER_AREA

        if(img_data.shape[0]<80 or img_data.shape[1]<80):

            interpolate = cv2.INTER_LINEAR

        img_data = cv2.resize(img_data, (80, 80),interpolation=interpolate)

        X.append(img_data)

    return X
X_test = create_test_data('test')
X_test = np.asarray(X_test)
X_test.shape
plt.imshow(X_train[100])

plt.show()

plt.imshow(X_test[150])

plt.show()
from keras.utils import to_categorical

Y_train = to_categorical(Y_train)
X_train.shape,Y_train.shape
from sklearn.utils.class_weight import compute_class_weight

weights = compute_class_weight('balanced',[0,1],np.argmax(Y_train,axis=1))

print(weights)
from keras.utils import normalize

X_train = normalize(X_train,axis=1)

X_test = normalize(X_test,axis=1)
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

X_train,Y_train = shuffle(X_train,Y_train,random_state = 2)

X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=0.2,random_state = 2)
X_train.shape,Y_train.shape,X_val.shape,Y_val.shape
from keras.models import Sequential,Model

from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, regularizers, BatchNormalization, Input

import keras.optimizers as optimizers

import keras.callbacks as callbacks

from sklearn import metrics
input_= Input((80,80,3))

conv1  = Conv2D(64,3,padding='same',activation='relu')(input_)

pool1 = MaxPooling2D()(conv1)

norm1 = BatchNormalization()(pool1)

conv2  = Conv2D(64,3,padding='same',activation='relu')(norm1)

pool2 = MaxPooling2D()(conv2)

norm2 = BatchNormalization()(pool2)

conv3  = Conv2D(128,3,padding='same',activation='relu')(norm2)

pool3 = MaxPooling2D()(conv3)

norm3 = BatchNormalization()(pool3)

conv4  = Conv2D(256,3,padding='same',activation='relu')(norm3)

conv5  = Conv2D(256,3,padding='same',activation='relu')(conv4)

pool6 = MaxPooling2D()(conv5)

flat7 = Flatten()(pool6)

drop8 = Dropout(0.05)(flat7)

dense9 = Dense(128,activation='relu')(drop8)

dense10 = Dense(64,activation='relu')(dense9)

dense11 = Dense(2,activation='sigmoid')(dense10)



model = Model(inputs=input_,outputs=dense11)

model.summary()
sgd = optimizers.SGD(lr = 0.001 ,momentum = 0.9, nesterov = False)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

lr_reduce = callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5)
history = model.fit(X_train,Y_train,epochs=25,batch_size=250,validation_data=(X_val,Y_val),callbacks=[lr_reduce])
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
pred = model.predict(X_val)

Y_pred = np.argmax(pred,axis=1)

print(metrics.accuracy_score(np.argmax(Y_val,axis=1),Y_pred))

print(metrics.f1_score(np.argmax(Y_val,axis=1),Y_pred))

print(metrics.confusion_matrix(np.argmax(Y_val,axis=1),Y_pred))

print(metrics.log_loss(np.argmax(Y_val,axis=1),Y_pred))
pre = model.predict(X_test)   

Y_pre = np.argmax(pre,axis=1) 
for i in range(55):

    plt.imshow(X_test[i])

    print(mapping_label[Y_pre[i]],pre[i],i)

    plt.show()