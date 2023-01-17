# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
path_train="../input/train"
path_test="../input/test"
train_data=next(os.walk(path_train+"/images"))[2]
test_data=next(os.walk(path_test+"/images"))[2]
len(train_data),len(test_data)
os.listdir(path_train)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
img=np.array(Image.open(path_train+"/masks/"+train_data[0]).convert('L'))
img.shape
img
X=[]
Y=[]
for i in range(len(train_data)):
    img=Image.open(path_train+"/images/"+train_data[i]).convert('L')
    img=img.resize((128,128))
    X.append(np.array(img))
    img1=Image.open(path_train+"/masks/"+train_data[i])
    img1=img1.resize((128,128))
    Y.append(np.array(img1))
X=np.array(X)
Y=np.array(Y)
X.shape,Y.shape
    
X=np.reshape(X,(len(X),128,128,1))
Y=np.reshape(Y,(len(Y),128,128,1))
X.shape,Y.shape
Y=np.asarray(Y).astype('bool')
# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
from keras.models import Model, load_model
from keras.layers import Input,Dropout,add,BatchNormalization
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam,SGD
from tqdm import tqdm_notebook, tnrange
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
inputs = Input((128,128,1))
s =Lambda(lambda x:x/255)(inputs)

a = Conv2D(16, (5, 5), activation='relu', padding='same')(s)
c1 = BatchNormalization()(a)
c1 = Conv2D(16, (3, 3), activation='relu', padding='same') (c1)
c1 = BatchNormalization()(c1)
c1 = Conv2D(16, (3, 3), activation='relu', padding='same') (c1)
c1 = BatchNormalization()(c1)
c1 = Conv2D(16, (3, 3), activation='relu', padding='same') (c1)
c1 = BatchNormalization()(c1)
c1 = add([a,c1])
#c1 = Dropout(0.25)(c1)
p1 = MaxPooling2D((2, 2)) (c1)


c2 = Conv2D(32, (3, 3), activation='relu', padding='same') (p1)
c2 = BatchNormalization()(c2)
temp = c2
c2 = Conv2D(32, (3, 3), activation='relu', padding='same') (c2)
c2 = BatchNormalization()(c2)
c2 = Conv2D(32, (3, 3), activation='relu', padding='same') (c2)
c2 = BatchNormalization()(c2)
c2 = add([temp,c2])
#c2 = Dropout(0.3)(c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(48, (3, 3), activation='relu', padding='same') (p2)
c3 = BatchNormalization()(c3)
temp = c3
c3 = Conv2D(48, (3, 3), activation='relu', padding='same') (c3)
c3 = BatchNormalization()(c3)
c3 = Conv2D(48, (3, 3), activation='relu', padding='same') (c3)
c3 = BatchNormalization()(c3)
c3 = add([temp,c3])
#c3 = Dropout(0.35)(c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = BatchNormalization()(c4)
temp = c4
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
c4 = BatchNormalization()(c4)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
c4 = BatchNormalization()(c4)
c4 = add([temp,c4])
#c4 = Dropout(0.4)(c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(96, (3, 3), activation='relu', padding='same') (p4)
c5 = BatchNormalization()(c5)
temp = c5
c5 = Conv2D(96, (3, 3), activation='relu', padding='same') (c5)
c5 = BatchNormalization()(c5)
c5 = Conv2D(96, (3, 3), activation='relu', padding='same') (c5)
c5 = BatchNormalization()(c5)
c5 = add([temp,c5])
#c5 = Dropout(0.45)(c5)
p5 = MaxPooling2D(pool_size=(2, 2)) (c5)

c6 = Conv2D(128, (3, 3), activation='relu', padding='same') (p5)
c6 = BatchNormalization()(c6)
temp = c6
c6 = Conv2D(128, (3, 3), activation='relu', padding='same') (c6)
c6 = BatchNormalization()(c6)
c6 = Conv2D(128, (3, 3), activation='relu', padding='same') (c6)
c6 = BatchNormalization()(c6)
c6 = add([temp,c6])
#c6 = Dropout(0.5)(c6)
p6 = MaxPooling2D(pool_size=(2, 2)) (c6)

l = Conv2D(196, (3, 3), activation='relu', padding='same') (p6)
temp = l
l = BatchNormalization()(l)
l = Conv2D(196, (3, 3), activation='relu', padding='same') (l)
l = BatchNormalization()(l)
l = Conv2D(196, (3, 3), activation='relu', padding='same') (l)
l = BatchNormalization()(l)
l = add([temp,l])

#l = Dropout(0.5)(l)
l = Conv2D(256, (3, 3), activation='relu', padding='same') (l)
l = BatchNormalization()(l)
temp=l
l = Conv2D(256, (3, 3), activation='relu', padding='same') (l)
l = BatchNormalization()(l)
l = Conv2D(256, (3, 3), activation='relu', padding='same') (l)
l = BatchNormalization()(l)
l = add([temp,l])

#l = Dropout(0.5)(l)
l = Conv2D(196, (3, 3), activation='relu', padding='same') (l)
l = BatchNormalization()(l)
temp=l
l = Conv2D(196, (3, 3), activation='relu', padding='same') (l)
l = BatchNormalization()(l)
l = Conv2D(196, (3, 3), activation='relu', padding='same') (l)
l = BatchNormalization()(l)
l = add([temp,l])

#l = Dropout(0.5)(l)

u6_1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (l)
u6_1 = concatenate([u6_1, c6])
u6_1 = Dropout(0.45)(u6_1)
l = Conv2D(128, (3, 3), activation='relu', padding='same') (u6_1)
l = Conv2D(128, (3, 3), activation='relu', padding='same') (l)
l = Conv2D(128, (3, 3), activation='relu', padding='same') (l)


u7 = Conv2DTranspose(96, (2, 2), strides=(2, 2), padding='same') (l)
u7 = concatenate([u7, c5])
u7 = Dropout(0.4)(u7)
c7 = Conv2D(96, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(96, (3, 3), activation='relu', padding='same') (c7)
c7 = Conv2D(96, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c4])
u8 = Dropout(0.35)(u8)
c8 = Conv2D(64, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(64, (3, 3), activation='relu', padding='same') (c8)
c8 = Conv2D(64, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(48, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c3])
u9 = Dropout(0.3)(u9)
c9 = Conv2D(48, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(48, (3, 3), activation='relu', padding='same') (c9)
c9 = Conv2D(48, (3, 3), activation='relu', padding='same') (c9)

u10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c9)
u10 = concatenate([u10, c2])
u10 = Dropout(0.25)(u10)
c10 = Conv2D(32, (3, 3), activation='relu', padding='same') (u10)
c10= Conv2D(32, (3, 3), activation='relu', padding='same') (c10)
c10= Conv2D(32, (3, 3), activation='relu', padding='same') (c10)

u11 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c10)
u11 = concatenate([u11, c1], axis=3)
u11 = Dropout(0.25)(u11)
c11 = Conv2D(16, (3, 3), activation='relu', padding='same') (u11)
c11 = Conv2D(16, (3, 3), activation='relu', padding='same') (c11)
c11 = Conv2D(16, (3, 3), activation='relu', padding='same') (c11)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c11)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[mean_iou,'accuracy'])
model.summary()

from sklearn.model_selection import train_test_split as tts
x_train,x_val,y_train,y_val=tts(X,Y,test_size=0.1,random_state=42)
model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=32, epochs=20)
K.set_value(model.optimizer.lr,1e-5)
model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=32, epochs=20)
model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=32, epochs=10)
K.set_value(model.optimizer.lr,1e-6)
model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=32, epochs=20)
X_test=[]
for i in range(len(test_data)):
    img=Image.open(path_test+"/images/"+test_data[i]).convert('L')
    img=img.resize((128,128))
    X_test.append(np.array(img))
    
X_test=np.array(X_test)
X_test=np.reshape(X_test,(len(X_test),128,128,1))
X_test.shape
preds_test = model.predict(X_test, verbose=1)
pred = (preds_test > 0.5).astype(np.uint8)
pred[0].shape
p=np.reshape(Y[0],(128,128))
tmp = np.asarray(p).astype(np.float32)
plt.imshow(tmp)
plt.show()
preds_test_up = []
for i in range(len(preds_test)):
    p=np.reshape(preds_test[i],(128,128))
    img=Image.fromarray(p)
    img=img.resize((101,101))
    preds_test_up.append(np.array(img))
preds_test_up=np.array(preds_test_up)
preds_test_up.shape
ix = np.random.randint(0, len(X_test))
p=np.reshape(X_test[ix],(128,128))
plt.imshow(np.dstack((p,p,p)))
plt.show()
q=np.reshape(preds_test[ix],(128,128))
tmp = np.asarray(q).astype(np.float32)
plt.imshow(np.dstack((q,q,q)))
plt.show()
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

pred_dict = {fn[:-4]:RLenc(np.round(preds_test_up[i])) for i,fn in tqdm_notebook(enumerate(test_data))}
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')









