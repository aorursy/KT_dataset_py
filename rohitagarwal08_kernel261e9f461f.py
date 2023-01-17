# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import regularizers, optimizers
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


traindf=pd.read_csv('../input/siim-isic-melanoma-classification/train.csv',dtype=str)
def append_ext(fn):
    return fn+".jpg"

traindf["image_name"]=traindf["image_name"].apply(append_ext)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(traindf['image_name'], traindf['target'], test_size=0.15, random_state=42)
print(len(X_test))
bs = 32
datagen=ImageDataGenerator(rescale=1./255,validation_split=0.15)
train_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="../input/siim128x128-mix/train/",
x_col="image_name",
y_col="target",
subset="training",
color_mode='rgb',
batch_size=bs,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(128,128))

val_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="../input/siim128x128-mix/train/",
x_col="image_name",
y_col="target",
subset="validation",
color_mode='rgb',
batch_size=bs,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(128,128))
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization, UpSampling2D, concatenate
from tensorflow.keras.models import Model



def unet(pretrained_weights = None ,input_size = (128,128,3)):
    i = Input(input_size)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(i)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    #drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)#conv4

    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    #drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))#conv5
    merge6 = concatenate([conv4,up6], axis = 3)#drop4
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    pool9 = MaxPooling2D(pool_size=(2, 2))(conv9)

    conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
    conv10 = BatchNormalization()(conv10)
    pool10 = MaxPooling2D(pool_size=(2, 2))(conv10)

    merge11 = concatenate([conv3,pool10], axis = 3)
    conv11 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge11)
    conv11 = BatchNormalization()(conv11)
    conv11 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    conv11 = BatchNormalization()(conv11)
    pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)

    conv12 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv12)
    conv12 = BatchNormalization()(conv12)
    #drop4 = Dropout(0.5)(conv4)
    pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)#conv4

    conv13 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv13)
    conv13 = BatchNormalization()(conv13)
    #drop5 = Dropout(0.5)(conv5)



    up14 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv13))#conv5
    merge14 = concatenate([conv12,up14], axis = 3)#drop4
    conv14 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge14)
    conv14 = BatchNormalization()(conv14)
    conv14 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv14)
    conv14 = BatchNormalization()(conv14)

    up15 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv14))
    merge15 = concatenate([conv11,up15], axis = 3)
    conv15 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge15)
    conv15 = BatchNormalization()(conv15)
    conv15 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv15)
    conv15 = BatchNormalization()(conv15)
    conv15 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv15)
    conv15 = BatchNormalization()(conv15)

    up16 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv15))
    merge16 = concatenate([conv10,up16], axis = 3)
    conv16 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge16)
    conv16 = BatchNormalization()(conv16)
    conv16 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv16)
    conv16 = BatchNormalization()(conv16)
    conv16 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv16)
    conv16 = BatchNormalization()(conv16)

    up17 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv16))
    merge17 = concatenate([conv9,up17], axis = 3)
    conv17 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge17)
    conv17 = BatchNormalization()(conv17)
    conv17 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv17)
    conv17 = BatchNormalization()(conv17)

    merge18= concatenate([conv1,conv17], axis = 3)
    conv18 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge18)
    conv18 = BatchNormalization()(conv18)
    conv18 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge18)
    #conv18 = BatchNormalization()(conv18)
    #conv18 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv18)
    #conv18 = BatchNormalization()(conv18)
    #conv18 = Conv2D(1, 1, activation = 'sigmoid')(conv18)

    x = Flatten()(conv18) 
    x = Dropout(0.2)(x) 
    x = Dense(1024, activation='relu')(x) 
    x = Dropout(0.2)(x) 
    x = Dense(2, activation='softmax')(x)
    
    
    model = Model(i,x)

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
model=unet()
#spe=26501 // 32
spe=len(train_generator.filenames) // bs
vs=len(val_generator.filenames) // bs
model.fit_generator(train_generator,steps_per_epoch=spe,epochs=10,validation_data=val_generator, validation_steps=vs)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        directory="../input/siim128x128-mix/",
        classes=["test"],
        target_size=(128, 128),
        batch_size=1,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False)
predict = model.predict_generator(test_generator,steps = 10982)
predict
test_generator.reset()
filenames=test_generator.filenames
f=[]
for p in filenames:
    p=p.replace('test/','')
    p=p.replace('.jpg','')
    f.append(p)
pd.DataFrame(
    {
     'image_name': f, 
     'target': (1-predict[:,0])
    }
).to_csv('submission.csv', index=False)    
df = pd.read_csv('submission.csv')
df.head()
