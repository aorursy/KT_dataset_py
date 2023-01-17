# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

from tensorflow import keras

import os

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,BatchNormalization,Dropout,GlobalMaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras import regularizers, optimizers
train = pd.read_csv("../input/imet-2020-fgvc7/train.csv")

test = pd.read_csv("../input/imet-2020-fgvc7/sample_submission.csv")

train["id"] = train["id"] + '.png'

test["id"] = test["id"] + ".png"

train["attribute_ids"] = train["attribute_ids"].apply(lambda x:x.split())

# valid = train.iloc[113701:]

# train = train.iloc[:113701]

# print(train.shape)

# print(valid.shape)

train.head()
trainGen = ImageDataGenerator(preprocessing_function=preprocess_input,width_shift_range=0.1,

                             height_shift_range=0.1,

                             zoom_range=0.2,rescale=1/255.0)

train_data = trainGen.flow_from_dataframe(dataframe=train,directory="../input/imet-2020-fgvc7/train/",

                                          x_col="id",y_col="attribute_ids",

                                          batch_size=128,seed=4,

                                          shuffle=True,target_size=(224,224))

# validGen = ImageDataGenerator(preprocessing_function=preprocess_input,width_shift_range=0.1,

#                              height_shift_range=0.1,

#                              zoom_range=0.2,rescale=1/255.0)

# valid_data = validGen.flow_from_dataframe(dataframe=valid,directory="../input/imet-2020-fgvc7/train/",

#                                           x_col="id",y_col="attribute_ids",

#                                           batch_size=128,seed=4,

#                                           shuffle=True,target_size=(224,224))

testGen = ImageDataGenerator(preprocessing_function=preprocess_input,rescale=1/255.0)

test_data = testGen.flow_from_dataframe(dataframe=test,directory="../input/imet-2020-fgvc7/test/",

                                          x_col="id",batch_size=1,seed=4,shuffle=False,

                                          class_mode=None,target_size=(224,224))
vgg = VGG16(weights=None, include_top=True,pooling='max')

x = vgg.layers[-13].output

# x = GlobalMaxPooling2D()(x)

x = Flatten()(x)

x = Dense(units=1024,activation='relu')(x)

# x = Dropout(rate=0.5)(x) 

# x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)

# x = Dense(units=256,activation='relu')(x)

# x = Dropout(rate=0.5)(x)

# x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)

# x = Dense(units=16,activation='relu')(x)

# x = Dropout(rate=0.5)(x)

# x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)

x = Dense(units=3471, activation="sigmoid")(x)

model = Model(inputs=vgg.input,outputs=x)
from tensorflow.keras.optimizers import Adam

opt = Adam(0.00001)

model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
def build_lrfn(lr_start=0.00001, lr_max=0.000075, 

               lr_min=0.000001, lr_rampup_epochs=20, 

               lr_sustain_epochs=0, lr_exp_decay=.8):

    

    def lrfn(epoch):

        if epoch < lr_rampup_epochs:

            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

        elif epoch < lr_rampup_epochs + lr_sustain_epochs:

            lr = lr_max

        else:

            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

        return lr

    

    return lrfn



lrfn = build_lrfn()

lr_schedule = LearningRateScheduler(lrfn, verbose=1)
STEP_SIZE_TRAIN=train_data.n//train_data.batch_size



hist=model.fit(train_data,steps_per_epoch=STEP_SIZE_TRAIN,epochs=1)#,callbacks=[lr_schedule]
train_data[0][0].shape
model.save("model.h5")
from tensorflow.keras.models import load_model

model=load_model("model.h5")
result = model.predict(test_data,verbose=1)
pred_bool = (result >0.2)

predictions=[]

labels = train_data.class_indices

labels = dict((v,k) for k,v in labels.items())

for row in pred_bool:

    l=[]

    

    for index,cls in enumerate(row):

        if cls:

            l.append(labels[index])

    predictions.append(" ".join(l))

    

filenames=test_data.filenames



results = pd.DataFrame({"id":filenames,"attribute_ids":predictions})

results["id"] = results["id"].apply(lambda x:x.split(".")[0])
results.to_csv("submission.csv",index=False)