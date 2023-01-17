# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/zindispotthemaskchallenge/train_labels.csv")

sub=pd.read_csv("/kaggle/input/zindispotthemaskchallenge/sample_sub.csv")
df.head()
import os

import matplotlib.pyplot as plt

data=os.listdir("/kaggle/input/zindispotthemaskchallenge/images/images")
data
import cv2



import random

sample=random.choice(data) #picking random sample from data list

img=cv2.imread("/kaggle/input/zindispotthemaskchallenge/images/images/"+sample)

plt.imshow(img,cmap="gray")
df["target"] = df["target"].replace({0: 'unmask', 1: 'mask'}) 

FAST_RUN = False

IMAGE_WIDTH=224

IMAGE_HEIGHT=224

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3 # RGB color
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization,GlobalMaxPooling2D

from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.applications import VGG16

from keras.models import Model
image_size = 224

input_shape = (image_size, image_size, 3)



batch_size = 16



pre_trained_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")

    

for layer in pre_trained_model.layers[:15]:

    layer.trainable = False



for layer in pre_trained_model.layers[15:]:

    layer.trainable = True

    

last_layer = pre_trained_model.get_layer('block5_pool')

last_output = last_layer.output

    



x = GlobalMaxPooling2D()(last_output)



x = Dense(512, activation='relu')(x)



x = Dropout(0.5)(x)



x = Dense(2, activation='softmax')(x)



model = Model(pre_trained_model.input, x)



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

              metrics=['accuracy'])



model.summary()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau



earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',patience=2,verbose=1,factor=0.5,min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]

from sklearn.model_selection import train_test_split



train_df,validate_df=train_test_split(df,test_size=0.2,random_state=42)

train_df = train_df.reset_index(drop='True')

validate_df = validate_df.reset_index(drop='True')

train_df.head()
train_df['target'].value_counts().plot.bar()
validate_df['target'].value_counts().plot.bar()
from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical
total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

batch_size=15
train_datagen=ImageDataGenerator(

                    rotation_range=15,

                    rescale=1./255,

                    shear_range=0.1,

                    zoom_range=0.2, # zoom range (1-0.2 to 1+0.2)

                    horizontal_flip=True,

                    width_shift_range=0.1,

                    height_shift_range=0.1

                 )

train_generator=train_datagen.flow_from_dataframe(

                    dataframe=train_df, 

                    directory="/kaggle/input/zindispotthemaskchallenge/images/images/", 

                    x_col="image",

                    y_col="target",

                    target_size=(image_size,image_size),

                    class_mode='categorical',

                    batch_size=15

                )
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    directory="/kaggle/input/zindispotthemaskchallenge/images/images/", 

    x_col="image",

    y_col="target",

    target_size=(image_size,image_size),

    class_mode='categorical',

    batch_size=batch_size

)
example_df = train_df.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(

    example_df, 

    directory="/kaggle/input/zindispotthemaskchallenge/images/images/", 

    x_col="image",

    y_col="target",

    target_size=(image_size,image_size),

    class_mode='categorical',

)
plt.figure(figsize=(12, 12))

for i in range(0, 15):

    plt.subplot(5, 3, i+1)

    for X_batch, Y_batch in example_generator:

        image = X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
epochs=100 

history = model.fit_generator(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
target=[]



for i in data:

    flag=0

    for j in df["image"]:

        if(i==j):

            flag=1

            break;

        else:

            continue

    if(flag==0):    

        target.append(i)
target
test = pd.DataFrame({

    'image': target,

    'target':"unmask"

})

test.head()
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test, 

    directory="/kaggle/input/zindispotthemaskchallenge/images/images/", 

    x_col="image",

    y_col="target",

    target_size=(image_size,image_size),

    class_mode='categorical',

    batch_size=15,

    shuffle=False

)

nb_samples = test.shape[0]

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
test["target"]=predict



test.to_csv("sub1.csv",index=False)

from IPython.display import FileLink

FileLink(r'sub1.csv')
