

import os

"""for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))"""



# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from keras.regularizers import l2

from keras.initializers import he_normal, random_normal

from keras.layers import Input

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from PIL import Image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from keras.models import Sequential,Model

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint

from keras.optimizers import RMSprop

from keras.layers import Concatenate, SpatialDropout2D

from keras.initializers import glorot_uniform
df = pd.read_csv('/kaggle/input/nnfl-cnn-lab2/upload/train_set.csv')
df.head()
df['label'] = df['label'].astype(str)

df.dtypes
df['label'].value_counts().plot.bar()
filenames = os.listdir('/kaggle/input/nnfl-cnn-lab2/upload/train_images/train_images/')

sample = random.choice(filenames)

image = load_img('/kaggle/input/nnfl-cnn-lab2/upload/train_images/train_images/'+sample)

plt.imshow(image)

print(image.size)
IMAGE_HEIGHT = 150

IMAGE_WIDTH = 150

IMAGE_CHANNELS = 3

IMAGE_SIZE = (IMAGE_HEIGHT,IMAGE_WIDTH)

batch_size = 15

num_filter = 48

wt_decay = 0.001

dropout_rate = 0.2

compression = 0.5

l = 12

num_classes = 6
def add_denseblock(input, num_filter = 12, dropout_rate = 0.2):

    global compression

    temp = input

    for _ in range(l):

        BatchNorm = BatchNormalization()(temp)

        relu = Activation('relu')(BatchNorm)

        #Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same', kernel_initializer=(random_normal(stddev=np.sqrt(2.0/(9*int(num_filter*compression))))))(relu)

        Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same',kernel_regularizer=l2(wt_decay) , kernel_initializer=(random_normal(stddev=np.sqrt(2.0/(9*int(num_filter))))))(relu)

        if dropout_rate>0:

            Conv2D_3_3 = SpatialDropout2D(dropout_rate)(Conv2D_3_3)

        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])

        

        temp = concat

        

    return temp

def add_transition(input, num_filter = 12, dropout_rate = 0.2):

    global compression

    BatchNorm = BatchNormalization()(input)

    relu = Activation('relu')(BatchNorm)

    Conv2D_BottleNeck = Conv2D(int(int(input.shape[-1])*compression), (1,1), use_bias=False ,padding='same',kernel_regularizer=l2(wt_decay) , kernel_initializer=(random_normal(stddev=np.sqrt(2.0/(9*int(num_filter*compression))))))(relu)

    #Conv2D_BottleNeck = Conv2D(int(num_filter*compression), (1,1), use_bias=False ,padding='same',kernel_initializer=(random_normal(stddev=np.sqrt(2.0/(9*int(num_filter*compression))))))(relu)

    if dropout_rate>0:

      Conv2D_BottleNeck = SpatialDropout2D(dropout_rate)(Conv2D_BottleNeck)

    avg = AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)

    

    return avg



def output_layer(input):

    global compression

    BatchNorm = BatchNormalization()(input)

    relu = Activation('relu')(BatchNorm)

    AvgPooling = AveragePooling2D(pool_size=(2,2))(relu)

    flat = Flatten()(AvgPooling)

    output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(wt_decay))(flat)

    #output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(flat)

    

    return output

input = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS,))

First_Conv2D = Conv2D(int(num_filter), (3,3), use_bias=False , padding='same', kernel_regularizer=l2(wt_decay), kernel_initializer=(random_normal(stddev=np.sqrt(2.0/(9*num_filter)))))(input)



First_Block = add_denseblock(First_Conv2D, num_filter, dropout_rate)

First_Transition = add_transition(First_Block, num_filter, dropout_rate)



#First_Transition = merge([First_Transition,First_Conv2D], mode='concat', concat_axis=-1)



Second_Block = add_denseblock(First_Transition, num_filter, dropout_rate)

Second_Transition = add_transition(Second_Block, num_filter, dropout_rate)



#Second_Transition = Concatenate(axis=-1)([Second_Transition,First_Transition,First_Conv2D])



Third_Block = add_denseblock(Second_Transition, num_filter, dropout_rate)

Third_Transition = add_transition(Third_Block, num_filter, dropout_rate)



#Third_Transition = Concatenate(axis=-1)([Third_Transition,Second_Transition,First_Transition,First_Conv2D])



Last_Block = add_denseblock(Third_Transition,  num_filter, dropout_rate)

output = output_layer(Last_Block)

model = Model(inputs=[input], outputs=[output])

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



model.summary()
earlystop = EarlyStopping(patience=60)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

Checkpoint = ModelCheckpoint('temp.h5', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

callbacks = [earlystop, learning_rate_reduction,Checkpoint]
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)
train_df['label'].value_counts().plot.bar()
validate_df['label'].value_counts().plot.bar()
total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

print(total_train, total_validate)
train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)



train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    "/kaggle/input/nnfl-cnn-lab2/upload/train_images/train_images/", 

    x_col='image_name',

    y_col='label',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    "/kaggle/input/nnfl-cnn-lab2/upload/train_images/train_images/", 

    x_col='image_name',

    y_col='label',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
epochs=50

history = model.fit_generator(

    train_generator, 

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
model.load_weights('temp.h5')
model.save_weights("model.h5")
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper left')

plt.show()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper left')

plt.show()
validation_generator = validation_datagen.flow_from_dataframe(

    pd.DataFrame(validate_df), 

    "/kaggle/input/nnfl-cnn-lab2/upload/train_images/train_images/", 

    x_col='image_name',

    y_col='label',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size,

    shuffle = False

)

y_pred = [np.argmax(a) for a in (model.predict_generator(validation_generator))]
f1_score(y_pred,validate_df['label'].astype(int), average = 'micro')
confusion_matrix(y_pred,validate_df['label'].astype(int))
test_df = pd.read_csv('/kaggle/input/nnfl-cnn-lab2/upload/sample_submission.csv')
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(

    test_df, 

    "/kaggle/input/nnfl-cnn-lab2/upload/test_images/test_images/", 

    x_col='image_name',

    y_col=None,

    target_size=IMAGE_SIZE,

    class_mode=None,

    batch_size=batch_size,

    shuffle = False

)

nb_samples = test_df.shape[0]
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
test_df['label'] = np.argmax(predict, axis = -1)
test_df.head()
test_df['label'].value_counts().plot.bar()
submission_df = test_df.copy()

submission_df['image_name'] = submission_df['image_name']

submission_df['label'] = submission_df['label']

submission_df.image_name = submission_df.image_name.apply(str)

#submission_df.drop(['filename', 'category'], axis=1, inplace=True)

print(submission_df.head())

submission_df.to_csv('submission.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html) 

create_download_link(submission_df) 