%reload_ext autoreload

%autoreload 2

%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input, Flatten

from keras.callbacks import EarlyStopping, ModelCheckpoint, History

from keras.applications.xception import Xception

from keras.applications.vgg16 import VGG16



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_dir = "../input/dogs-vs-cats-redux-kernels-edition/train/train"

filenames = os.listdir(data_dir)

dog_files = [f for f in filenames if 'dog' in f]

cat_files = [f for f in filenames if 'cat' in f]



df = pd.DataFrame({

    'filename': dog_files + cat_files,

    'label': ['dog'] * len(dog_files) + ['cat'] * len(cat_files)

})

df = df.sample(frac=1).reset_index(drop=True)

df.head()
for i in range(5):

    img = mpimg.imread(data_dir + '/' + df.iloc[i]['filename'])

    plt.imshow(img)

    plt.show()
# Constants

input_shape = (128,128,3)
def get_model():

    X_input = Input(shape=input_shape)

    X = Conv2D(128, 

               kernel_size=(3,3), 

               padding='same', 

               activation='relu')(X_input)

    X = MaxPooling2D(pool_size=(3,3))(X)

    X = Dropout(0.25)(X)



    X = Conv2D(256, 

               kernel_size=(3,3), 

               padding='same',

               activation='relu')(X)

    X = MaxPooling2D(pool_size=(3,3))(X)

    X = Dropout(0.25)(X)

    

    X = Conv2D(256, 

               kernel_size=(3,3), 

               padding='same',

               activation='relu')(X)

    X = MaxPooling2D(pool_size=(3,3))(X)

    X = Dropout(0.25)(X)

    

    X = Flatten()(X)

    X = Dense(128, activation='relu')(X)

#     X = Dropout(0.5)(X)

    out = Dense(1, activation='sigmoid')(X)

    return Model(X_input, [out])
model = get_model()
train_size = int(df.shape[0] * 0.80)

train_df = df.iloc[:train_size].reset_index(drop=True)

valid_df = df.iloc[train_size:].reset_index(drop=True)



train_datagen = ImageDataGenerator(

    rescale=1./255,

    width_shift_range=0.2,

    height_shift_range=0.2,

    rotation_range=30,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)

valid_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_dataframe(

    train_df,

    data_dir,

    x_col='filename',

    y_col='label',

    target_size=(input_shape[0], input_shape[1]),

    batch_size=32,

    class_mode='binary',

    shuffle=True,

    seed=42,

)

valid_generator = valid_datagen.flow_from_dataframe(

    valid_df,

    data_dir,

    x_col='filename',

    y_col='label',

    target_size=(input_shape[0], input_shape[1]),

    batch_size=32,

    class_mode='binary',

    shuffle=True,

    seed=42,

)



print(train_df['label'].value_counts())

print(valid_df['label'].value_counts())

for c in train_generator.class_indices:

    if train_generator.class_indices[c] != valid_generator.class_indices[c]:

        raise ValueError(f"Mismatching Classses: {train_generator.class_indices[c]} {valid_generator.class_indices[c]}")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



early_stop_callback = EarlyStopping(monitor='val_loss',

                                    verbose=1,

                                    patience=2)

checkpoint_callback = ModelCheckpoint('./best-model.h5', save_best_only=True)
history = model.fit_generator(

    train_generator,

    epochs=100,

    callbacks=[

        early_stop_callback,

        checkpoint_callback

    ],

    validation_data=valid_generator,

    verbose=1,

    shuffle=True,

)
def plot_results(h):

    plt.plot(h['loss'], label='Train Loss')

    plt.plot(h['val_loss'], label='Val. Loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend()

    plt.show()
plot_results(history)
def get_transfer_model():

    pretrained_model = Xception(weights='../input/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',

                                include_top=False,

                                input_shape=input_shape)

    for l in pretrained_model.layers:

        l.trainable = False

    

    X_input = Input(shape=input_shape)

    X = pretrained_model(X_input)

    X = Flatten()(X)

    X = Dense(256, activation='relu')(X)

    X = Dropout(0.25)(X)

    out = Dense(1, activation='sigmoid')(X)

    return Model(X_input, [out])
transfer_model = get_transfer_model()
transfer_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



early_stop_callback = EarlyStopping(monitor='val_loss',

                                    verbose=1,

                                    patience=5,

                                    min_delta=1e-3)

checkpoint_callback = ModelCheckpoint('./transfer-best-model.h5', save_best_only=True)
transfer_history = transfer_model.fit_generator(

    train_generator,

    epochs=100,

    callbacks=[

        early_stop_callback,

        checkpoint_callback

    ],

    validation_data=valid_generator,

    verbose=1,

    shuffle=True,

)
all_history = {

    'loss': [],

    'val_loss': []

}

all_history['loss'] = all_history['loss'] + transfer_history.history['loss']

all_history['val_loss'] = all_history['val_loss'] + transfer_history.history['val_loss']



plot_results(all_history)