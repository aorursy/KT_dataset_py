# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import tensorflow.keras

from tensorflow.keras.preprocessing.image import img_to_array

import cv2

import tensorflow

from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D 

from tensorflow.keras.layers import MaxPooling2D 

from tensorflow.keras.layers import Activation 

from tensorflow.keras.layers import Flatten 

from tensorflow.keras.layers import Dense 

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Input

from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.optimizers import Adam

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

test_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

sample_sub_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
train_df_.head()
train_df_ = train_df_.drop(['grapheme'], axis=1, inplace=False)
train_df_.head()
train_df_[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = train_df_[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')
width=64

height=64

depth=1
for i in range(4):

    train_df = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
train_df.columns
X = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)
X.columns
def get_dummies(df):

    cols = []

    for col in df:

        cols.append(pd.get_dummies(df[col].astype(str)))

    return pd.concat(cols, axis=1)
Y_train_root = pd.get_dummies(train_df['grapheme_root']).values

Y_train_vowel = pd.get_dummies(train_df['vowel_diacritic']).values

Y_train_consonant = pd.get_dummies(train_df['consonant_diacritic']).values
print(f'Training images: {X.shape}')

print(f'Training labels root: {Y_train_root.shape}')

print(f'Training labels vowel: {Y_train_vowel.shape}')

print(f'Training labels consonants: {Y_train_consonant.shape}')
X.shape
X.iloc[0]
def preprocess(image):

    image = cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA)

    return img_to_array(image)
data=[]

for i in range(X.shape[0]):

    preprocess_image = preprocess(X.iloc[i].values) 

    data.append(preprocess_image)
len(data)
del X

del train_df

del train_df_
data = np.array(data).astype("float") / 255.0
x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(data, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.1)
x_train.shape
#del data

del Y_train_root, Y_train_vowel, Y_train_consonant
def build_category(inputs, numCategories,catname, finalAct="softmax", chanDim=-1):



    #CONV => RELU => POOL

    x = Conv2D(32, (3, 3), padding="same")(inputs)

    x = Activation("relu")(x)

    x = BatchNormalization(axis=chanDim)(x)

    x = MaxPooling2D(pool_size=(3, 3))(x)

    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding="same")(x)

    x = Activation("relu")(x)

    x = BatchNormalization(axis=chanDim)(x)

    x = Conv2D(64, (3, 3), padding="same")(x)

    x = Activation("relu")(x)

    x = BatchNormalization(axis=chanDim)(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), padding="same")(x)

    x = Activation("relu")(x)

    x = BatchNormalization(axis=chanDim)(x)

    x = Conv2D(128, (3, 3), padding="same")(x)

    x = Activation("relu")(x)

    x = BatchNormalization(axis=chanDim)(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(0.25)(x)

    x = Flatten()(x)

    x = Dense(256)(x)

    x = Activation("relu")(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(numCategories)(x)

    x = Activation(finalAct, name=catname)(x)

    # return the category prediction sub-network

    return x
inputShape = (height, width, 1)

inputs=Input(shape=inputShape)

root = build_category(inputs,168,'cat_root')

vowel = build_category(inputs,11,'cat_vowel')

consonant = build_category(inputs,7,'cat_consonant')



model=Model(inputs=inputs, outputs=[root,vowel,consonant],name="bengaliai")

INIT_LR=0.1

EPOCHS=40

losses = {"cat_root": "categorical_crossentropy","cat_vowel": "categorical_crossentropy","cat_consonant": "categorical_crossentropy"}

lossWeights = {"cat_root": 1.0, "cat_vowel": 1.0, "cat_consonant":1.0}



opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,metrics=["accuracy"])
H=model.fit(x_train, {"cat_root":y_train_root,"cat_vowel":y_train_vowel, "cat_consonant":y_train_consonant}, 

           validation_data=(x_test,{"cat_root":y_test_root,"cat_vowel":y_test_vowel, "cat_consonant":y_test_consonant}),

           epochs=EPOCHS,

           verbose=1)