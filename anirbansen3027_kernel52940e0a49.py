# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from keras.callbacks import ModelCheckpoint



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm.auto import tqdm

import time, gc

import cv2



import matplotlib.image as mpimg

from keras.models import Sequential, Model

from keras.models import clone_model

from keras.layers import Dense,Flatten,Dropout, Input

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

import seaborn as sns

from keras.applications.densenet import DenseNet121

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def buildmodel(): 

    model = DenseNet121(weights=None,input_tensor = Input(shape=(64, 64, 1)), include_top=False)

    x = model.layers[-1].output

    x = Flatten()(x)

    model_root = Dense(168, activation = 'softmax')(x)

    model_vowel = Dense(11, activation = 'softmax')(x)

    model_consonant = Dense(7, activation = 'softmax')(x)



    outputs_list = [model_root, model_vowel, model_consonant]



    model = Model(inputs = model.layers[0].input, outputs=[model_root, model_vowel, model_consonant])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_weights("/kaggle/input/pretrained-25eto50e/keras50e.model")

    return model
model = buildmodel()
class_map = pd.read_csv("../input/bengaliai-cv19/class_map.csv")

sample_submission = pd.read_csv("../input/bengaliai-cv19/sample_submission.csv")

test = pd.read_csv("../input/bengaliai-cv19/test.csv")
def resize(df, size=64):

    resized = {}

    for i in tqdm(range(df.shape[0])):

        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

        resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T

    return resized
IMG_SIZE=64

N_CHANNELS=1
preds_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}

components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target=[] # model predictions placeholder

row_id=[] # row_id place holder

for i in tqdm(range(4)):

    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

    df_test_img.set_index('image_id', inplace=True)



    X_test = resize(df_test_img)/255

    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

    

    preds = model.predict(X_test)



    for i, p in enumerate(preds_dict):

        preds_dict[p] = np.argmax(preds[i], axis=1)



    for k,id in enumerate(df_test_img.index.values):  

        for i,comp in enumerate(components):

            id_sample=id+'_'+comp

            row_id.append(id_sample)

            target.append(preds_dict[comp][k])

    del df_test_img

    del X_test

    gc.collect()



df_sample = pd.DataFrame(

    {

        'row_id': row_id,

        'target':target

    },

    columns = ['row_id','target'] 

)

df_sample.to_csv('submission.csv',index=False)

df_sample.head()