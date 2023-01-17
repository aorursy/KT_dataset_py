# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from matplotlib import pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

import random 

import os

import cv2

import gc

from tqdm.auto import tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from tensorflow.keras.models import load_model

model_root = load_model('/kaggle/input/pretrained/model_root.h5')

model_vowel = load_model('/kaggle/input/pretrained/model_vowel.h5')

model_consonant = load_model('/kaggle/input/pretrained/model_consonant.h5')
# Single data inference



from tensorflow.keras.models import load_model

import numpy as np

import pandas as pd



# model_root = load_model('model_root.h5')

# model_vowel = load_model('model_vowel.h5')

# model_consonant = load_model('model_consonant.h5')



# model_root.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])

# model_vowel.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])

# model_consonant.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])



def resize(df, size=64, need_progress_bar=True):

    resized = {}

    for i in range(df.shape[0]):

        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

        resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T

    return resized



model_dict = {

    'grapheme_root': model_root,

    'vowel_diacritic': model_vowel,

    'consonant_diacritic': model_consonant

}



# preds_dict = {

#     'grapheme_root': [],

#     'vowel_diacritic': [],

#     'consonant_diacritic': []

# }



test_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}



components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target=[] # model predictions placeholder

row_id=[] # row_id place holder

top_5_scores=[] # top 5 target:confidence scores



# test_img = df_train_img_0.iloc[0]



df_test_img_0 = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_0.parquet') 

df_test_img_0.set_index('image_id', inplace=True)



test_img = df_test_img_0[df_test_img_0.index == 'Test_2']



X_test = resize(test_img)/255

X_test = X_test.values.reshape(-1, 64, 64, 1)



# for pred in preds_dict:

#     preds_dict[pred]=np.argmax(model_dict[pred].predict(X_test), axis=1)

    

for pred in test_dict:

    np.set_printoptions(suppress=True)

#     preds_dict[pred]=np.argmax(model_dict[pred].predict(X_test), axis=1)

    predictions = model_dict[pred].predict(X_test)

    top5scores = predictions.flatten()[np.argsort(-predictions.flatten())[:5]]

    top5targets = np.argsort(-predictions.flatten())[:5]

    result = dict(zip(top5targets,top5scores))

    resList = [(k,v) for k,v in result.items()]

    test_dict[pred] = resList

    



for k,id in enumerate(test_img.index.values):  

    for i,comp in enumerate(components):

        id_sample=id+'_'+comp

        row_id.append(id_sample)

        target.append(test_dict[comp][0])

        top_5_scores.append(test_dict[comp])

        

del test_img

del X_test

gc.collect()



df_sample = pd.DataFrame(

    {

        'row_id': row_id,

        'target':target,

        'top_5_scores':top_5_scores

    },

    columns = ['row_id','target','top_5_scores'] 

)

# df_sample.to_csv('submission_root.csv',index=False)

df_sample.head()
# load saved model and infer on test data to get submission file



import numpy as np

import pandas as pd



# model_root = load_model('model_root.h5')

# model_vowel = load_model('model_vowel.h5')

# model_consonant = load_model('model_consonant.h5')



# model_root.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])

# model_vowel.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])

# model_consonant.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])



def resize(df, size=64, need_progress_bar=True):

    resized = {}

    for i in range(df.shape[0]):

        image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

        resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T

    return resized



model_dict = {

    'grapheme_root': model_root,

    'vowel_diacritic': model_vowel,

    'consonant_diacritic': model_consonant

}



preds_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}



components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target=[] # model predictions placeholder

row_id=[] # row_id place holder

for i in range(4):

    test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

    test_img.set_index('image_id', inplace=True)



    X_test = resize(test_img)/255

    X_test = X_test.values.reshape(-1, 64, 64, 1)



    for pred in preds_dict:

        preds_dict[pred]=np.argmax(model_dict[pred].predict(X_test), axis=1)



    for k,id in enumerate(test_img.index.values):  

        for i,comp in enumerate(components):

            id_sample=id+'_'+comp

            row_id.append(id_sample)

            target.append(preds_dict[comp][k])

    del test_img

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