# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



import os

# import gc

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# !pip uninstall tensorflow --yes

# !pip install tensorflow-gpu 

# The GPU id to use, usually either "0" or "1";

# os.environ["CUDA_VISIBLE_DEVICES"]="0"; 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import keras

import tensorflow as tf

from tensorflow.keras.models import load_model

# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.
test_files = glob.glob('/kaggle/input/bengaliai-cv19/test_image_data_*')

test_files.sort()

# model = load_model('/kaggle/input/resnet50/ResNet50.hdf5')
# test_files
# dict_={"row_id":[],"target":[]}

# testX = []

# image_ids = []

# batch_size=256

# for file in test_files:

    

#     df = pd.read_parquet(file, engine='pyarrow')

    

#     for idx in range(len(df)):

        

#         img_name = df.iloc[idx]['image_id']

#         image_ids.append(img_name)

#         image = df.loc[df.index[idx]].values[1:].reshape(137,236)

#         image = np.uint8(image)

#         testX.append(np.repeat(image[..., np.newaxis], 3, -1))

        

#         if len(testX) >= batch_size:

            

#             grayscale_batch = np.array(testX)

# #             rgb_batch = np.repeat(grayscale_batch[..., np.newaxis], 3, -1)



#             predictions = model.predict(grayscale_batch)



#             for idx in range(len(image_ids)):

#                 img_name = image_ids[idx]

#                 for key,value in {0:"grapheme_root",1:"vowel_diacritic",2:"consonant_diacritic"}.items():

#                     name  = img_name+'_'+value

#                     val = np.argmax(predictions[key][idx])

#                     dict_['row_id'].append(name)

#                     dict_['target'].append(val)

#             testX = []

#             image_ids = []

            

            

            

            

        

        

        

#     if len(testX)>0:     

#         grayscale_batch = np.array(testX)

#     #     rgb_batch = np.repeat(grayscale_batch[..., np.newaxis], 3, -1)



#         predictions = model.predict(grayscale_batch)



#         for idx in range(len(image_ids)):

#             img_name = image_ids[idx]

#             for key,value in {0:"grapheme_root",1:"vowel_diacritic",2:"consonant_diacritic"}.items():

#                 name  = img_name+'_'+value

#                 val = np.argmax(predictions[key][idx])

#                 dict_['row_id'].append(name)

#                 dict_['target'].append(val)

#         testX = []

#         image_ids = []

    

    
dict_={"row_id":[],"target":[]}

testX = []

for file in test_files:

    df = pd.read_parquet(file, engine='pyarrow')

    for idx in range(len(df)):

        img_name = df.iloc[idx]['image_id']

        for key,value in {0:"grapheme_root",1:"vowel_diacritic",2:"consonant_diacritic"}.items():

            name  = img_name+'_'+value

            val = 0

            dict_['row_id'].append(name)

            dict_['target'].append(val)

# dict_={"row_id":[],"target":[]}

# df = pd.read_csv('../input/submission/submission.csv')

# for index in range(len(df)):

#     dict_['row_id'].append(df.iloc[index]['row_id'])

#     dict_['target'].append(df.iloc[index]['target'])
submission = pd.DataFrame(dict_)

submission.to_csv('submission.csv',index=False)

    
submission.head()
len(submission)