import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import time

import gc

start_time = time.time()
random_seed = 17025

np.random.seed(random_seed)

from tensorflow.keras.models import load_model

from sklearn.metrics import confusion_matrix

from skimage.transform import resize

import matplotlib.pyplot as plt
#Load train and test data

train = pd.read_csv('../input/bengaliai-cv19/train.csv')

test = pd.read_csv('../input/bengaliai-cv19/test.csv')

gmap = pd.read_csv('../input/bengaliai-cv19/class_map.csv')



pq_paths = {'Train_0':'../input/bengaliai-cv19/train_image_data_0.parquet',

            'Train_1':'../input/bengaliai-cv19/train_image_data_1.parquet',

            'Train_2':'../input/bengaliai-cv19/train_image_data_2.parquet',

            'Train_3':'../input/bengaliai-cv19/train_image_data_3.parquet',

            'Test_0':'../input/bengaliai-cv19/test_image_data_0.parquet',

            'Test_1':'../input/bengaliai-cv19/test_image_data_1.parquet',

            'Test_2':'../input/bengaliai-cv19/test_image_data_2.parquet',

            'Test_3':'../input/bengaliai-cv19/test_image_data_3.parquet'}



target_cols = ['grapheme_root','vowel_diacritic','consonant_diacritic']
def values_from_pqt(name, batch_size):

    

    pqt = pd.read_parquet(pq_paths[name])

    

    PQT_COLUMNS = pqt.columns[1:]

    concurrent_n = batch_size//4

    

    for i in range(0, pqt.shape[0], batch_size):

        

        accumulator_x = np.array([], dtype='float16')

        

        for n in range(i, i+batch_size, concurrent_n):

            print('PQT %s begin %i end %i' %(name, i, n+concurrent_n))

            try:

                value = 1 - (pqt[PQT_COLUMNS][n:n+concurrent_n].values.reshape(concurrent_n, 137, 236, 1))/255

                value = resize(value, (concurrent_n, 80, 80, 1))

            except:

                value = pqt[PQT_COLUMNS][n:].values

                if value.shape[0]:

                    value = 1 - (value.reshape(value.shape[0], 137, 236, 1))/255

                    value = resize(value, (value.shape[0], 80, 80, 1))

                else:

                    break

            try:

                accumulator_x = np.concatenate([accumulator_x, value.astype('float16')], axis=0)

            except:

                accumulator_x = value.astype('float16')

        try:

            yield (np.array(accumulator_x).astype('float16'), pqt.image_id[i:i+batch_size].values.tolist())

        except:

            yield (np.array(accumulator_x).astype('float16'), pqt.image_id[i:].values.tolist())

    del pqt

    gc.collect()
prefix = "../input/three-part-model-training-kernel-only-part-2-3/" #Viable to change



targets = {'grapheme_root': load_model(prefix+'multi_out_cnn_model_root.h5'),

           'vowel_diacritic': load_model(prefix+'multi_out_cnn_model_vowel.h5'),

           'consonant_diacritic': load_model(prefix+'multi_out_cnn_model_consonant.h5')

          }



target_cols = ['consonant_diacritic','grapheme_root','vowel_diacritic'] #Arrange in this order in sample_submission



predictions = {'row_id':[], 'target':{

    'consonant_diacritic':[],

    'grapheme_root':[],

    'vowel_diacritic':[]}

              }



print("Start reading data:")



for i in range(4):

    for test_x, ids in values_from_pqt("Test_{}".format(i), 10000):

        print("Currently Predicting: Test_{}".format(i))

        #For X values

        for component in target_cols:

            print("For %s:" %component)

            preds = np.argmax(targets[component].predict(test_x), axis=1)

            try:

                predictions['target'][component] = np.concatenate([

                    predictions['target'][component], preds],

                    axis=0

                )

            except:

                predictions['target'][component] = preds

        #For Y values

        try:

            predictions['row_id'] = np.concatenate([

                predictions['row_id'], ids

            ])

        except:

            predictions['row_id'] = ids

        print("#"*72)

ids = []

tgt = []

for i in range(len(predictions['row_id'])):

    for col in target_cols:

        ids.append(predictions['row_id'][i]+'_'+col)

        tgt.append(int(predictions['target'][col][i]))



submission = pd.DataFrame()

submission['row_id'] = ids

submission['target'] = tgt

submission.to_csv('submission.csv', index=False)

print(submission)
end_time = time.time()

total_time = end_time - start_time

hours = total_time//3600

minutes = (total_time%3600)//60

seconds = (total_time%60)

print("Total Time spent is: %i hours, %i minutes, and %i seconds" %((hours, minutes, seconds)))