#Import dependencies

import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import gc

import glob

import os

from tqdm import tqdm
sub_file = pd.read_csv('../input/sample_submission.csv', index_col=0)
%%time

#Load data from train file

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.float32, 'time_to_failure': np.float32})
#Split the big train series into a fixed number of segments. Searching directly in the full

#train series was too much for my computer 

number_train_segments = 3

train_segment_length = int(train.shape[0] / number_train_segments)
#Read segment test data names

tests = glob.glob('../input/test/**')

tests_names = os.listdir('../input/test/')
#Search the test pattern in the train data segments using the correlation coefficient and OpenCV

sub_list = []

for j in tqdm(range(5)): #range(len(tests))): Only 5 as an example, because commiting with all of them takes too much

                         #I only want to share the methodology

    #Read segment data

    segment_test = pd.read_csv(tests[j], dtype={'acoustic_data': np.float32})

    segment_test = np.float32( segment_test['acoustic_data'].values )

    #Resize the vector to have the correct dimensions

    segment_test_tp = np.resize(segment_test, (1,len(segment_test)))



    coefs = []

    for i in range(0,number_train_segments):

        print('Searching similarity for test segment {} with {}-segment of train data:'.format(tests_names[j], i+1))

        segment_train = train.iloc[train_segment_length*i : train_segment_length*(i+1)]

        segment_train = np.float32( segment_train['acoustic_data'].values )

        segment_train_tp = np.resize( segment_train, (1,len(segment_train)) )

    

        gc.collect()

    

        result = cv2.matchTemplate(segment_test_tp, segment_train_tp, cv2.TM_CCORR_NORMED)

        

        #Append the best matching for that train segment (coeff and position)

        coefs.append([np.max(result), train_segment_length*i + np.argmax(result) + segment_test_tp.shape[1]-1])

    

    #Apprend the best result among all train segments

    coefs = np.array(coefs)

    sub_list.append( [ tests_names[j], train.time_to_failure.iloc[int(coefs[np.argmax(coefs[:,0]),1])] ] )
sub_df = pd.DataFrame(data=sub_list, columns=['seg_id','time_to_failure'])

sub_df['seg_id'] = sub_df['seg_id'].apply(lambda x: x[:-4])

sub_df.set_index('seg_id',inplace=True)



#Read submission_file and rearrange the index in the 

sub_file = pd.read_csv('../input/sample_submission.csv', index_col=0)

sub_df = sub_df.reindex(sub_file.index)

sub_file = sub_df



sub_file.to_csv('sub_file_v00.csv')