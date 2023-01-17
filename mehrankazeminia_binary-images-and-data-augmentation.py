import os

import gc

import cv2

import csv

import json

import time

import copy



import numpy as np

import pandas as pd

import seaborn as sns

import plotly.express as px

import plotly.figure_factory as ff

import matplotlib.pyplot as plt

from matplotlib import colors

from pathlib import Path



from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Dense, Dropout, Activation



%matplotlib inline

%load_ext autoreload

%autoreload 2

for dirname, _, filenames in os.walk('/kaggle/input/abstraction-and-reasoning-challenge/'):

    print(dirname)

    

data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'

working_path = Path('/kaggle/working/')



training_tasks = sorted(os.listdir(training_path))

evaluation_tasks = sorted(os.listdir(evaluation_path))

test_tasks = sorted(os.listdir(test_path))



print()

print(training_tasks[:5])

print(evaluation_tasks[:5])

print(test_tasks[:5])      

    
EM_PATH = test_path



em_task_files = sorted(os.listdir(EM_PATH))

em_tasks = []



for task_file in em_task_files:

    with open(str(EM_PATH / task_file), 'r') as f:

        task = json.load(f)

        em_tasks.append(task)



EM_LEN = len(em_tasks)

str_tasks_prdn = []

str_tasks = []

prdn = []



print(EM_LEN)

print(task.keys())

print(em_task_files[:5])

print()

print(em_tasks[0])

for i in range(5):    

    emi = em_tasks[i]

    

    contains_train = len(emi['train'])

    contains_test = len(emi['test'])

    

    print(f'i = {i} >>> task contains: {contains_train} training pairs')

    print(f'i = {i} >>> task contains: {contains_test} test pairs')

    

#   ######################################################################

#   ######################################################################



#   Data Augmentation - for mlp

#   train_input & train_output



    train_input0 = np.array(emi['train'][0]['input'])           

    train_input0_fllr = np.fliplr(train_input0) # Flip array in the left/right direction.

    train_input0_flud = np.flipud(train_input0) # Flip array in the up/down direction.

    train_input0_r180 = np.rot90(train_input0, 2) # Rotation direction (180 deg.) is from the first towards the second axis.

    matrix_train_input = np.vstack((train_input0, train_input0_fllr, train_input0_flud, train_input0_r180)) 



        

    train_output0 = np.array(emi['train'][0]['output'])           

    train_output0_fllr = np.fliplr(train_output0) # Flip array in the left/right direction.

    train_output0_flud = np.flipud(train_output0) # Flip array in the up/down direction.

    train_output0_r180 = np.rot90(train_output0, 2) # Rotation direction (180 deg.) is from the first towards the second axis.

    matrix_train_output = np.vstack((train_output0, train_output0_fllr, train_output0_flud, train_output0_r180)) 

    

                

    for j in range(contains_train):

        train_inputj = np.array(emi['train'][j]['input'])

        train_outputj = np.array(emi['train'][j]['output'])

        

        if train_inputj.shape[1] == train_input0.shape[1]: 

            if train_outputj.shape[1] == train_output0.shape[1]:   

                if j > 0:                    

                    train_inputj_fllr = np.fliplr(train_inputj) 

                    train_inputj_flud = np.flipud(train_inputj) 

                    train_inputj_r180 = np.rot90(train_inputj, 2)

                    matrix_train_input = np.vstack((matrix_train_input, train_inputj, train_inputj_fllr, 

                                                    train_inputj_flud, train_inputj_r180))    

                    

                    train_outputj_fllr = np.fliplr(train_outputj) 

                    train_outputj_flud = np.flipud(train_outputj) 

                    train_outputj_r180 = np.rot90(train_outputj, 2)

                    matrix_train_output = np.vstack((matrix_train_output, train_outputj, train_outputj_fllr, 

                                                     train_outputj_flud, train_outputj_r180))    

            

    flattrainin = matrix_train_input.flatten()

    flattrainout = matrix_train_output.flatten()

    

#   ######################################################################

    

#   test_input & test_output



    test_input0 = np.array(emi['test'][0]['input'])    

    test_input = test_input0

    flattestin = test_input.flatten()

         

    # test_output0 = np.array(emi['test'][0]['output'])           

    # test_output = test_output0

    # flattestout = test_output.flatten()                

            

#   ######################################################################

#   ######################################################################



#   Categorical Data - for mlp



    flatdata = np.hstack((flattrainin, flattrainout, flattestin)) 

    

    lb = LabelBinarizer()

    zipBinar = lb.fit(flatdata)

    zipdata = zipBinar.transform(flatdata)

            

    dim = zipdata.shape[1]

    df = set(flatdata)

    

    ziptrainin = zipBinar.transform(flattrainin)

    flatziptrainin = ziptrainin.flatten()

    

    ziptrainout = zipBinar.transform(flattrainout)

    flatziptrainout = ziptrainout.flatten()

    

    ziptestin = zipBinar.transform(flattestin)

    flatziptestin = ziptestin.flatten()

    

    # ziptestout = zipBinar.transform(flattestout)

    # flatziptestout = ziptestout.flatten()



    print(f'i = {i} for mlp >>> ziptrainin.shape: {ziptrainin.shape}')

    print(f'i = {i} for mlp >>> flatziptrainin.shape: {flatziptrainin.shape}')    

    print(f'i = {i} for mlp >>> ziptrainout.shape: {ziptrainout.shape}')

    print(f'i = {i} for mlp >>> flatziptrainout.shape: {flatziptrainout.shape}')    

    print(f'i = {i} for mlp >>> ziptestin.shape: {ziptestin.shape}')

    print(f'i = {i} for mlp >>> flatziptestin.shape: {flatziptestin.shape}')    

    # print(f'i = {i} for mlp >>> ziptestout.shape: {ziptestout.shape}')

    # print(f'i = {i} for mlp >>> flatziptestout.shape: {flatziptestout.shape}')

    print(f'colors: {df}')

    print()



#   ######################################################################

#   ######################################################################



#   Data Augmentation - for cnn

#   train_input & train_output

       

    train_input = []

    ztrain_input = []

    

    flattrainin01 = train_input0.flatten()

    ziptrainin01 = zipBinar.transform(flattrainin01) 

    trainin = np.reshape(ziptrainin01, (train_input0.shape[0], train_input0.shape[1], dim))

    ztrainin = ziptrainin01.flatten()    

    train_input += [trainin]

    ztrain_input += [ztrainin]

    

    flattrainin02 = train_input0_fllr.flatten()

    ziptrainin02 = zipBinar.transform(flattrainin02)

    trainin = np.reshape(ziptrainin02, (train_input0.shape[0], train_input0.shape[1], dim)) 

    ztrainin = ziptrainin02.flatten()

    train_input += [trainin]

    ztrain_input += [ztrainin]

    

    flattrainin03 = train_input0_flud.flatten()

    ziptrainin03 = zipBinar.transform(flattrainin03)

    trainin = np.reshape(ziptrainin03, (train_input0.shape[0], train_input0.shape[1], dim))   

    ztrainin = ziptrainin03.flatten()

    train_input += [trainin]

    ztrain_input += [ztrainin]

       

    flattrainin04 = train_input0_r180.flatten()

    ziptrainin04 = zipBinar.transform(flattrainin04)

    trainin = np.reshape(ziptrainin04, (train_input0.shape[0], train_input0.shape[1], dim)) 

    ztrainin = ziptrainin04.flatten()

    train_input += [trainin]

    ztrain_input += [ztrainin]

    

    

    

    train_output = []

    ztrain_output = []

    

    flattrainout01 = train_output0.flatten()

    ziptrainout01 = zipBinar.transform(flattrainout01) 

    trainout = np.reshape(ziptrainout01, (train_output0.shape[0], train_output0.shape[1], dim))

    ztrainout = ziptrainout01.flatten()

    # ztrainout =np.reshape(ziptrainout01, (-1, dim))     

    train_output += [trainout]

    ztrain_output += [ztrainout]

    

    flattrainout02 = train_output0_fllr.flatten()

    ziptrainout02 = zipBinar.transform(flattrainout02)

    trainout = np.reshape(ziptrainout02, (train_output0.shape[0], train_output0.shape[1], dim))

    ztrainout = ziptrainout02.flatten()

    # ztrainout =np.reshape(ziptrainout02, (-1, dim))

    train_output += [trainout]

    ztrain_output += [ztrainout]

    

    flattrainout03 = train_output0_flud.flatten()

    ziptrainout03 = zipBinar.transform(flattrainout03)

    trainout = np.reshape(ziptrainout03, (train_output0.shape[0], train_output0.shape[1], dim))

    ztrainout = ziptrainout03.flatten()

    # ztrainout =np.reshape(ziptrainout03, (-1, dim))

    train_output += [trainout]

    ztrain_output += [ztrainout]

        

    flattrainout04 = train_output0_r180.flatten()

    ziptrainout04 = zipBinar.transform(flattrainout04)

    trainout = np.reshape(ziptrainout04, (train_output0.shape[0], train_output0.shape[1], dim))

    ztrainout = ziptrainout04.flatten()

    # ztrainout =np.reshape(ziptrainout04, (-1, dim))

    train_output += [trainout]

    ztrain_output += [ztrainout]

                

          

    for j in range(contains_train):        

        train_inputj = np.array(emi['train'][j]['input']) 

        train_outputj = np.array(emi['train'][j]['output']) 

        

        if train_inputj.shape == train_input0.shape: 

            if train_outputj.shape == train_output0.shape:

                if j > 0:                    

                    flattraininj1 = train_inputj.flatten()

                    ziptraininj1 = zipBinar.transform(flattraininj1) 

                    trainin = np.reshape(ziptraininj1, (train_input0.shape[0], train_input0.shape[1], dim))  

                    ztrainin = ziptraininj1.flatten()

                    train_input += [trainin]

                    ztrain_input += [ztrainin]

    

                    train_inputj_fllr = np.fliplr(train_inputj) 

                    flattraininj2 = train_inputj_fllr.flatten()

                    ziptraininj2 = zipBinar.transform(flattraininj2)            

                    trainin = np.reshape(ziptraininj2, (train_input0.shape[0], train_input0.shape[1], dim)) 

                    ztrainin = ziptraininj2.flatten()

                    train_input += [trainin]

                    ztrain_input += [ztrainin]

    

                    train_inputj_flud = np.flipud(train_inputj) 

                    flattraininj3 = train_inputj_flud.flatten()

                    ziptraininj3 = zipBinar.transform(flattraininj3)

                    trainin = np.reshape(ziptraininj3, (train_input0.shape[0], train_input0.shape[1], dim))

                    ztrainin = ziptrainin03.flatten()

                    train_input += [trainin]

                    ztrain_input += [ztrainin]

    

                    train_inputj_r180 = np.rot90(train_inputj, 2)    

                    flattraininj4 = train_inputj_r180.flatten()

                    ziptraininj4 = zipBinar.transform(flattraininj4)

                    trainin = np.reshape(ziptraininj4, (train_input0.shape[0], train_input0.shape[1], dim))

                    ztrainin = ziptraininj4.flatten()

                    train_input += [trainin]

                    ztrain_input += [ztrainin]

                    

                        

      

                    flattrainoutj1 = train_outputj.flatten()

                    ziptrainoutj1 = zipBinar.transform(flattrainoutj1) 

                    trainout = np.reshape(ziptrainoutj1, (train_output0.shape[0], train_output0.shape[1], dim))

                    ztrainout = ziptrainoutj1.flatten()

                    # ztrainout =np.reshape(ziptrainoutj1, (-1, dim))

                    train_output += [trainout]

                    ztrain_output += [ztrainout]

    

                    train_outputj_fllr = np.fliplr(train_outputj) 

                    flattrainoutj2 = train_outputj_fllr.flatten()

                    ziptrainoutj2 = zipBinar.transform(flattrainoutj2)                  

                    trainout = np.reshape(ziptrainoutj2, (train_output0.shape[0], train_output0.shape[1], dim)) 

                    ztrainout = ziptrainoutj2.flatten()

                    # ztrainout =np.reshape(ziptrainoutj2, (-1, dim))

                    train_output += [trainout]

                    ztrain_output += [ztrainout]

    

                    train_outputj_flud = np.flipud(train_outputj) 

                    flattrainoutj3 = train_outputj_flud.flatten()

                    ziptrainoutj3 = zipBinar.transform(flattrainoutj3)

                    trainout = np.reshape(ziptrainoutj3, (train_output0.shape[0], train_output0.shape[1], dim))

                    ztrainout = ziptrainoutj3.flatten()

                    # ztrainout =np.reshape(ziptrainoutj3, (-1, dim))

                    train_output += [trainout]

                    ztrain_output += [ztrainout]

    

                    train_outputj_r180 = np.rot90(train_outputj, 2) 

                    flattrainoutj4 = train_outputj_r180.flatten()

                    ziptrainoutj4 = zipBinar.transform(flattrainoutj4)

                    trainout = np.reshape(ziptrainoutj4, (train_output0.shape[0], train_output0.shape[1], dim))

                    ztrainout = ziptrainoutj4.flatten()

                    # ztrainout =np.reshape(ziptrainoutj4, (-1, dim))

                    train_output += [trainout]

                    ztrain_output += [ztrainout]

                    

    ntrain_input=np.array(train_input)

    nztrain_input=np.array(ztrain_input)

    

    ntrain_output=np.array(train_output)    

    nztrain_output=np.array(ztrain_output)

                          

#   ######################################################################



#   test_input & test_output  



    flattestin = test_input.flatten()

    ziptestin = zipBinar.transform(flattestin)

    testin = np.reshape(ziptestin, (test_input.shape[0], test_input.shape[1], dim))    

    ntestin = np.reshape(testin, (-1, test_input.shape[0], test_input.shape[1], dim))

    

    ztestin = ziptestin.flatten()    

    nztestin = np.array(ztestin)

    nztestin = ztestin[None, :]

    

    # flattestout = test_output.flatten()

    # ziptestout = zipBinar.transform(flattestout)

    # testout = np.reshape(ziptestout, (test_output.shape[0], test_output.shape[1], dim))

    

    # ntestout = testout.reshape(-1, dimin0, dimin1, dim)

    

#   ######################################################################    

                    

    print(f'i = {i} for cnn >>> len(train_input): {len(train_input)}')

    print(f'i = {i} for cnn >>> ntrain_input.shape: {ntrain_input.shape}')

    print()    

    print(f'i = {i} for cnn >>> len(ztrain_input): {len(ztrain_input)}')

    print(f'i = {i} for cnn >>> nztrain_input.shape: {nztrain_input.shape}')

    print()      

    print(f'i = {i} for cnn >>> len(train_output): {len(train_output)}')

    print(f'i = {i} for cnn >>> ntrain_output.shape: {ntrain_output.shape}')    

    print()        

    print(f'i = {i} for cnn >>> len(ztrain_output): {len(ztrain_output)}')

    print(f'i = {i} for cnn >>> nztrain_output.shape: {nztrain_output.shape}')    

    print()    

    print(f'i = {i} for cnn >>> testin.shape: {testin.shape}')

    print(f'i = {i} for cnn >>> ntestin.shape: {ntestin.shape}')  

    print()

    print(f'i = {i} for cnn >>> ztestin.shape: {ztestin.shape}')

    print(f'i = {i} for cnn >>> nztestin.shape: {nztestin.shape}')

    print()

    # print(f'i = {i} for cnn >>> testout.shape: {testout.shape}')

    # print(f'i = {i} for cnn >>> ntestout.shape: {ntestout.shape}')    

    # print() 

    print()

    print()

#   ######################################################################

submission = pd.read_csv(data_path / 'sample_submission.csv', index_col='output_id')

display(submission.head())

for i in range(EM_LEN):    

    emi = em_tasks[i]

    aaa = 777

    

    contains_train = len(emi['train'])

    contains_test = len(emi['test'])

    

    print(f'i = {i} >>> task contains {contains_train} training pairs')

    print(f'i = {i} >>> task contains {contains_test} test pairs')    

    

#   ########################################################################################################  

#   ########################################################################################################  



#   Data Augmentation

#   train_input & train_output



    train_input0 = np.array(emi['train'][0]['input'])           

    train_input0_fllr = np.fliplr(train_input0) # Flip array in the left/right direction.

    train_input0_flud = np.flipud(train_input0) # Flip array in the up/down direction.

    train_input0_r180 = np.rot90(train_input0, 2) # Rotation direction (180 deg.) is from the first towards the second axis.

    train_input0_r270 = np.rot90(train_input0, 3) # Rotation direction (270 deg.) is from the first towards the second axis.

    train_input0_r90 = np.rot90(train_input0, 1) # Rotation direction (90 deg.) is from the first towards the second axis.

    

    flattrainin0 = train_input0.flatten()

    flattrainin0_fllr = train_input0_fllr.flatten() 

    flattrainin0_flud = train_input0_flud.flatten() 

    flattrainin0_r180 = train_input0_r180.flatten()

    flattrainin0_r270 = train_input0_r270.flatten()

    flattrainin0_r90 = train_input0_r90.flatten()    

    flattraininx = flattrainin0

    flattraininxx =np.hstack((flattrainin0, flattrainin0_fllr, flattrainin0_flud, flattrainin0_r180))

    flattrainin = np.hstack((flattrainin0, flattrainin0_fllr, flattrainin0_flud, flattrainin0_r180, 

                             flattrainin0_r270, flattrainin0_r90))

    

    train_output0 = np.array(emi['train'][0]['output'])           

    train_output0_fllr = np.fliplr(train_output0) # Flip array in the left/right direction.

    train_output0_flud = np.flipud(train_output0) # Flip array in the up/down direction.

    train_output0_r180 = np.rot90(train_output0, 2) # Rotation direction (180 deg.) is from the first towards the second axis.

    train_output0_r270 = np.rot90(train_output0, 3) # Rotation direction (270 deg.) is from the first towards the second axis.

    train_output0_r90 = np.rot90(train_output0, 1) # Rotation direction (90 deg.) is from the first towards the second axis.

    

    flattrainout0 = train_output0.flatten()

    flattrainout0_fllr = train_output0_fllr.flatten() 

    flattrainout0_flud = train_output0_flud.flatten() 

    flattrainout0_r180 = train_output0_r180.flatten()

    flattrainout0_r270 = train_output0_r270.flatten()

    flattrainout0_r90 = train_output0_r90.flatten() 

    flattrainoutx = flattrainout0

    flattrainoutxx = np.hstack((flattrainout0, flattrainout0_fllr, flattrainout0_flud, flattrainout0_r180))    

    flattrainout = np.hstack((flattrainout0, flattrainout0_fllr, flattrainout0_flud, flattrainout0_r180, 

                              flattrainout0_r270, flattrainout0_r90))

                

    for j in range(contains_train):                                 

        if j > 0:

            train_inputj = np.array(emi['train'][j]['input']) 

            train_inputj_fllr = np.fliplr(train_inputj) 

            train_inputj_flud = np.flipud(train_inputj) 

            train_inputj_r180 = np.rot90(train_inputj, 2)

            train_inputj_r270 = np.rot90(train_inputj, 3)

            train_inputj_r90 = np.rot90(train_inputj, 1)

            

            flattraininj = train_inputj.flatten()

            flattraininj_fllr = train_inputj_fllr.flatten() 

            flattraininj_flud = train_inputj_flud.flatten() 

            flattraininj_r180 = train_inputj_r180.flatten()

            flattraininj_r270 = train_inputj_r270.flatten()

            flattraininj_r90 = train_inputj_r90.flatten()            

            flattraininx = np.hstack((flattraininx, flattraininj))

            flattraininxx = np.hstack((flattraininxx, flattraininj, flattraininj_fllr, flattraininj_flud, flattraininj_r180))            

            flattrainin = np.hstack((flattrainin, flattraininj, flattraininj_fllr, flattraininj_flud, flattraininj_r180, 

                                     flattraininj_r270, flattraininj_r90))          

            

            train_outputj = np.array(emi['train'][j]['output']) 

            train_outputj_fllr = np.fliplr(train_outputj) 

            train_outputj_flud = np.flipud(train_outputj) 

            train_outputj_r180 = np.rot90(train_outputj, 2)

            train_outputj_r270 = np.rot90(train_outputj, 3)

            train_outputj_r90 = np.rot90(train_outputj, 1)

            

            flattrainoutj = train_outputj.flatten()

            flattrainoutj_fllr = train_outputj_fllr.flatten() 

            flattrainoutj_flud = train_outputj_flud.flatten() 

            flattrainoutj_r180 = train_outputj_r180.flatten()

            flattrainoutj_r270 = train_outputj_r270.flatten()

            flattrainoutj_r90 = train_outputj_r90.flatten() 

            flattrainoutx = np.hstack((flattrainoutx, flattrainoutj))

            flattrainoutxx = np.hstack((flattrainoutxx, flattrainoutj, flattrainoutj_fllr, flattrainoutj_flud, flattrainoutj_r180))       

            flattrainout = np.hstack((flattrainout, flattrainoutj, flattrainoutj_fllr, flattrainoutj_flud, flattrainoutj_r180, 

                                      flattrainoutj_r270, flattrainoutj_r90))



#   ########################################################################################################     

    

#   test_input & test_output

                        

    test_input0 = np.array(emi['test'][0]['input'])    

    test_input = test_input0

    flattestin = test_input.flatten()

    

    # test_output0 = np.array(emi['test'][0]['output'])           

    # test_output = test_output0

    # flattestout = test_output.flatten()                

            

#   ######################################################################################################## 

#   ######################################################################################################## 



#   Categorical Data



    flatdata = np.hstack((flattrainin, flattrainout, flattestin))

    

    lb = LabelBinarizer()

    zipBinar = lb.fit(flatdata)

    zipdata = zipBinar.transform(flatdata)

    

    dim = zipdata.shape[1]

    df = set(flatdata)

        

    ziptraininx = zipBinar.transform(flattraininx)

    ziptrainoutx = zipBinar.transform(flattrainoutx)

    

    ziptraininxx = zipBinar.transform(flattraininxx)

    ziptrainoutxx = zipBinar.transform(flattrainoutxx)

    

    ziptrainin = zipBinar.transform(flattrainin)

    ziptrainout = zipBinar.transform(flattrainout)

    

    ziptestin = zipBinar.transform(flattestin)

    # ziptestout = zipBinar.transform(flattestout)

    

    

    print(f'i = {i} >>> ziptraininx.shape: {ziptraininx.shape} ... ziptrainoutx.shape: {ziptrainoutx.shape}')

    print(f'i = {i} >>> ziptraininxx.shape: {ziptraininxx.shape} ... ziptrainoutxx.shape: {ziptrainoutxx.shape}')

    print(f'i = {i} >>> ziptrainin.shape: {ziptrainin.shape} ... ziptrainout.shape: {ziptrainout.shape}')

    print(f'i = {i} >>> ziptestin.shape: {ziptestin.shape}')

    print(f'colors: {df}')

    print()

    

#   ######################################################################################################## 

#   ######################################################################################################## 



#   Limitations and Exceptions



    dimtrainin = flattrainin.shape[0]

    dimtrainout = flattrainout.shape[0]

    

    if dimtrainin < dimtrainout:

        aaa = 0

        

    if dimtrainin > 1.5 * dimtrainout:

        aaa = 0

        

    if np.sum(train_input0) == 0:

        aaa = 0

            

    dimin0 = test_input.shape[0]

    dimin1 = test_input.shape[1]



    dim1 = dim * dimin1

    dim2 = dim * dimin0 * dimin1

    dim3 = dimin0 * dimin1

                 

#   ########################################################################################################

#   ######################################################################################################## 



#   K-Nearest Neighbor classification >>> For predicted-1



    neigh1 = KNeighborsClassifier(n_neighbors=3)

    

    if aaa == 777:

        neigh1.fit(ziptrainin, ziptrainout)  

        pred1_classes = neigh1.predict(ziptestin)

        pred1_classes = zipBinar.inverse_transform(pred1_classes) 



        if test_input0.shape == train_input0.shape :

            pr1 = pred1_classes.reshape(-1, train_output0.shape[1])

            

        if test_input0.shape != train_input0.shape :

            pr1 = pred1_classes.reshape(-1, test_input0.shape[1])

 

#   ########################################################################################################



#   K-Nearest Neighbor classification >>> For predicted-2



    neigh2 = KNeighborsClassifier(n_neighbors=dim)

    

    if aaa == 777:

        neigh2.fit(ziptraininxx, ziptrainoutxx)  

        pred2_classes = neigh2.predict(ziptestin)

        pred2_classes = zipBinar.inverse_transform(pred2_classes) 



        if test_input0.shape == train_input0.shape :

            pr2 = pred2_classes.reshape(-1, train_output0.shape[1])

            

        if test_input0.shape != train_input0.shape :

            pr2 = pred2_classes.reshape(-1, test_input0.shape[1])

 

#   ########################################################################################################  



#   Sequential model >>> For predicted-3



    model = tf.keras.Sequential()

    model.add(layers.Dense(dim2, activation='relu', input_dim=dim))

    model.add(layers.Dense(dim1, activation='relu'))

    model.add(layers.Dense(dim, activation='softmax'))    

    model.summary()



    if aaa == 777:

        model.compile(loss='binary_crossentropy',

                      optimizer= 'adam',

                      metrics=['accuracy'])

    

        model.fit(ziptrainin, ziptrainout,

                  epochs=5,

                  batch_size=dim, verbose=0)    

        

        predicted_classes = model.predict_classes(ziptestin)

    

        if test_input0.shape == train_input0.shape :

            pr3 = predicted_classes.reshape(-1, train_output0.shape[1])

            

        if test_input0.shape != train_input0.shape :

            pr3 = predicted_classes.reshape(-1, test_input0.shape[1])

       

#   ########################################################################################################  



    if aaa != 777:    

        if contains_train == 2:            

            pr1 = train_output0

            pr2 = np.array(emi['train'][1]['output']) 

            pr3 = test_input0

            

        if contains_train != 2:            

            pr1 = train_output0

            pr2 = np.array(emi['train'][1]['output'])

            pr3 = np.array(emi['train'][2]['output'])    

            

#   ######################################################################################################## 



    if train_output0.shape == (1,1): 

        if dim > 4:

            dflist = list(df)             

            pr1 = np.array([[dflist[2]]])

            pr2 = np.array([[dflist[3]]])            

            pr3 = np.array([[dflist[4]]]) 



#   ########################################################################################################

#   ######################################################################################################## 



    def plot_task(task):



        cmap = colors.ListedColormap(

            ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

             '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

        norm = colors.Normalize(vmin=0, vmax=9)

        fig, axs = plt.subplots(1, 8, figsize=(15,15))

        axs[0].imshow(emi['train'][0]['input'], cmap=cmap, norm=norm)

        axs[0].axis('off')

        axs[0].set_title('Train Input')

        axs[1].imshow(emi['train'][0]['output'], cmap=cmap, norm=norm)

        axs[1].axis('off')

        axs[1].set_title('Train Output')    

        axs[2].imshow(emi['train'][1]['input'], cmap=cmap, norm=norm)

        axs[2].axis('off')

        axs[2].set_title('Train Input')

        axs[3].imshow(emi['train'][1]['output'], cmap=cmap, norm=norm)

        axs[3].axis('off')

        axs[3].set_title('Train Output')   

        axs[4].imshow(emi['test'][0]['input'], cmap=cmap, norm=norm)

        axs[4].axis('off')

        axs[4].set_title('Test Input')

        axs[5].imshow(pr1, cmap=cmap, norm=norm)

        axs[5].axis('off')

        axs[5].set_title('predicted-1')        

        axs[6].imshow(pr2, cmap=cmap, norm=norm)

        axs[6].axis('off')

        axs[6].set_title('predicted-2')                

        axs[7].imshow(pr3, cmap=cmap, norm=norm)

        axs[7].axis('off')

        axs[7].set_title('predicted-3')                

        plt.tight_layout()

        plt.show()

    

    plot_task(task)

    

#   ########################################################################################################

#   ######################################################################################################## 



    def flattener(pred):

        str_pred = str([row for row in pred])

        str_pred = str_pred.replace(', ', '')

        str_pred = str_pred.replace('[[', '|')

        str_pred = str_pred.replace('][', '|')

        str_pred = str_pred.replace(']]', '|')

        return str_pred    

        

    prd1 = pr1.tolist()

    prd1 = flattener(prd1)



    prd2 = pr2.tolist()

    prd2 = flattener(prd2)



    prd3 = pr3.tolist()

    prd3 = flattener(prd3)

    

    prd = prd1 + ' ' + prd2 + ' ' + prd3 + ' '

    prdn.append(prd)    

        

    s_tasks = test_tasks[i]    

    s_tasks = s_tasks.replace('.json', '_0')

    str_tasks.append(s_tasks)



#   ######################################################################################################## 



# You can write up to 5GB to the current directory (/kaggle/working/) 



    for output_id in submission.index:

        if s_tasks == output_id :         

            task_id = output_id.split('_')[0]

            f = str(test_path / str(task_id + '.json'))

            with open(f, 'r') as read_file:

                task = json.load(read_file)

            

            submission.loc[output_id, 'output'] = prd   

                        

            submission.to_csv(working_path / 'submission.csv')

            

            submission = pd.read_csv(working_path / 'submission.csv', index_col= 'output_id')



#   ######################################################################################################## 

#   ######################################################################################################## 



submission.to_csv('submission.csv')
