# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D



import os

import gc

import time

from IPython.display import clear_output

from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import ModelCheckpoint as MC

from tensorflow.keras import backend as K



# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print('Reading test data...')

test = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/test.csv")

print(test.shape)

test.head()
def convert_to_rgb(array):

    array = array.reshape((512, 512, 1))

    return np.stack([array, array, array], axis=2).reshape((512, 512, 3))

    

def custom_dcom_image_generator(batch_size, dataset, test=False, debug=False):

    

    fnames = dataset[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']]

    

    if not test:

        Y = dataset[['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',

                     'leftsided_pe', 'chronic_pe', 'rightsided_pe',

                     'acute_and_chronic_pe', 'central_pe', 'indeterminate']]

        prefix = 'input/rsna-str-pulmonary-embolism-detection/train'

        

    else:

        prefix = 'input/rsna-str-pulmonary-embolism-detection/test'

    

    X = []

    batch = 0

    for st, sr, so in fnames.values:

        if debug:

            print(f"Current file: ../{prefix}/{st}/{sr}/{so}.dcm")



        dicom = get_img(f"../{prefix}/{st}/{sr}/{so}.dcm")

        image = convert_to_rgb(dicom)

        X.append(image)

        

        del st, sr, so

        

        if len(X) == batch_size:

            if test:

                yield np.array(X)

                del X

            else:

                yield np.array(X), Y[batch*batch_size:(batch+1)*batch_size].values

                del X

                

            gc.collect()

            X = []

            batch += 1

        

    if test:

        yield np.array(X)

    else:

        yield np.array(X), Y[batch*batch_size:(batch+1)*batch_size].values

        del Y

    del X

    gc.collect()

    return
import vtk

from vtk.util import numpy_support

import cv2



reader = vtk.vtkDICOMImageReader()

def get_img(path):

    reader.SetFileName(path)

    reader.Update()

    _extent = reader.GetDataExtent()

    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]



    ConstPixelSpacing = reader.GetPixelSpacing()

    imageData = reader.GetOutput()

    pointData = imageData.GetPointData()

    arrayData = pointData.GetArray(0)

    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)

    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')

    ArrayDicom = cv2.resize(ArrayDicom,(512,512))

    return ArrayDicom

from tensorflow.keras import backend as K



predictions = {}

stopper = 3600 * 4 #4 hours limit for prediction

pred_start_time = time.time()



p, c = time.time(), time.time()

batch_size = 1000

    

l = 0

n = test.shape[0]



for x in custom_dcom_image_generator(batch_size, test, True, False):

    clear_output(wait=True)

    model = load_model("../input/dataset/pe_detection_model_.h5")

    preds = model.predict(x, batch_size=8, verbose=1)

    

    try:

        for key in preds.keys():

            predictions[key] += preds[key].flatten().tolist()

            

    except Exception as e:

        print(e)

        for key in preds.keys():

            predictions[key] = preds[key].flatten().tolist()

            

    l = (l+batch_size)%n

    print('Total predicted:', len(predictions['indeterminate']),'/', n)

    p, c = c, time.time()

    print("One batch time: %.2f seconds" %(c-p))

    print("ETA: %.2f" %((n-l)*(c-p)/batch_size))

    

    if c - pred_start_time >= stopper:

        print("Time's up!")

        break

    

    del model

    K.clear_session()

    

    del x, preds

    gc.collect()
for key in predictions.keys():

    print(key, np.array(predictions[key]).shape)
test_ids = []

for v in test.StudyInstanceUID:

    if v not in test_ids:

        test_ids.append(v)

        

test_preds = test.copy()

test_preds = pd.concat([test_preds, pd.DataFrame(predictions)], axis=1)

test_preds
IDS = []

labels = []



for label in ['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',

                 'leftsided_pe', 'chronic_pe', 'rightsided_pe',

                 'acute_and_chronic_pe', 'central_pe', 'indeterminate']:

    for key in test_ids:

        temp = test_preds.loc[test_preds.StudyInstanceUID==key]

        

        IDS.append('_'.join([key, label]))

        labels.append(np.max(temp[label]))
IDS += test_preds.SOPInstanceUID.tolist()

labels += test_preds['negative_exam_for_pe'].tolist()



sub = pd.DataFrame({"id":IDS, 'label':labels})

sub
sub.fillna(0.2, inplace=True)

sub.to_csv('submission.csv', index=False)