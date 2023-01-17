import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import glob

import pydicom

from tqdm import tqdm



import multiprocessing

from concurrent.futures import ThreadPoolExecutor



BASE_PATH = '../input/rsna-str-pulmonary-embolism-detection'

print(os.listdir('../input/rsna-str-pulmonary-embolism-detection'))
# train_df = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))

# test_df = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))
LABEL_COLUMNS = ['negative_exam_for_pe', 'indeterminate', 'chronic_pe', 'acute_and_chronic_pe', 'central_pe', 'leftsided_pe', 'rightsided_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1']

WEIGHTS       = [0.0736196319, 0.09202453988, 0.1042944785, 0.1042944785, 0.1877300613, 0.06257668712, 0.06257668712, 0.2346625767, 0.0782208589]
train_images = glob.glob(os.path.join(BASE_PATH, 'train/*/*/*.dcm'))

test_images = glob.glob(os.path.join(BASE_PATH, 'test/*/*/*.dcm'))
meta_columns = ['SOPInstanceUID', 

                'StudyInstanceUID', 

                'SeriesInstanceUID', 

                'InstanceNumber',

                'SOPClassUID', 

                'SliceThickness', 

                'KVP', 

#                 'GantryDetectorTilt', 

                'TableHeight', 

#                 'RotationDirection', 

                'XRayTubeCurrent', 

                'Exposure', 

#                 'ConvolutionKernel', 

                'PatientPosition', 

                'ImagePositionPatient', 

#                 'ImageOrientationPatient', 

                'FrameOfReferenceUID', 

#                 'SamplesPerPixel', 

#                 'PhotometricInterpretation', 

#                 'Rows', 

#                 'Columns', 

#                 'PixelSpacing', 

#                 'BitsAllocated', 

#                 'BitsStored', 

#                 'HighBit', 

#                 'PixelRepresentation', 

                'WindowCenter', 

                'WindowWidth', 

                'RescaleIntercept', 

                'RescaleSlope']
# Initialize dictionaries to collect the metadata

col_dict_train = {col: [] for col in meta_columns}

col_dict_test = {col: [] for col in meta_columns}
def get_first_of_dicom_field_as_int(x):

    """

    https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing

    """

    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)





def get_windowing(data):

    """

    https://www.kaggle.com/omission/eda-view-dicom-images-with-correct-windowing

    """

    dicom_fields = [data[('0028', '1050')].value,  # window center

                    data[('0028', '1051')].value,  # window width

                    data[('0028', '1052')].value,  # intercept

                    data[('0028', '1053')].value]  # slope

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
def process_train(img_path):

    dicom_object = pydicom.dcmread(img_path)

    for col in meta_columns: 

        window_center, window_width, intercept, slope = get_windowing(dicom_object)

        if col == 'WindowWidth':

            col_dict_train['WindowWidth'].append(window_width)

        elif col == 'WindowCenter':

            col_dict_train['WindowCenter'].append(window_center)

        elif col == 'RescaleIntercept':

            col_dict_train['RescaleIntercept'].append(intercept)

        elif col == 'RescaleSlope':

            col_dict_train['RescaleSlope'].append(slope)

        else:

            col_dict_train[col].append(str(getattr(dicom_object, col)))



def process_test(img_path):

    dicom_object = pydicom.dcmread(img_path)

    for col in meta_columns:

        window_center, window_width, intercept, slope = get_windowing(dicom_object)

        if col == 'WindowWidth':

            col_dict_test['WindowWidth'].append(window_width)

        elif col == 'WindowCenter':

            col_dict_test['WindowCenter'].append(window_center)

        elif col == 'RescaleIntercept':

            col_dict_test['RescaleIntercept'].append(intercept)

        elif col == 'RescaleSlope':

            col_dict_test['RescaleSlope'].append(slope)

        else:

            col_dict_test[col].append(str(getattr(dicom_object, col)))
%%time

# use multithreading to improve network or I/O bound tasks (such as read/write)

with ThreadPoolExecutor() as threads:

    threads.map(process_train, train_images)



meta_df_train = pd.DataFrame(col_dict_train)

del col_dict_train

gc.collect()
%%time

with ThreadPoolExecutor() as threads:

    threads.map(process_test, test_images)



meta_df_test = pd.DataFrame(col_dict_test)

del col_dict_test

gc.collect()
meta_df_train
meta_df_test
meta_df_train.to_csv('train_meta.csv', index=False)

meta_df_test.to_csv('test_meta.csv', index=False)