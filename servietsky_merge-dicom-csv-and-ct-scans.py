import numpy as np

import pandas as pd

from tqdm.notebook import tqdm 

import gc

import glob, os

import pydicom

from PIL import Image

import matplotlib.pyplot as plt

import lightgbm as lgb

from sklearn.decomposition import PCA

from bayes_opt import BayesianOptimization

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import NMF

pd.set_option('display.max_columns', 500)
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in tqdm(df.columns):

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
# train_scans = pd.read_pickle('../input/ct-scans-to-dataframe/train_out.pkl')

# tmp = train_scans[['Instance','Patient']]

# train_scans = pd.get_dummies(train_scans[[x for x in train_scans.columns if x not in ['Instance','Patient']]], drop_first = True)



# pca = NMF(n_components=2000)

# train_scans = pca.fit_transform(train_scans)
train_inf = pd.read_pickle('../input/osic-transform-dicom-into-dataframe/output_data.pkl')

train_scans = pd.read_pickle('../input/ct-scans-to-dataframe/train_out.pkl')



test_inf = pd.read_pickle('../input/osic-transform-dicom-into-dataframe/output_data_test.pkl')

test_scans  = pd.read_pickle('../input/ct-scans-to-dataframe/test_out.pkl')
train_scans.columns = train_scans.columns.astype(np.str)

pca = PCA(n_components=200)

new_pixels = pca.fit_transform(train_scans.loc[:, '0': '7395'])



train_scans.drop(train_scans.loc[:, '0': '7395'].columns, axis = 1, inplace = True)



train_scans = pd.merge(train_scans, pd.DataFrame(new_pixels), left_index=True, right_index=True)
test_scans.columns = test_scans.columns.astype(np.str)

pca = PCA(n_components=200)

new_pixels = pca.fit_transform(test_scans.loc[:, '0': '7395'])



test_scans.drop(test_scans.loc[:, '0': '7395'].columns, axis = 1, inplace = True)



test_scans = pd.merge(test_scans, pd.DataFrame(new_pixels), left_index=True, right_index=True)
col_to_drop = ['Modality', 'ImageType', 'SOPInstanceUID', 'PatientName', 'PatientID', 'PatientSex', 'DeidentificationMethod', 'BodyPartExamined'

              ,'GantryDetectorTilt', 'RotationDirection','StudyInstanceUID', 'SeriesInstanceUID', 'StudyID', 'ImagePositionPatient'

              , 'ImageOrientationPatient', 'FrameOfReferenceUID', 'SamplesPerPixel', 'PhotometricInterpretation', 'PixelSpacing', 'BitsAllocated'

              , 'RescaleSlope'

#               , 'Patient'

              ]

train_inf = train_inf[[x for x in train_inf.columns if x not in col_to_drop]]

test_inf = test_inf[[x for x in test_inf.columns if x not in col_to_drop]]
train_inf.loc[train_inf.ManufacturerModelName == '','ManufacturerModelName'] = 'unk'

train_inf.SliceThickness = train_inf.SliceThickness.astype('float')

train_inf.KVP = train_inf.KVP.astype('float')

train_inf.SpacingBetweenSlices = train_inf.SpacingBetweenSlices.astype('float')

train_inf.TableHeight = train_inf.TableHeight.astype('float')

train_inf.XRayTubeCurrent = train_inf.XRayTubeCurrent.astype('int')

train_inf.InstanceNumber = train_inf.InstanceNumber.astype('int')

train_inf.loc[train_inf.PositionReferenceIndicator == '','PositionReferenceIndicator'] = 'unk'

train_inf.SliceLocation = train_inf.SliceLocation.astype('float')

train_inf.Rows = train_inf.Rows.astype('int')

train_inf.Columns = train_inf.Columns.astype('int')

train_inf.BitsStored = train_inf.BitsStored.astype('int')

train_inf.HighBit = train_inf.HighBit.astype('int')

train_inf.loc[train_inf.WindowCenter == '[-500, 40]','WindowCenter'] = '-500'

train_inf.loc[train_inf.WindowCenter == '-500.0','WindowCenter'] = '-500'

train_inf.WindowCenter = train_inf.WindowCenter.astype('int')

train_inf.loc[train_inf.WindowWidth == '[1500, 350]','WindowWidth'] = '1500'

train_inf.loc[train_inf.WindowWidth == '-1500.0','WindowWidth'] = '-1500'

train_inf.WindowWidth = train_inf.WindowWidth.astype('int')

train_inf.loc[train_inf.RescaleIntercept == '0.','RescaleIntercept'] = '0'

train_inf.loc[train_inf.RescaleIntercept == '-1024.','RescaleIntercept'] = '-1024'

train_inf.loc[train_inf.RescaleIntercept == '-1024.0','RescaleIntercept'] = '-1024'

train_inf.RescaleIntercept = train_inf.RescaleIntercept.astype('int')
test_inf.loc[test_inf.ManufacturerModelName == '','ManufacturerModelName'] = 'unk'

test_inf.SliceThickness = test_inf.SliceThickness.astype('float')

test_inf.KVP = test_inf.KVP.astype('float')

test_inf.SpacingBetweenSlices = test_inf.SpacingBetweenSlices.astype('float')

test_inf.TableHeight = test_inf.TableHeight.astype('float')

test_inf.XRayTubeCurrent = test_inf.XRayTubeCurrent.astype('int')

test_inf.InstanceNumber = test_inf.InstanceNumber.astype('int')

test_inf.loc[test_inf.PositionReferenceIndicator == '','PositionReferenceIndicator'] = 'unk'

test_inf.SliceLocation = test_inf.SliceLocation.astype('float')

test_inf.Rows = test_inf.Rows.astype('int')

test_inf.Columns = test_inf.Columns.astype('int')

test_inf.BitsStored = test_inf.BitsStored.astype('int')

test_inf.HighBit = test_inf.HighBit.astype('int')

test_inf.loc[test_inf.WindowCenter == '[-500, 40]','WindowCenter'] = '-500'

test_inf.loc[test_inf.WindowCenter == '-500.0','WindowCenter'] = '-500'

test_inf.WindowCenter = test_inf.WindowCenter.astype('int')

test_inf.loc[test_inf.WindowWidth == '[1500, 350]','WindowWidth'] = '1500'

test_inf.loc[test_inf.WindowWidth == '-1500.0','WindowWidth'] = '-1500'

test_inf.WindowWidth = test_inf.WindowWidth.astype('int')

test_inf.loc[test_inf.RescaleIntercept == '0.','RescaleIntercept'] = '0'

test_inf.loc[test_inf.RescaleIntercept == '-1024.','RescaleIntercept'] = '-1024'

test_inf.loc[test_inf.RescaleIntercept == '-1024.0','RescaleIntercept'] = '-1024'

test_inf.RescaleIntercept = test_inf.RescaleIntercept.astype('int')
train_inf = reduce_mem_usage(train_inf)

test_inf = reduce_mem_usage(test_inf)
train_final = train_inf.merge(train_scans, left_on=['Patient', 'InstanceNumber'], right_on=['Patient', 'Instance'])
train_final.info()
train_final.drop('InstanceNumber', axis = 1, inplace = True)

train_final.to_pickle("train_final.pkl")
test_final = test_inf.merge(test_scans, left_on=['Patient', 'InstanceNumber'], right_on=['Patient', 'Instance'])
test_final[['FVC','Percent']] = -1

test_final.info()
test_final.drop('InstanceNumber', axis = 1, inplace = True)

test_final.to_pickle("test_final.pkl")