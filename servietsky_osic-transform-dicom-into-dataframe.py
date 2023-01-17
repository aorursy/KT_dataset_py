import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pydicom

from pydicom import dcmread

from pydicom.data import get_testdata_files

import glob, os

from collections import defaultdict

import tqdm

import gc

import seaborn as sns

import ast
col_name = ['ImageType', 'SOPInstanceUID', 'Modality', 'Manufacturer','ManufacturerModelName', 'PatientName', 'PatientID', 'PatientSex', 'DeidentificationMethod',

'BodyPartExamined', 'SliceThickness', 'KVP', 'SpacingBetweenSlices', 'GantryDetectorTilt', 'TableHeight', 'RotationDirection', 'XRayTubeCurrent', 'ConvolutionKernel',                 

'PatientPosition', 'StudyInstanceUID', 'SeriesInstanceUID', 'StudyID', 'InstanceNumber', 'ImagePositionPatient', 'ImageOrientationPatient', 'FrameOfReferenceUID',          

'PositionReferenceIndicator', 'SliceLocation', 'SamplesPerPixel', 'PhotometricInterpretation', 'Rows', 'Columns', 'PixelSpacing','BitsAllocated','BitsStored',                    

'HighBit', 'PixelRepresentation','WindowCenter','WindowWidth','RescaleIntercept','RescaleSlope']



df = pd.DataFrame(columns=col_name)

my_dict = defaultdict(list)



for name in tqdm.tqdm(glob.glob('/kaggle/input/osic-pulmonary-fibrosis-progression/train/*/*')):

    ds = pydicom.read_file(name)

    for i in col_name :

        if i in ds :

            my_dict[i].append(str(ds[i].value))

        else:

            my_dict[i].append(np.nan)

    df = pd.concat([df, pd.DataFrame(my_dict)], ignore_index = True)

    del my_dict

    my_dict = defaultdict(list)

gc.collect()



for name in tqdm.tqdm(glob.glob('/kaggle/input/osic-pulmonary-fibrosis-progression/test/*/*')):

    ds = pydicom.read_file(name)

    for i in col_name :

        if i in ds :

            my_dict[i].append(str(ds[i].value))

        else:

            my_dict[i].append(np.nan)

    df = pd.concat([df, pd.DataFrame(my_dict)], ignore_index = True)

    del my_dict

    my_dict = defaultdict(list)

gc.collect()

#     ds = pydicom.read_file(name)

#     df = pd.DataFrame(ds.values())

#     df[0] = df[0].apply(lambda x: pydicom.dataelem.DataElement_from_raw(x) if isinstance(x, pydicom.dataelem.RawDataElement) else x)

#     df['name'] = df[0].apply(lambda x: x.name)

#     df['value'] = df[0].apply(lambda x: x.value)

#     df = df[['name', 'value']]



#     df = df.set_index('name').T.reset_index(drop=True)

#     df.drop('Pixel Data', axis = 1, inplace = True)
train = pd.concat([pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv'), pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')], ignore_index = True)
# train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

Data = df.merge(train , left_on='PatientID', right_on='Patient')



type_dict_all = ['ORIGINAL', 'PRIMARY', 'AXIAL', 'CT_SOM5 SPI', 'HELIX', 'CT_SOM5 SEQ', 'SECONDARY', 'DERIVED', 'JP2K LOSSY 6:1', 'VOLUME', 'OTHER', 'CSA MPR', 'CSAPARALLEL', 

                'CSA RESAMPLED', 'REFORMATTED', 'AVERAGE', 'CT_SOM7 SPI DUAL', 'STD', 'SNRG', 'DET_AB']

for i in type_dict_all:

    Data[i] = np.nan

for index, row in tqdm.tqdm(Data.iterrows()):

    for i in type_dict_all:

        if i in Data.loc[index, 'ImageType'] :

            Data.loc[index, i] = 1

            

Data.fillna(0, inplace = True)



Data.loc[Data['ImagePositionPatient'] == 0, 'ImagePositionPatient'] = '[0,0,0]'

Data['ImagePositionPatient'] = Data['ImagePositionPatient'].apply(ast.literal_eval)

Data[['ImagePositionPatient_x','ImagePositionPatient_y', 'ImagePositionPatient_z']] = pd.DataFrame(Data.ImagePositionPatient.tolist(), index= Data.index)



Data.loc[Data['ImageOrientationPatient'] == 0, 'ImageOrientationPatient'] = '[0,0,0,0,0,0]'

Data['ImageOrientationPatient'] = Data['ImageOrientationPatient'].apply(ast.literal_eval)

Data[['ImageOrientationPatient_a','ImageOrientationPatient_b', 'ImageOrientationPatient_c', 'ImageOrientationPatient_d', 'ImageOrientationPatient_e', 'ImageOrientationPatient_f']] = pd.DataFrame(Data.ImageOrientationPatient.tolist(), index= Data.index)



tmp1 = Data[['ImagePositionPatient_x','ImagePositionPatient_y', 'ImagePositionPatient_z', 'ImageOrientationPatient_a','ImageOrientationPatient_b', 'ImageOrientationPatient_c']]

tmp1.columns = ['x','y','z','a','b','c']



tmp1['Cos'] = 'red'

tmp2 = Data[['ImagePositionPatient_x','ImagePositionPatient_y', 'ImagePositionPatient_z', 'ImageOrientationPatient_d','ImageOrientationPatient_e', 'ImageOrientationPatient_f']]

tmp2.columns = ['x','y','z','a','b','c']

tmp2['Cos'] = 'blue'



cos = pd.concat([tmp1, tmp2], ignore_index = True)

cos['width'] = 10



cos[['a','b','c']] = cos[['a','b','c']] * 200



Data.loc[Data['PixelSpacing'] == 0, 'PixelSpacing'] = '[0,0]'

Data['PixelSpacing'] = Data['PixelSpacing'].apply(ast.literal_eval)

Data[['PixelSpacing_row','PixelSpacing_column']] = pd.DataFrame(Data.PixelSpacing.tolist(), index= Data.index)



Data.to_pickle('output_data.pkl')
Data.head(10)
col_name = ['ImageType', 'SOPInstanceUID', 'Modality', 'Manufacturer','ManufacturerModelName', 'PatientName', 'PatientID', 'PatientSex', 'DeidentificationMethod',

'BodyPartExamined', 'SliceThickness', 'KVP', 'SpacingBetweenSlices', 'GantryDetectorTilt', 'TableHeight', 'RotationDirection', 'XRayTubeCurrent', 'ConvolutionKernel',                 

'PatientPosition', 'StudyInstanceUID', 'SeriesInstanceUID', 'StudyID', 'InstanceNumber', 'ImagePositionPatient', 'ImageOrientationPatient', 'FrameOfReferenceUID',          

'PositionReferenceIndicator', 'SliceLocation', 'SamplesPerPixel', 'PhotometricInterpretation', 'Rows', 'Columns', 'PixelSpacing','BitsAllocated','BitsStored',                    

'HighBit', 'PixelRepresentation','WindowCenter','WindowWidth','RescaleIntercept','RescaleSlope']



df = pd.DataFrame(columns=col_name)

my_dict = defaultdict(list)



for name in tqdm.tqdm(glob.glob('/kaggle/input/osic-pulmonary-fibrosis-progression/test/*/*.dcm')):

    ds = pydicom.read_file(name)

    for i in col_name :

        if i in ds :

            my_dict[i].append(str(ds[i].value))

        else:

            my_dict[i].append(np.nan)

    df = pd.concat([df, pd.DataFrame(my_dict)], ignore_index = True)

    del my_dict

    my_dict = defaultdict(list)

gc.collect()
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub =  sub[['Patient','Weeks','Patient_Week']]

test = sub.merge(test.drop('Weeks', axis=1), on="Patient")



Data = df.merge(test , left_on='PatientID', right_on='Patient')



type_dict_all = ['ORIGINAL', 'PRIMARY', 'AXIAL', 'CT_SOM5 SPI', 'HELIX', 'CT_SOM5 SEQ', 'SECONDARY', 'DERIVED', 'JP2K LOSSY 6:1', 'VOLUME', 'OTHER', 'CSA MPR', 'CSAPARALLEL', 

                'CSA RESAMPLED', 'REFORMATTED', 'AVERAGE', 'CT_SOM7 SPI DUAL', 'STD', 'SNRG', 'DET_AB']

for i in type_dict_all:

    Data[i] = np.nan

for index, row in tqdm.tqdm(Data.iterrows()):

    for i in type_dict_all:

        if i in Data.loc[index, 'ImageType'] :

            Data.loc[index, i] = 1
Data.fillna(0, inplace = True)



Data.loc[Data['ImagePositionPatient'] == 0, 'ImagePositionPatient'] = '[0,0,0]'

Data['ImagePositionPatient'] = Data['ImagePositionPatient'].apply(ast.literal_eval)

Data[['ImagePositionPatient_x','ImagePositionPatient_y', 'ImagePositionPatient_z']] = pd.DataFrame(Data.ImagePositionPatient.tolist(), index= Data.index)



Data.loc[Data['ImageOrientationPatient'] == 0, 'ImageOrientationPatient'] = '[0,0,0,0,0,0]'

Data['ImageOrientationPatient'] = Data['ImageOrientationPatient'].apply(ast.literal_eval)

Data[['ImageOrientationPatient_a','ImageOrientationPatient_b', 'ImageOrientationPatient_c', 'ImageOrientationPatient_d', 'ImageOrientationPatient_e', 'ImageOrientationPatient_f']] = pd.DataFrame(Data.ImageOrientationPatient.tolist(), index= Data.index)



tmp1 = Data[['ImagePositionPatient_x','ImagePositionPatient_y', 'ImagePositionPatient_z', 'ImageOrientationPatient_a','ImageOrientationPatient_b', 'ImageOrientationPatient_c']]

tmp1.columns = ['x','y','z','a','b','c']



tmp1['Cos'] = 'red'

tmp2 = Data[['ImagePositionPatient_x','ImagePositionPatient_y', 'ImagePositionPatient_z', 'ImageOrientationPatient_d','ImageOrientationPatient_e', 'ImageOrientationPatient_f']]

tmp2.columns = ['x','y','z','a','b','c']

tmp2['Cos'] = 'blue'



cos = pd.concat([tmp1, tmp2], ignore_index = True)

cos['width'] = 10



cos[['a','b','c']] = cos[['a','b','c']] * 200



Data.loc[Data['PixelSpacing'] == 0, 'PixelSpacing'] = '[0,0]'

Data['PixelSpacing'] = Data['PixelSpacing'].apply(ast.literal_eval)

Data[['PixelSpacing_row','PixelSpacing_column']] = pd.DataFrame(Data.PixelSpacing.tolist(), index= Data.index)



Data.to_pickle('output_data_test.pkl')
Data.head()