import os

import gc

import pydicom # For accessing DICOM files

import numpy as np

import pandas as pd 

import random as rn

from tqdm import tqdm

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



# Importing Libraries for random forest classifier

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

%matplotlib inline



input_path = '../input/rsna-str-pulmonary-embolism-detection/'

img_trainpath = '../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf/'

img_testpath = '../input/rsna-str-pulmonary-embolism-detection/test/00268ff88746/75d23269adbd/'

train_csv = input_path + 'train.csv'

test_csv = input_path + 'test.csv'



seed = 1234

np.random.seed(seed)

rn.seed(seed)
train_csv_df = pd.read_csv(train_csv)

test_csv_df = pd.read_csv(test_csv)
train_images = os.listdir(img_trainpath)

test_images = os.listdir(img_testpath)

first_dicom_file = pydicom.dcmread(img_trainpath + train_images[0])
meta_cols=['ImageType','SOPClassUID','SOPInstanceUID',

               'Modality','SliceThickness','KVP',

               'TableHeight','RotationDirection','Exposure',

               'ConvolutionKernel','PatientPosition',

               'StudyInstanceUID','SeriesInstanceUID','SeriesNumber','InstanceNumber',

               'ImagePositionPatient','ImageOrientationPatient','PhotometricInterpretation',

               'Rows','Columns','PixelSpacing','BitsAllocated','BitsStored',

               'HighBit','PixelRepresentation','WindowCenter','WindowWidth',

               'RescaleIntercept','RescaleSlope','PixelData','SamplesPerPixel']

col_dict_train = {col: [] for col in meta_cols}

col_dict_test = {col: [] for col in meta_cols}
for img in tqdm(train_images): 

    dicom_object = pydicom.dcmread(img_trainpath + img)

    for col in meta_cols: 

        col_dict_train[col].append(str(getattr(dicom_object, col)))

meta_df_train = pd.DataFrame(col_dict_train)

del col_dict_train

gc.collect()
for img in tqdm(test_images): 

    dicom_object = pydicom.dcmread(img_testpath + img)

    for col in meta_cols: 

        col_dict_test[col].append(str(getattr(dicom_object, col)))

meta_df_test = pd.DataFrame(col_dict_test)

del col_dict_test

gc.collect()
for df in [meta_df_train, meta_df_test]:

    # ImagePositionPatient

    ipp1 = []

    ipp2 = []

    ipp3 = []

    for value in df['ImagePositionPatient'].fillna('[-9999,-9999,-9999]').values:

        value_list = eval(value)

        ipp1.append(float(value_list[0]))

        ipp2.append(float(value_list[1]))

        ipp3.append(float(value_list[2]))

    df['ImagePositionPatient_1'] = ipp1

    df['ImagePositionPatient_2'] = ipp2

    df['ImagePositionPatient_3'] = ipp3

    

    # ImageOrientationPatient

    iop1 = []

    iop2 = []

    iop3 = []

    iop4 = []

    iop5 = []

    iop6 = []

    # Fill missing values and collect all Image Orientation information

    for value in df['ImageOrientationPatient'].fillna('[-9999,-9999,-9999,-9999,-9999,-9999]').values:

        value_list = eval(value)

        iop1.append(float(value_list[0]))

        iop2.append(float(value_list[1]))

        iop3.append(float(value_list[2]))

        iop4.append(float(value_list[3]))

        iop5.append(float(value_list[4]))

        iop6.append(float(value_list[5]))

    df['ImageOrientationPatient_1'] = iop1

    df['ImageOrientationPatient_2'] = iop2

    df['ImageOrientationPatient_3'] = iop3

    df['ImageOrientationPatient_4'] = iop4

    df['ImageOrientationPatient_5'] = iop5

    df['ImageOrientationPatient_6'] = iop6

    

    # Pixel Spacing

    ps1 = []

    ps2 = []

    # Fill missing values and collect all pixal spacing features

    for value in df['PixelSpacing'].fillna('[-9999,-9999]').values:

        value_list = eval(value)

        ps1.append(float(value_list[0]))

        ps2.append(float(value_list[1]))

    df['PixelSpacing_1'] = ps1

    df['PixelSpacing_2'] = ps2
# Save to CSV

meta_df_train.to_csv('train_with_metadata.csv', index=False)

meta_df_test.to_csv('test_with_metadata.csv', index=False)
input_traincsv_data = './train_with_metadata.csv'

df = pd.read_csv(input_traincsv_data, header=None)

df
column_names=['ImageType','SOPClassUID','SOPInstanceUID',

               'Modality','SliceThickness','KVP',

               'TableHeight','RotationDirection','Exposure',

               'ConvolutionKernel','PatientPosition',

               'StudyInstanceUID','SeriesInstanceUID','SeriesNumber','InstanceNumber',

               'ImagePositionPatient','ImageOrientationPatient','PhotometricInterpretation',

               'Rows','Columns','PixelSpacing','BitsAllocated','BitsStored',

               'HighBit','PixelRepresentation','WindowCenter','WindowWidth',

               'RescaleIntercept','RescaleSlope','PixelData','SamplesPerPixel','ImagePositionPatient_1','ImagePositionPatient_2',

              'ImagePositionPatient_3','ImageOrientationPatient_1','ImageOrientationPatient_2','ImageOrientationPatient_3','ImageOrientationPatient_4',

              'ImageOrientationPatient_5','ImageOrientationPatient_6','PixelSpacing_1','PixelSpacing_2']

df.columns = column_names
y = df.drop(['ImageType','SOPClassUID','Modality','SliceThickness','KVP',

             'TableHeight','RotationDirection','Exposure','ConvolutionKernel','PatientPosition',

             'SeriesInstanceUID','SeriesNumber','InstanceNumber','ImagePositionPatient','ImageOrientationPatient',

             'PhotometricInterpretation','Rows','Columns','PixelSpacing','BitsAllocated','BitsStored','HighBit',

             'PixelRepresentation','WindowCenter','WindowWidth','RescaleIntercept','RescaleSlope','PixelData',

             'SamplesPerPixel','ImagePositionPatient_1','ImagePositionPatient_2','ImagePositionPatient_3',

             'ImageOrientationPatient_1','ImageOrientationPatient_2','ImageOrientationPatient_3','ImageOrientationPatient_4',

             'ImageOrientationPatient_5','ImageOrientationPatient_6','PixelSpacing_1','PixelSpacing_2'], axis = 1)
X = df.drop(['SOPInstanceUID','StudyInstanceUID','ImageType'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

X_train.shape, X_test.shape
y_train.shape, y_test.shape
import category_encoders as ce

x_encoder = ce.OrdinalEncoder(cols=['SOPClassUID','Modality','SliceThickness','KVP','TableHeight','RotationDirection','Exposure','ConvolutionKernel','PatientPosition','SeriesInstanceUID','SeriesNumber','InstanceNumber','ImagePositionPatient','ImageOrientationPatient','PhotometricInterpretation',

               'Rows','Columns','PixelSpacing','BitsAllocated','BitsStored','HighBit','PixelRepresentation','WindowCenter','WindowWidth',

               'RescaleIntercept','RescaleSlope','PixelData','SamplesPerPixel','ImagePositionPatient_1','ImagePositionPatient_2',

                                  'ImagePositionPatient_3','ImageOrientationPatient_1','ImageOrientationPatient_2','ImageOrientationPatient_3','ImageOrientationPatient_4',

                                  'ImageOrientationPatient_5','ImageOrientationPatient_6','PixelSpacing_1','PixelSpacing_2'])

X_train = x_encoder.fit_transform(X_train)

X_test = x_encoder.fit_transform(X_test)





y_encoder = ce.OrdinalEncoder(cols=['SOPInstanceUID','StudyInstanceUID'])

y_train = y_encoder.fit_transform(y_train)
import category_encoders as ce

y_encoder = ce.OrdinalEncoder(cols=['SOPInstanceUID','StudyInstanceUID'])

y_test = y_encoder.fit_transform(y_test)
yts = y_test.SOPInstanceUID.values

print(yts)
xts = X_test.values

print(xts)
xtr = X_train.values

print(xtr)
ytr=y_train.SOPInstanceUID.values

print(ytr)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

random_forest_classifier = RandomForestClassifier(random_state=0)

random_forest_classifier.fit(xtr, ytr)
random_forest_classifier.score(xtr,ytr)
#y_pred = random_forest_classifier.predict(xts)

random_forest_classifier.score(xts,yts)
feature_scores = pd.Series(random_forest_classifier.feature_importances_, index=X_train.columns).sort_values(ascending=False)

feature_scores
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

from sklearn import metrics
lrc = LinearRegression()
column_names=['ImageType','SOPClassUID','SOPInstanceUID',

               'Modality','SliceThickness','KVP',

               'TableHeight','RotationDirection','Exposure',

               'ConvolutionKernel','PatientPosition',

               'StudyInstanceUID','SeriesInstanceUID','SeriesNumber','InstanceNumber',

               'ImagePositionPatient','ImageOrientationPatient','PhotometricInterpretation',

               'Rows','Columns','PixelSpacing','BitsAllocated','BitsStored',

               'HighBit','PixelRepresentation','WindowCenter','WindowWidth',

               'RescaleIntercept','RescaleSlope','PixelData','SamplesPerPixel','ImagePositionPatient_1','ImagePositionPatient_2',

              'ImagePositionPatient_3','ImageOrientationPatient_1','ImageOrientationPatient_2','ImageOrientationPatient_3','ImageOrientationPatient_4',

              'ImageOrientationPatient_5','ImageOrientationPatient_6','PixelSpacing_1','PixelSpacing_2']

df.columns = column_names
df
y = df.drop(['ImageType','SOPClassUID','Modality','SliceThickness','KVP',

             'TableHeight','RotationDirection','Exposure','ConvolutionKernel','PatientPosition',

             'SeriesInstanceUID','SeriesNumber','InstanceNumber','ImagePositionPatient','ImageOrientationPatient',

             'PhotometricInterpretation','Rows','Columns','PixelSpacing','BitsAllocated','BitsStored','HighBit',

             'PixelRepresentation','WindowCenter','WindowWidth','RescaleIntercept','RescaleSlope','PixelData',

             'SamplesPerPixel','ImagePositionPatient_1','ImagePositionPatient_2','ImagePositionPatient_3',

             'ImageOrientationPatient_1','ImageOrientationPatient_2','ImageOrientationPatient_3','ImageOrientationPatient_4',

             'ImageOrientationPatient_5','ImageOrientationPatient_6','PixelSpacing_1','PixelSpacing_2'], axis = 1)
X = df.drop(['SOPInstanceUID','StudyInstanceUID','ImageType'],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

X_train.shape, X_test.shape
import category_encoders as ce

x_encoder = ce.OrdinalEncoder(cols=['SOPClassUID','Modality','SliceThickness','KVP','TableHeight','RotationDirection','Exposure','ConvolutionKernel','PatientPosition','SeriesInstanceUID','SeriesNumber','InstanceNumber','ImagePositionPatient','ImageOrientationPatient','PhotometricInterpretation',

               'Rows','Columns','PixelSpacing','BitsAllocated','BitsStored','HighBit','PixelRepresentation','WindowCenter','WindowWidth',

               'RescaleIntercept','RescaleSlope','PixelData','SamplesPerPixel','ImagePositionPatient_1','ImagePositionPatient_2',

                                  'ImagePositionPatient_3','ImageOrientationPatient_1','ImageOrientationPatient_2','ImageOrientationPatient_3','ImageOrientationPatient_4',

                                  'ImageOrientationPatient_5','ImageOrientationPatient_6','PixelSpacing_1','PixelSpacing_2'])

X_train = x_encoder.fit_transform(X_train)

X_test = x_encoder.fit_transform(X_test)





y_encoder = ce.OrdinalEncoder(cols=['SOPInstanceUID','StudyInstanceUID'])

y_train = y_encoder.fit_transform(y_train)
import category_encoders as ce

y_encoder = ce.OrdinalEncoder(cols=['SOPInstanceUID','StudyInstanceUID'])

y_test = y_encoder.fit_transform(y_test)
Xtr = X_train.values

print(Xtr)
Xts = X_test.values

print(Xts)
Ytr = y_train.SOPInstanceUID.values

print(ytr)
Yts= y_test.SOPInstanceUID.values

print(Yts)
lrc.fit(X_train,y_train)
y_pred = lrc.predict(Xts)

print(y_pred)
print(metrics.mean_squared_error(Yts,y_pred))
print(metrics.mean_absolute_error(Yts,y_pred))