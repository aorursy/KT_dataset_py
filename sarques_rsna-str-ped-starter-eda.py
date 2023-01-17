import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(style = "darkgrid")

import pydicom as dcm

import matplotlib.cm as cm

import gc
df_train = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")

df_test = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/test.csv")

df_sub = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/sample_submission.csv")
df_train.head().T
df_train.shape
train_cols = ['pe_present_on_image', 'negative_exam_for_pe', 'qa_motion',

       'qa_contrast', 'flow_artifact', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',

       'leftsided_pe', 'chronic_pe', 'true_filling_defect_not_pe',

       'rightsided_pe', 'acute_and_chronic_pe', 'central_pe', 'indeterminate']
count = 0

for col in train_cols:

    if len(df_train[col].value_counts())==2:

        count += 1

if count==len(train_cols):

    print("All the features other than UID's are Binary features.")
def plot_grid(cols = train_cols):

    fig=plt.figure(figsize=(8, 22))

    columns = 2

    rows = 7

    for i in range(1, columns*rows + 1):

        col = cols[i-1]

        fig.add_subplot(rows, columns, i)

        df_train[col].value_counts().plot(kind = "bar", color = "Purple", alpha = 0.4)

        count_0 = df_train[col].value_counts()[0]

        count_1 = df_train[col].value_counts()[1]

        ratio = count_0/count_1

        plt.xlabel(f"Feature name: {col}\n Count 0: {count_0}\n Count 1: {count_1}\n Ratio(0:1): {ratio:.1f}:1")

    plt.tight_layout()

    plt.show()



plot_grid()
corr_mat = df_train[train_cols].corr()

mask = np.triu(np.ones_like(corr_mat, dtype=bool))

f, ax = plt.subplots(figsize=(14, 12))

sns.heatmap(corr_mat, mask = mask, cmap = "summer", annot = True, vmax = 0.3, square = False, linewidths = 0.5, center = 0)
def details_first_three(df = df_train):

    print(f"Number of unique entries in StudyInstanceUID: {len(df.StudyInstanceUID.value_counts())}")

    print(f"Number of unique entries in SeriesInstanceUID: {len(df.SeriesInstanceUID.value_counts())}")

    print(f"Number of unique entries in SOPInstanceUID: {len(df.SOPInstanceUID.value_counts())}")

details_first_three()
df_test.head(3)
df_test.shape
details_first_three(df_test)
df_sub.head(3)
df_sub.shape
df_sub["label_features"] = df_sub.id.apply(lambda x: "_".join(x.split("_")[1:]))
df_sub.label_features[df_sub.label_features == ""] = "UID"
df_sub.label_features.value_counts()
img_addr = ["../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf/00ac73cfc372.dcm", 

           "../input/rsna-str-pulmonary-embolism-detection/train/005df0f53614/5e0e0d0b7a65/081c2fa491a1.dcm",

           "../input/rsna-str-pulmonary-embolism-detection/train/0072baad76be/d555455a1dc2/096497b1da4e.dcm", 

           "../input/rsna-str-pulmonary-embolism-detection/train/00d4f4409f0c/38a51605b9ab/079e029c0d1a.dcm", 

           "../input/rsna-str-pulmonary-embolism-detection/test/00e7015490cb/291c07d4a7c0/09c25538116c.dcm", 

            "../input/rsna-str-pulmonary-embolism-detection/test/0227030d6278/599fccda6e2b/0c247bfd9c27.dcm", 

           "../input/rsna-str-pulmonary-embolism-detection/train/00102474a2db/c1a6d49ce580/06ce8f7a39ae.dcm", 

           "../input/rsna-str-pulmonary-embolism-detection/train/00102474a2db/c1a6d49ce580/0fd29873e8e4.dcm",

           "../input/rsna-str-pulmonary-embolism-detection/train/00617c9fe236/16ed05bf3395/01d00e27c5ac.dcm", 

            "../input/rsna-str-pulmonary-embolism-detection/test/08115e1b649d/f69e3f9c7067/10ba32beefb2.dcm"]
def plot_image_grid(addresses = img_addr):

    fig=plt.figure(figsize=(12, 12))

    columns = 5

    rows = 2

    for i in range(1, columns*rows + 1):

        addr = addresses[i-1]

        fig.add_subplot(rows, columns, i)

        plt.imshow(dcm.dcmread(addr).pixel_array)

        plt.axis("off")

    plt.tight_layout()

    plt.show()



plot_image_grid()
dicom_atts = ["SpecificCharacterSet","ImageType","SOPInstanceUID","Modality","Manufacturer", "ManufacturerModelName","PatientName","PatientID",

             "PatientSex","DeidentificationMethod","BodyPartExamined","SliceThickness", "KVP","SpacingBetweenSlices","DistanceSourceToDetector",

              "DistanceSourceToPatient","GantryDetectorTilt", "TableHeight","RotationDirection","XRayTubeCurrent","GeneratorPower",

              "FocalSpots","ConvolutionKernel","PatientPosition","RevolutionTime", "SingleCollimationWidth","TotalCollimationWidth","TableSpeed","TableFeedPerRotation",

              "SpiralPitchFactor", "StudyInstanceUID","SeriesInstanceUID","StudyID","InstanceNumber","PatientOrientation",

              "ImagePositionPatient","ImageOrientationPatient","FrameOfReferenceUID","PositionReferenceIndicator",

              "SliceLocation","SamplesPerPixel","PhotometricInterpretation", "Rows","Columns","PixelSpacing","BitsAllocated","BitsStored","HighBit",

              "PixelRepresentation","PixelPaddingValue","WindowCenter","WindowWidth","RescaleIntercept", "RescaleSlope","RescaleType"]



list_attributes = ["ImageType","ImagePositionPatient","ImageOrientationPatient","PixelSpacing"]



def dicom_metadata(folder_path):

    files = os.listdir(folder_path)

    patient_id = folder_path.split('/')[-1]

    

    ## Each row is an image file:

    base_data = {'Patient': [patient_id]*len(files), 'File': files}

    patient_df = pd.DataFrame(data=base_data)

    

    ## Add Columns by looping through DICOM attributes for each image file:

    slices = [dcm.dcmread(folder_path + '/' + s) for s in files] 

    for d in dicom_atts:

        attribute_i = []

        for s in slices:

            try:

                attribute_i.append(s[d].value)

            except:

                attribute_i.append(np.nan)

        patient_df[d] = attribute_i

        

    ## Store min pixel value for each image file 

    attribute_min_pixel = []

    for s in slices:

        try:

            mp = np.min(s.pixel_array.astype(np.int16).flatten())

        except:

            mp = np.nan

        attribute_min_pixel.append(mp)

    patient_df["MinPixelValue"] = attribute_min_pixel

  

    return patient_df
df = dicom_metadata("../input/rsna-str-pulmonary-embolism-detection/test/00268ff88746/75d23269adbd")
df.head(3)
print(f"CT Scan resolution is: {df.Rows.value_counts().index[0]}x{df.Columns.value_counts().index[0]}")
del df_train, df_test, df_sub, df, count, dicom_atts, list_attributes, img_addr, corr_mat, mask, train_cols

gc.collect()