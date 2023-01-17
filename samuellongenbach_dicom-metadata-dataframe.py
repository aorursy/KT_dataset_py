##

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

##

import pydicom 
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

test_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")



train_df.head(5)
print("# of Patients in Train: ",len(np.unique(train_df["Patient"])))

print("# of Patients in Test: ",len(np.unique(test_df["Patient"])))

print("Train/Test overlap?: ",len(np.intersect1d(train_df["Patient"],test_df["Patient"])))
# Base directory for Train .dcm files

osic_dir = "../input/osic-pulmonary-fibrosis-progression/train/"



train_df["Path"] = osic_dir + train_df["Patient"] 



# Calculate how many CT images each patient has

train_df["CT_images"] = 0



for k, path in enumerate(train_df["Path"]):

    train_df["CT_images"][k] = len(os.listdir(path))



train_df.head(5)
# CT Scans per Patient

data = train_df.groupby(by="Patient")["CT_images"].first().reset_index(drop=False)



# Sort by number of CT Scans

data = data.sort_values(['CT_images']).reset_index(drop=True)

print("Minimum number of CT images: {}".format(data["CT_images"].min()), "\n" +

      "Maximum number of CT images: {}".format(data["CT_images"].max()), "\n" +

      "Median number of CT images: {}".format(data["CT_images"].median()))



# Plot

plt.figure(figsize = (16, 6))

p = sns.barplot(data["Patient"], data["CT_images"], color="darkgreen")

plt.axvline(x=85, color="lightgreen", linestyle='--', lw=3)



plt.title("Number of CT images in baseline for each Patient", fontsize = 17)

plt.xlabel('Patient', fontsize=14)

plt.ylabel('Frequency', fontsize=14)



plt.text(86, 850, "Median=98", fontsize=13)



p.axes.get_xaxis().set_visible(False);
path = "../input/osic-pulmonary-fibrosis-progression/train/ID00122637202216437668965"



slices = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path)] 

slices.sort(key = lambda x: int(x.InstanceNumber)) 



print("Patient: ","ID00122637202216437668965")

print("Image/Slice: ",slices[36]["InstanceNumber"].value)

slices[36]
dicom_atts = ["SpecificCharacterSet","ImageType","SOPInstanceUID","Modality","Manufacturer","ManufacturerModelName","PatientName","PatientID",

             "PatientSex","DeidentificationMethod","BodyPartExamined","SliceThickness","KVP","SpacingBetweenSlices","DistanceSourceToDetector","DistanceSourceToPatient","GantryDetectorTilt",

             "TableHeight","RotationDirection","XRayTubeCurrent","GeneratorPower","FocalSpots","ConvolutionKernel","PatientPosition","RevolutionTime","SingleCollimationWidth","TotalCollimationWidth","TableSpeed","TableFeedPerRotation","SpiralPitchFactor",

              "StudyInstanceUID","SeriesInstanceUID","StudyID","InstanceNumber","PatientOrientation","ImagePositionPatient","ImageOrientationPatient","FrameOfReferenceUID","PositionReferenceIndicator","SliceLocation","SamplesPerPixel","PhotometricInterpretation",

             "Rows","Columns","PixelSpacing","BitsAllocated","BitsStored","HighBit","PixelRepresentation","PixelPaddingValue","WindowCenter","WindowWidth","RescaleIntercept","RescaleSlope","RescaleType"]



list_attributes = ["ImageType","ImagePositionPatient","ImageOrientationPatient","PixelSpacing"]



def Metadata_for_Patient(folder_path):

    files = os.listdir(folder_path)

    patient_id = folder_path.split('/')[-1]

    

    ## Each row is an image file:

    base_data = {'Patient': [patient_id]*len(files), 'File': files}

    patient_df = pd.DataFrame(data=base_data)

    

    ## Add Columns by looping through DICOM attributes for each image file:

    slices = [pydicom.dcmread(folder_path + '/' + s) for s in files] 

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
Metadata_for_Patient(path).head()
DICOM_Meta_df = pd.DataFrame()



## For all 176 Patient Folders: 

unique_patient_df = train_df.groupby(by="Patient").first()

for pth in unique_patient_df["Path"]:

    temp_df = Metadata_for_Patient(pth)

    DICOM_Meta_df = pd.concat([DICOM_Meta_df,temp_df],ignore_index=True)

    

## SAVE:

DICOM_Meta_df.to_pickle("DICOM_Metadata.pkl")



## LOAD:

#load_df = pd.read_pickle("DICOM_Metadata.pkl")
## All Images:

print("Shape: ", DICOM_Meta_df.shape)

DICOM_Meta_df.info()
## All Patients:

unique_meta_df = DICOM_Meta_df.groupby(by="Patient").first()

unique_meta_df.info()
unique_meta_df.describe()
issues = DICOM_Meta_df[DICOM_Meta_df["MinPixelValue"].isna()]

issues
issues["Patient"].value_counts()
issue_path = "../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/6.dcm"



fff = pydicom.dcmread(issue_path) 

fff.pixel_array