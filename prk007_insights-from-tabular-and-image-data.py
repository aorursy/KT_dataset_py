import numpy as np, pandas as pd, pydicom as dcm

import matplotlib.pyplot as plt, seaborn as sns

import os, glob
train_df = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")

test_df = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/test.csv")
print(f"Total no. of studies = {train_df['StudyInstanceUID'].unique().shape[0]}")

print(f"Total no. of series = {train_df['SeriesInstanceUID'].unique().shape[0]}")
x = train_df.groupby('StudyInstanceUID').mean()

x.sum()
# for i in ['qa_motion', 'qa_contrast', 'true_filling_defect_not_pe', 'flow_artifact']

print(f"Total indeterminate using qa_motion and qa_contrast = {train_df[(train_df['qa_motion']== 1.0) | (train_df['qa_contrast']== 1.0)].shape[0]}")

print(f"Total indeterminate directly = {train_df[(train_df['indeterminate']== 1.0)].shape[0]}")



print(f"Total indeterminate and has PE = {train_df[(train_df['indeterminate']== 1.0) & (train_df['pe_present_on_image']== 1.0)].shape[0]}")

print(f"Total Flow artifact and has PE = {train_df[(train_df['indeterminate']== 1.0) & (train_df['flow_artifact']== 1.0)].shape[0]}")

print(f"Total True filling defect not PE and has PE = {train_df[(train_df['indeterminate']== 1.0) & (train_df['true_filling_defect_not_pe']== 1.0)].shape[0]}")

import pydicom as dicom

# Load the scans in given folder path

def load_scan(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices



def get_pixels_hu(slices):

    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)



    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

    for slice_number in range(len(slices)):

        

        intercept = slices[slice_number].RescaleIntercept

        slope = slices[slice_number].RescaleSlope

        

        if slope != 1:

            image[slice_number] = slope * image[slice_number].astype(np.float64)

            image[slice_number] = image[slice_number].astype(np.int16)

            

        image[slice_number] += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)
first_patient = load_scan('../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf')

first_patient_pixels = get_pixels_hu(first_patient)



plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)

plt.show()
fig, plots = plt.subplots(8, 10, sharex='col', sharey='row', figsize=(20, 16))

for i in range(80):

    plots[i // 10, i % 10].axis('off')

    plots[i // 10, i % 10].imshow(first_patient_pixels[i], cmap=plt.cm.bone) 