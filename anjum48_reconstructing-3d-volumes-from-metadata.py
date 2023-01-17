import os

import pandas as pd

import numpy as np

import pydicom

import matplotlib.pylab as plt

from tqdm import tqdm_notebook

%matplotlib inline



data_path = "../input/rsna-intracranial-hemorrhage-detection"

metadata_path = "../input/rsna-ich-metadata"

os.listdir(metadata_path)
train_df = pd.read_csv(f'{data_path}/stage_1_train.csv').drop_duplicates()

train_df['ImageID'] = train_df['ID'].str.slice(stop=12)

train_df['Diagnosis'] = train_df['ID'].str.slice(start=13)

train_labels = train_df.pivot(index="ImageID", columns="Diagnosis", values="Label")

train_labels.head()
def get_metadata(image_dir):



    labels = [

        'BitsAllocated', 'BitsStored', 'Columns', 'HighBit', 

        'ImageOrientationPatient_0', 'ImageOrientationPatient_1', 'ImageOrientationPatient_2',

        'ImageOrientationPatient_3', 'ImageOrientationPatient_4', 'ImageOrientationPatient_5',

        'ImagePositionPatient_0', 'ImagePositionPatient_1', 'ImagePositionPatient_2',

        'Modality', 'PatientID', 'PhotometricInterpretation', 'PixelRepresentation',

        'PixelSpacing_0', 'PixelSpacing_1', 'RescaleIntercept', 'RescaleSlope', 'Rows', 'SOPInstanceUID',

        'SamplesPerPixel', 'SeriesInstanceUID', 'StudyID', 'StudyInstanceUID', 

        'WindowCenter', 'WindowWidth', 'Image',

    ]



    data = {l: [] for l in labels}



    for image in tqdm_notebook(os.listdir(image_dir)):

        data["Image"].append(image[:-4])



        ds = pydicom.dcmread(os.path.join(image_dir, image))



        for metadata in ds.dir():

            if metadata != "PixelData":

                metadata_values = getattr(ds, metadata)

                if type(metadata_values) == pydicom.multival.MultiValue and metadata not in ["WindowCenter", "WindowWidth"]:

                    for i, v in enumerate(metadata_values):

                        data[f"{metadata}_{i}"].append(v)

                else:

                    if type(metadata_values) == pydicom.multival.MultiValue and metadata in ["WindowCenter", "WindowWidth"]:

                        data[metadata].append(metadata_values[0])

                    else:

                        data[metadata].append(metadata_values)



    return pd.DataFrame(data).set_index("Image")
# Generate metadata dataframes

train_metadata = get_metadata(os.path.join(data_path, "stage_1_train_images"))

test_metadata = get_metadata(os.path.join(data_path, "stage_1_test_images"))



train_metadata.to_parquet(f'{data_path}/train_metadata.parquet.gzip', compression='gzip')

test_metadata.to_parquet(f'{data_path}/test_metadata.parquet.gzip', compression='gzip')
train_metadata = pd.read_parquet(f'{metadata_path}/train_metadata.parquet.gzip')

test_metadata = pd.read_parquet(f'{metadata_path}/test_metadata.parquet.gzip')



train_metadata["Dataset"] = "train"

test_metadata["Dataset"] = "test"



train_metadata = train_metadata.join(train_labels)



metadata = pd.concat([train_metadata, test_metadata], sort=True)

metadata.sort_values(by="ImagePositionPatient_2", inplace=True, ascending=False)

metadata.head()
metadata["StudyInstanceUID"].nunique()
studies = metadata.groupby("StudyInstanceUID")

studies_list = list(studies)



study_name, study_df = studies_list[0]

study_df.head()
studies.size().describe()
plt.hist(studies.size());
studies.filter(lambda x: x["Dataset"].nunique() > 1)
def window_img(dcm, width=None, level=None, norm=True):

    pixels = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

    

    # Pad non-square images

    if pixels.shape[0] != pixels.shape[1]:

        (a,b) = pixels.shape

        if a > b:

            padding = ((0, 0), ((a-b) // 2, (a-b) // 2))

        else:

            padding = (((b-a) // 2, (b-a) // 2), (0, 0))

        pixels = np.pad(pixels, padding, mode='constant', constant_values=0)

            

    if not width:

        width = dcm.WindowWidth

        if type(width) != pydicom.valuerep.DSfloat:

            width = width[0]

    if not level:

        level = dcm.WindowCenter

        if type(level) != pydicom.valuerep.DSfloat:

            level = level[0]

    lower = level - (width / 2)

    upper = level + (width / 2)

    img = np.clip(pixels, lower, upper)

    

    if norm:

        return (img - lower) / (upper - lower)

    else:

        return img
volume, labels = [], []

for index, row in study_df.iterrows():

    if row["Dataset"] == "train":

        dcm = pydicom.dcmread(os.path.join(data_path, "stage_1_train_images", index+".dcm"))

    else:

        dcm = pydicom.dcmread(os.path.join(data_path, "stage_1_test_images", index+".dcm"))

        

    img = window_img(dcm)

    label = row[["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]]

    volume.append(img)

    labels.append(label)

    

volume = np.array(volume)

labels = np.array(labels)
volume.shape, labels.shape
# Axial

plt.figure(figsize=(8, 8))

plt.imshow(volume[20, :, :], cmap=plt.cm.bone)

plt.vlines(300, 0, 512, colors='g')

plt.hlines(300, 0, 512, colors='b');
# Sagittal

plt.figure(figsize=(8, 8))

plt.imshow(volume[:, :, 300], aspect=9, cmap=plt.cm.bone)

plt.vlines(300, 0, 40, colors='b')

plt.hlines(20, 0, 512, colors='r');
# Coronal

plt.figure(figsize=(8, 8))

plt.imshow(volume[:, 300, :], aspect=9, cmap=plt.cm.bone)

plt.vlines(300, 0, 40, colors='g')

plt.hlines(20, 0, 512, colors='r');