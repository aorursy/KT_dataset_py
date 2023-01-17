import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import tqdm

import re

import cv2
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
train_df.head()
train_df.info()
test_df.head()
test_df.info()
import pydicom

import glob
def visualize_dicom(images, limit = 16):

    images = images[:limit]

    

    fig, ax = plt.subplots(4, 4, figsize = (20, 20))

    ax = ax.flatten()

    

    for index, file in enumerate(images):

        image_data = pydicom.read_file(file).pixel_array

        ax[index].imshow(image_data, cmap = plt.cm.bone)

        

        name = '-'.join(file.split('/')[-2:])

        ax[index].set_title(name)
TRAIN_PATH = '/kaggle/input/osic-pulmonary-fibrosis-progression/train'

image_files = glob.glob(os.path.join(TRAIN_PATH, '*', '*.dcm'))



visualize_dicom(image_files)
# !pip3 install med2image
TRAIN_PATH_PATIENT = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140'



images_patient = glob.glob(os.path.join('/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140/*.dcm'

))



images_patient.sort(key=lambda f: int(re.sub('\D', '', f)))



for index, file in enumerate(images_patient[:10]):

    print(file)
# ! mkdir "patient-png-1"
!pip3 install mritopng
import mritopng

mritopng.convert_folder('/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140', './patient-png-1/')
images_patient_png = glob.glob(os.path.join('/kaggle/working/patient-png-1/*.png'

))

images_patient_png.sort(key=lambda f: int(re.sub('\D', '', f)))



for index, file in enumerate(images_patient_png[:10]):

    print(file)
img_array = []

for frame in images_patient_png:

    img = cv2.imread(frame)

    height, width, layers = img.shape

    size = (width,height)

    img_array.append(img)

 

 

out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

 

for i in range(len(img_array)):

    out.write(img_array[i])

out.release()
image_data = pydicom.read_file(image_files[3837])

image_data
dir(image_data)
def extract_metadata(file):

    image_data = pydicom.read_file(file)

    

    record = {

        'patient_ID': image_data.PatientID,

        'patient_name': image_data.PatientName,

        'patient_sex': image_data.PatientSex,

        'modality': image_data.Modality,

        'body_part_examined': image_data.BodyPartExamined,

        'photometric_interpretation': image_data.PhotometricInterpretation,

        'rows': image_data.Rows,

        'columns': image_data.Columns,

        'pixel_spacing': image_data.PixelSpacing,

        'window_center': image_data.WindowCenter,

        'window_width': image_data.WindowWidth,

        'bits_allocated': image_data.BitsAllocated

    }

    

    return record
metadata_list = []



for file in tqdm.tqdm(image_files):

    metadata_list.append(extract_metadata(file))
metadata_df = pd.DataFrame.from_dict(metadata_list)

metadata_df.head()
len(metadata_df)
from collections import Counter

import plotly.express as px

import seaborn as sns
smoker_counts = dict(Counter(train_df['SmokingStatus']))

smoker_counts = {'status': list(smoker_counts.keys()), 'count': list(smoker_counts.values())}

smoker_df = pd.DataFrame(smoker_counts)



fig_smoker = px.pie(smoker_df, values = 'count', names = 'status', title = 'Smoker Status', hole = .5, color_discrete_sequence = px.colors.diverging.Portland)

fig_smoker.show()
sex_counts = dict(Counter(train_df['Sex']))

sex_counts = {'sex': list(sex_counts.keys()), 'count': list(sex_counts.values())}

sex_df = pd.DataFrame(sex_counts)



fig_sex = px.pie(sex_df, values = 'count', names = 'sex', title = 'Gender Distribution', hole = .5, color_discrete_sequence = px.colors.sequential.Agsunset)

fig_sex.show()
# Uncomment for interactive histogram

# fig_age = px.histogram(train_df, x="Age")

# fig_age.update_layout(title_text='Age Distribution')

# fig_age.show()



plt.figure(figsize = (10, 7))

ax = sns.distplot(train_df['Age'])

ax.set_title('Histogram for Age')
# Uncomment for interactive histogram

# fig_fvc = px.histogram(train_df, x="FVC")

# fig_fvc.update_layout(title_text='FVC Distribution')

# fig_fvc.show()



plt.figure(figsize = (10, 7))

ax = sns.distplot(train_df['FVC'])

ax.set_title('Histogram for FVC')
# Code for deleting output visualizations (reduces the chances of slow loading of the kernel)

! rm -rf './patient-png-1/'