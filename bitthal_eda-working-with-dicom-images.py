import numpy as np

import pandas as pd



import os

import matplotlib.pyplot as plt

import seaborn as sns



import pydicom

from pydicom.data import get_testdata_files



PATH = '/kaggle/input/siim-isic-melanoma-classification'
train_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

print(train_df.shape)

train_df.head()
print("Number of Train Images: ", train_df.image_name.nunique())

print("Number of Train Patirnts: ", train_df.patient_id.nunique())
temp = train_df[['patient_id', 'image_name']].groupby('patient_id').count()



plt.hist(temp.image_name, 30)

plt.xlabel('Image Count per Patient')

plt.ylabel('Number of Patients')

plt.title(r'Per Patient Image Count Distribution')

plt.show()
fig = plt.figure(figsize=(16,6))

axes = fig.subplots(1, 3)



plt.xticks(

    rotation=45, 

    horizontalalignment='right',

)



sns.countplot(x='sex', data=train_df[['patient_id', 'sex']].drop_duplicates(), ax=axes[0])

axes[0].set_title("Sex Ratio of Patients")

sns.countplot(x='benign_malignant', data=train_df, ax=axes[1])

axes[1].set_title("Malignamt Images")

sns.countplot(x='diagnosis', data=train_df, ax=axes[2])

axes[2].set_title("Different Disgnosis in Images");
train_df.head()
fig = plt.figure(figsize=(16,6))

# axes = fig.subplots(1, 3)

temp = train_df[['patient_id', 'target']].groupby('patient_id').sum()

sns.countplot(x='target', data=temp)

plt.show()

print("Values")

temp.reset_index().groupby('target').count()
temp = train_df[['patient_id', 'target']].groupby('patient_id').sum()

temp.reset_index(inplace=True)

temp.target = temp.target.apply(lambda x: 1 if x != 0 else 0)

print(temp.shape)

temp = temp.merge(train_df[['patient_id', 'sex']].drop_duplicates(), on='patient_id', how='left')



sns.countplot(x='target', hue='sex', data=temp)

plt.title('Sex Based Malignamt Patients Distibution');
import random

from PIL import Image

img_name = random.choice(train_df.image_name.tolist())
img = Image.open(os.path.join(PATH, "jpeg/train", img_name+'.jpg'))

plt.imshow(np.array(img))
image_data = pydicom.dcmread(os.path.join(PATH, "train", img_name+'.dcm'))

image_data
def read_dicom_image(img_name, train=True):

    if train:

        img_path = os.path.join(PATH, "train", img_name+'.dcm')

    else:

        img_path = os.path.join(PATH, "test", img_name+'.dcm')

    image_data = pydicom.dcmread(img_path)



    img_data = {

            "patient_name": image_data.PatientName,

            "patient_id": image_data.PatientID,

            "modality": image_data.Modality,

            "sex": image_data.PatientSex,

            "image_name": image_data.StudyID,

            "rows": int(image_data.Rows),

            "cols": int(image_data.Columns),

            "body_part_examined": image_data.BodyPartExamined,

            "age": int(image_data.PatientAge.replace('Y', '')),

            }



    return image_data.pixel_array, img_data



def show_dicom_image(img):

    plt.imshow(img, cmap=plt.cm.bone)

    plt.show()

    

def show_image(img_name, train=True):

    if train:

        img_path = os.path.join(PATH, "jpeg/train", img_name+'.jpg')

    else:

        img_path = os.path.join(PATH, "jpeg/test", img_name+'.jpg')



    img = Image.open(img_path)

    plt.imshow(np.array(img))

    plt.show()
img, img_data =read_dicom_image('ISIC_0149568')

show_dicom_image(img)

img_data
temp = train_df[['patient_id', 'image_name']].groupby('patient_id').count()

temp = temp.merge(train_df[['patient_id', 'target']].drop_duplicates(), on='patient_id', how='left')

temp[(temp.target == 1) & (temp.image_name == 6)].head(2)
fig = plt.figure(figsize=(16,6))

axes = fig.subplots(2, 3)



for i, row in train_df[train_df.patient_id == 'IP_0274810'].reset_index().iterrows():

    img_path = os.path.join(PATH, "jpeg/train", row.image_name+'.jpg')

    axes[i//3][i%3].imshow(np.array(Image.open(img_path)))

    axes[i//3][i%3].set_title(row.benign_malignant)

plt.show()

fig = plt.figure(figsize=(16,6))

axes = fig.subplots(2, 3)



for i, row in train_df[train_df.patient_id == 'IP_0274810'].reset_index().iterrows():

    img_path = os.path.join(PATH, "jpeg/train", row.image_name+'.jpg')

    img, _ = read_dicom_image(row.image_name)

    axes[i//3][i%3].imshow(img, cmap=plt.cm.bone)

    axes[i//3][i%3].set_title(row.benign_malignant)

plt.show()
test_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')

print(test_df.shape)

test_df.head()
print("Number of Train Images: ", test_df.image_name.nunique())

print("Number of Train Patirnts: ", test_df.patient_id.nunique())