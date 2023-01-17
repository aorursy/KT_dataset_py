import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import tqdm

import pydicom



from typing import Dict

import glob



import tensorflow as tf

from tensorflow.keras import models, layers
plt.style.use("fivethirtyeight")



sns.set(style="whitegrid")
train_df = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")

test_df = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")

sub_df = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv")



train_df.head(), test_df.head(), sub_df.head()
train_df.info(), test_df.info()
test_df.head()
train_df["Patient"].value_counts()
test_df["Patient"].value_counts().unique
train_df_patient_Id = set(train_df["Patient"].unique())

test_df_patient_Id = set(test_df["Patient"].unique())

doubles = train_df_patient_Id & test_df_patient_Id

doubles
patient_df = train_df[["Patient", "Age", "Sex", "SmokingStatus"]].drop_duplicates()

patient_df.info(), patient_df.head()
train_dir = "../input/osic-pulmonary-fibrosis-progression/train/"

test_dir = "../input/osic-pulmonary-fibrosis-progression/test/"



patient_ids = os.listdir(train_dir)

patient_ids = sorted(patient_ids)





num_instances = []

age = []

sex = []

smoking_status = []



for patient_id in patient_ids:

    patient_info = train_df[train_df["Patient"] == patient_id].reset_index()

    num_instances.append(len(os.listdir(train_dir + patient_id)))

    age.append(patient_info["Age"][0])

    sex.append(patient_info["Sex"][0])

    smoking_status.append(patient_info["SmokingStatus"][0])

    

    

patient_df = pd.DataFrame(list(zip(patient_ids, num_instances, age, sex, smoking_status)),

                               columns =["Patient", "num_instances", "Age", "Sex", "SmokingStatus"])





patient_df.info(), patient_df.head()
patient_df["Sex"].value_counts().plot(kind="bar", color='yellow', title="Sex Distribution")
patient_df["SmokingStatus"].value_counts().plot(kind="bar", color='red', title="Smoking History")
patient_df["Age"].plot(kind="hist", bins=20, color="blue", title="Age Distribution")
def plot_pixel_array(ds):

    plt.figure()

    plt.grid=False

    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

    plt.show()
def show_dcm(ds):

    print("FileName..:", file_path)

    print()

    

    patient = ds.PatientName

    display_name = patient.family_name + ", " + patient.given_name

    print("Patient Name:", display_name)

    print("Patient ID:", ds.PatientID)

    print("Sex:", ds.PatientSex)

    print("Modality:", ds.Modality)

    print("Body Part Examined:", ds.BodyPartExamined)

    

    if "PixelData" in ds:

        rows = int(ds.Rows)

        cols = int(ds.Columns)

        print("Image Size: {rows: d} x {cols: d}, {size: d} bytes".format(

            rows=rows, cols=cols, size=len(ds.PixelData)))

        if "PixelSpacing" in ds:

            print("Pixel Spacing", ds.PixelSpacing)

            ds.PixelSpacing = [1, 1]

        plt.figure()

        plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

        plt.show()



      

for file_path in glob.glob("../input/osic-pulmonary-fibrosis-progression/train/*/*.dcm"):

    ds = pydicom.dcmread(file_path)

    show_dcm(ds)

    print(ds)

    break 