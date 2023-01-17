# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore")

import os

import glob

from tqdm.auto import tqdm

import pydicom

from matplotlib import pyplot as plt
!conda install -c conda-forge gdcm -y
train_data = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
unique_patients = train_data["Patient"].unique()
for patient in tqdm(unique_patients,total=len(unique_patients)):

    patient_ID = glob.glob("/kaggle/input/osic-pulmonary-fibrosis-progression/train/"+str(patient)+"/*.dcm")

    patient_subset = train_data[train_data["Patient"]==patient]

    patient_info = {"ID":patient_subset["Patient"].unique()[0],

                    "Age":patient_subset["Age"].unique()[0],

                    "Sex":patient_subset["Sex"].unique()[0],

                    "Smoking Status":patient_subset["SmokingStatus"].unique()[0]}

    f = plt.figure(figsize=(15,10))

    #plt.tight_layout()

    plt.suptitle("Patient ID : {}    Age : {}     Sex : {}    Smoking Status : {}".format(patient_info["ID"],patient_info["Age"],patient_info["Sex"],patient_info["Smoking Status"]),fontsize=16)

    for i,week in enumerate(patient_subset["Weeks"]):

        file_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"+str(patient)+"/"+str(week)+".dcm"

        patient_fvc = patient_subset[patient_subset["Weeks"]==week]["FVC"].values[0]

        patient_percent = patient_subset[patient_subset["Weeks"]==week]["Percent"].values[0]

        

        try:

            file = pydicom.dcmread(file_path)

            total_size = len(patient_subset["Weeks"])

            cols = 3

            rows = np.ceil(total_size / 3)

        

            plt.subplot(rows,cols,i+1)

            plt.figsize=(10,10)

            #plt.subplots_adjust(top=0.85)

            #plt.tight_layout(pad=2)

            plt.imshow(file.pixel_array,cmap=plt.cm.bone)

            plt.axis("off")

            plt.title("Week : {} , FVC : {} , Percent : {} %".format(week,patient_fvc,np.round(patient_percent,2)))

        except FileNotFoundError:

            print("File not found in kaggle input data repository")

        except:

            print("Error due to gdcm")

        

    plt.savefig("report_"+str(patient_info["ID"])+".jpg")