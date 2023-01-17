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
# import the packages and H2ODeepLearningEstimator object

import h2o

import numpy as np

import matplotlib.pyplot as plt

import pydicom as dcm



from pathlib import Path

from os.path import join, isfile

import os

import fnmatch

import math

from random import *



from h2o.estimators.deeplearning import H2ODeepLearningEstimator
#Starting H2O

h2o.init()
# import the OSIC Pulmonary Fibrosis  data

train = h2o.import_file("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")

test = h2o.import_file("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
# Get a brief summary of the data

print("dimension of train data: ",train.dim)

print("dimension of test data: ",test.dim)





print("description of train data: \n" )

train.describe()

print("description of test data: \n" )

test.describe()



# Group number of pateints by sex

Sex_patients = train.group_by("Sex")

Sex_patients.count()

Sex_patients.get_frame()



# Group number of pateints by age

Age_patients = train.group_by("Age")

Age_patients.count()

Age_patients.get_frame()



# Group number of pateints by SmokingStatus

SmokingStatus_patients = train.group_by("SmokingStatus")

SmokingStatus_patients.count()

SmokingStatus_patients.get_frame()



# Find number of patients per Sex based on the Age

cols = ["Sex","Age"]

patients_by_Sex_Age = train.group_by(by=cols).count()

patients_by_Sex_Age.get_frame()



# Mean impute the FVC column based on the Sex and Age columns

FVC_imputeSA = train.impute("FVC", method = "mean", by = ["Sex", "Age"])

FVC_imputeSA



# Mean impute the FVC column based on the Sex and SmokingStatus columns

FVC_imputeSSS = train.impute("FVC", method = "mean", by = ["Sex", "SmokingStatus"])

FVC_imputeSSS



# Mean impute the FVC column based on the Age and SmokingStatus columns

FVC_imputeASS = train.impute("FVC", method = "mean", by = ["Age", "SmokingStatus"])

FVC_imputeASS

# Show dcm images

workingFolder = os.getcwd()

listFolderDCM=os.listdir("/kaggle/input/osic-pulmonary-fibrosis-progression/train")

dirpath=Path(f'/kaggle/input/osic-pulmonary-fibrosis-progression/train')



listFolderDCM100=[]

for varFolder in listFolderDCM :

  

   varFolder1=Path(f'{dirpath}/{varFolder}')

   lenFolder=len([name for name in os.listdir(varFolder1) if os.path.isfile(os.path.join(varFolder1, name))])

   if lenFolder <=100 :

      listFolderDCM100.append(varFolder)

      

folderDCMSample=choice(listFolderDCM100)

pathDCM= Path(f'/kaggle/input/osic-pulmonary-fibrosis-progression/train/{folderDCMSample}/')



numberDCM=len(fnmatch.filter(os.listdir(pathDCM), "*.dcm"))

quotient = numberDCM / (int(math.sqrt(numberDCM))*int(math.sqrt(numberDCM)))

quotient=int(quotient)



numberRows   =int(math.sqrt(numberDCM))+quotient

numberColumns=int(math.sqrt(numberDCM))



print("number of DCM: ", numberDCM ,"number of rows: ",numberRows, "number of columns: ", numberColumns )



fig=plt.figure(figsize=(16, 16))



for i in range(1, numberDCM+1):

    dcmr = dcm.dcmread(pathDCM / f"{i}.dcm")

    img = dcmr.pixel_array

    img[img == -2000] = 0

    fig.add_subplot(numberRows, numberColumns, i)

    plt.imshow(img)

plt.show()
