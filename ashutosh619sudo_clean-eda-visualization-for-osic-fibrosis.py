# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sn

# Input data files are available in the read-only "../input/" director



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")

test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
train.head()
train.shape,test.shape
train["Patient"].unique().shape
print("Mean of age: ",train["Age"].mean())



plt.figure(figsize=(10,6))

sn.distplot(train["Age"])
print("Number of Male: ", train[train["Sex"]=="Male"].shape[0])

print("Number of female: ",train[train["Sex"]=="Female"].shape[0])

plt.figure(figsize=(10,6))

sn.countplot(train["Sex"])
train.head()
train["SmokingStatus"].unique()
print("Number of Ex-smoker: ",train[train["SmokingStatus"]=="Ex-smoker"].shape[0])

print("Number of Never a smoker: ",train[train["SmokingStatus"]=="Never smoked"].shape[0])

print("Number of Currently smokes: ",train[train['SmokingStatus']=="Currently smokes"].shape[0])



plt.figure(figsize=(10,6))

sn.countplot(train["SmokingStatus"])
plt.figure(figsize=(10,6))

sn.boxplot(x="SmokingStatus",y="FVC",data=train)
train.head()
train["normal_person"] = train["FVC"]+(train["FVC"]*train["Percent"]/100)
train.head()
patients = train["Patient"][:3]



plt.figure(figsize=(16,4))



for k,patient in enumerate(patients):

    

    plt.suptitle(f"FVC vs Week", fontsize = 16)

    ax = plt.subplot(1, 3, k+1)

    ax.set_title(f"FVC vs Week for {patient}")

    sn.lineplot(x="Weeks",y="FVC",data=train[train["Patient"]==patient],ax=ax)
plt.figure(figsize=(10,6))

sn.lineplot(x="Weeks",y="FVC",data=train[train["Patient"]==patients[0]])
g = sn.pairplot(train[['FVC', 'Weeks', 'Percent', 'Age', 'SmokingStatus']], hue='SmokingStatus', aspect=1.4, height=5, diag_kind='kde', kind='reg')
import pydicom
import os



path= "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/"

exa_img = list(os.listdir(path))[:5]



for k,img in enumerate(exa_img):

    plt.figure(figsize=(25,20))

    plt.subplot(1,5,k+1)

    img = pydicom.read_file(path+img)

    img = img.pixel_array

    plt.imshow(img)
path= "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/"

exa_img = list(os.listdir(path))[:5]



for k,img in enumerate(exa_img):

    plt.figure(figsize=(25,15))

    plt.subplot(1,5,k+1)

    img = pydicom.read_file(path+img)



    img = img.pixel_array

    plt.imshow(img,cmap="gray")
path= "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/"

exa_img = list(os.listdir(path))[:5]



for k,img in enumerate(exa_img):

    plt.figure(figsize=(25,15))

    plt.subplot(1,5,k+1)

    img = pydicom.read_file(path+img)



    img = img.pixel_array

    plt.imshow(img,cmap=plt.cm.bone)