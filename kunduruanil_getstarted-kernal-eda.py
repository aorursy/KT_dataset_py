# load Data and liberaies

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import pydicom

import seaborn as sns

import sklearn

import os,glob

base = "../input/osic-pulmonary-fibrosis-progression/"

print(os.listdir(base))

train = pd.read_csv(base + "train.csv")

test  = pd.read_csv(base + "test.csv")

sub = pd.read_csv(base + "sample_submission.csv")

print("train shape: " , train.shape , "test shape: ",test.shape,"submision shape: ",sub.shape)

train.head()
df = pd.DataFrame(columns=["Patient","Weeks","FVC","Percent","Age","Sex","SmokingStatus"])

c=0

for p,d in train.groupby(["Patient"]):

    df.loc[c,["Patient","Age","Sex","SmokingStatus"]] = d[["Patient","Age","Sex","SmokingStatus"]].drop_duplicates().values

    df.loc[c,"Weeks"] = d['Weeks'].values

    df.loc[c,"FVC"] = d['FVC'].values

    df.loc[c,"Percent"] = d['Percent'].values

    c+=1

print(df.shape)

df["Age"]=df['Age'].astype("int")

df.head()

print(df["Age"].describe())

df['Age'].hist()
df.isnull().sum()
sns.catplot(x='Sex',kind="count",data=df,hue="SmokingStatus")
sns.catplot(x='Sex',y="Age",data=df,hue="SmokingStatus")
fig,ax = plt.subplots(nrows=6,ncols=5,figsize=(20,30))

i,j=0,0

for file in glob.glob(base+"train/"+"ID00007637202177411956430/"+"*.dcm"):

    ax[j][i].imshow(pydicom.dcmread(file).pixel_array, cmap=plt.cm.bone)

    ax[j][i].set_title(file.split("/")[-1])

    i+=1

    if i==5:

        i=0

        j+=1

plt.show()
from IPython.display import IFrame, YouTubeVideo

YouTubeVideo('K7bFxiHCwxM',width=600, height=400)
from IPython.display import IFrame, YouTubeVideo

YouTubeVideo('YGAO7ted0UU',width=600, height=400)
files = glob.glob(base+"train/"+"ID00007637202177411956430/"+"*.dcm")

print(files[0])

image = pydicom.dcmread(files[0])
# reference https://www.kaggle.com/gunesevitan/osic-pulmonary-fibrosis-progression-eda#5.-DICOM-Files

def load_scan(patient_name):

    

    patient_directory = [pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}/{s}') for s in os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}')]

    patient_directory.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    patient_slices = np.zeros((len(patient_directory), patient_directory[0].Rows, patient_directory[0].Columns))



    for i, s in enumerate(patient_directory):

        patient_slices[i] = s.pixel_array

            

    return patient_slices



patient = 'ID00228637202259965313869'

patient_slices = load_scan(patient)

print(f'Patient {patient} CT scan is loaded - Volume Shape: {patient_slices.shape}')
import matplotlib.animation as animation

from IPython.display import HTML



fig = plt.figure(figsize=(7, 7))



ims = []

for i in patient_slices:

    im = plt.imshow(i, animated=True, cmap=plt.cm.bone)

    plt.axis('off')

    ims.append([im])



ani = animation.ArtistAnimation(fig, ims, interval=25, blit=False, repeat_delay=1000)

HTML(ani.to_html5_video())
print(test)

new = pd.DataFrame([i for i in sub['Patient_Week'].str.split("_")],columns=["Patient","Week"])

for g in new.groupby("Patient"):

    print(g[1]["Week"].unique())

    print(len(g[1]["Week"].unique()))

    break
for g in train.groupby("Patient"):

    if g[0]=="ID00007637202177411956430":

        print(g[1]["Weeks"].unique())