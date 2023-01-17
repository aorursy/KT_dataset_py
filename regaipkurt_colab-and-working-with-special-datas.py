import tensorflow as tf

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()



import io

veri = pd.read_csv(io.BytesIO(uploaded['veri.csv'])) #we must change name of veri.csv

#now our data imported as pandas dataframe
print(veri.head())
url = 'https://raw.githubusercontent.com/RegaipKURT/R-Project-Giri-/master/cancer.csv' #Copied GitHub Link

#When you load the URL, enter "RAW" and copy the link.



#NOT ONLY APPLY TO GITHUB, WE CAN TRANSFER DATA FROM OTHER PLACES.



veri_cancer = pd.read_csv(url)

# Dataset is now stored in a Pandas Dataframe



print(veri_cancer.head())
from google.colab import drive

drive.mount('/content/gdrive')
veri = pd.read_csv("/content/gdrive/My Drive/Colab Notebooks/datasets/timesData.csv")

print(veri.head())
from google.colab import files



uploaded = files.upload()



for fn in uploaded.keys():

  print('User uploaded file "{name}" with length {length} bytes'.format(

      name=fn, length=len(uploaded[fn])))

  

# Then move kaggle.json into the folder where the API expects to find it.

!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json



# WE NEED TO IMPORT KAGGLE MODULE

!pip install kaggle

import kaggle

!kaggle datasets download "center-for-policing-equity/data-science-for-good"
pd.read_csv("veri.csv") 
from google.colab import drive

drive.mount('/content/gdrive')
!mv /content/data-science-for-good.zip /content/gdrive/My\ Drive/Colab\ Notebooks/datasets
#DOWNLOADING DATA

!wget http://openpsychometrics.org/_rawdata/16PF.zip #link address
#exrating data from zip

import zipfile

with zipfile.ZipFile("16PF.zip", 'r') as zip_ref:

    zip_ref.extractall("ornek_zip_klasörü") #çıkaracağımız klasör ve yeri