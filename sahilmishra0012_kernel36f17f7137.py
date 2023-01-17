# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
folders = os.listdir('/kaggle/input/intel-image-classification/seg_train/seg_train')
street = pd.DataFrame(columns = ['Image ID'])
forest = pd.DataFrame(columns = ['Image ID'])
mountain = pd.DataFrame(columns = ['Image ID'])
buildings = pd.DataFrame(columns = ['Image ID'])
glacier = pd.DataFrame(columns = ['Image ID'])
sea = pd.DataFrame(columns = ['Image ID'])
street['Image ID'] = os.listdir('/kaggle/input/intel-image-classification/seg_train/seg_train/street')
street['Street'] = 1
street['Forest'] = 0
street['Mountain'] = 0
street['Buildings'] = 0
street['Glacier'] = 0
street['Sea'] = 0
street
forest['Image ID'] = os.listdir('/kaggle/input/intel-image-classification/seg_train/seg_train/forest')
forest['Street'] = 0
forest['Forest'] = 1
forest['Mountain'] = 0
forest['Buildings'] = 0
forest['Glacier'] = 0
forest['Sea'] = 0
forest
mountain['Image ID'] = os.listdir('/kaggle/input/intel-image-classification/seg_train/seg_train/mountain')
mountain['Street'] = 0
mountain['Forest'] = 0
mountain['Mountain'] = 1
mountain['Buildings'] = 0
mountain['Glacier'] = 0
mountain['Sea'] = 0
mountain
buildings['Image ID'] = os.listdir('/kaggle/input/intel-image-classification/seg_train/seg_train/buildings')
buildings['Street'] = 0
buildings['Forest'] = 0
buildings['Mountain'] = 0
buildings['Buildings'] = 1
buildings['Glacier'] = 0
buildings['Sea'] = 0
buildings
glacier['Image ID'] = os.listdir('/kaggle/input/intel-image-classification/seg_train/seg_train/glacier')
glacier['Street'] = 0
glacier['Forest'] = 0
glacier['Mountain'] = 0
glacier['Buildings'] = 0
glacier['Glacier'] = 1
glacier['Sea'] = 0
glacier
sea['Image ID'] = os.listdir('/kaggle/input/intel-image-classification/seg_train/seg_train/sea')
sea['Street'] = 0
sea['Forest'] = 0
sea['Mountain'] = 0
sea['Buildings'] = 0
sea['Glacier'] = 0
sea['Sea'] = 1
sea
data = pd.concat([street,forest,mountain,buildings,glacier,sea])
data
data.to_csv('output.csv',index=False)
PROJECT_ID = 'aaria-263910'
from google.cloud import storage
# IntelClass/data
import warnings
from google.cloud import storage
warnings.filterwarnings("ignore")

def upload_blob(bucket_name, source_file_name, destination_blob_name):

    '''Function to upload blobs to GCP GCS bucket directory

        Parameters:
            bucket_name             - The name of the bucket to be instantiated.
            source_blob_name        - The blob resource to upload.
            destination_blob_name   - A file handle to which to write the blob’s data.
        Return Value:
            None
    '''

    storage_client = storage.Client(project = PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print("File {} uploaded to {}.".format(source_file_name, destination_blob_name))
for i in os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA'):
    upload_blob('imgcls','/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/'+i,'X-Ray/Train/PNEUMONIA/'+i)
folders = os.listdir('/kaggle/input/intel-image-classification/seg_train/seg_train/')
for i in folders:
    files = os.listdir('/kaggle/input/intel-image-classification/seg_train/seg_train/'+i)
    for j in files:
        upload_blob('imgcls','/kaggle/input/intel-image-classification/seg_train/seg_train/'+i+'/'+j,'IntelClass/data/'+j)
sssss