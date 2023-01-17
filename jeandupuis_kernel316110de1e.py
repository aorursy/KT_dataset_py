!pip install urllib3

!export GOOGLE_APPLICATION_CREDENTIALS="key_google_cloud_storage"

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



from google.cloud import storage



from google.oauth2 import service_account



# Any results you write to the current directory are saved as output.

# Set your own project id here

PROJECT_ID = 'databricks-nltk'

BUCKET_NAME = 'wikipedia_dataset_nltk_english_articles'

DIRNAME_DISTANT = 'dataset'

KEY_PATH = 'https://storage.googleapis.com/wikipedia_dataset_nltk_english_articles/security/databricks-nltk-1f4e726c4db8.json'



import urllib3

import shutil 







url = KEY_PATH

c = urllib3.PoolManager()

FILE_KEY = "key_google_cloud_storage"

with c.request('GET',url, preload_content=False) as resp, open(FILE_KEY, 'wb') as out_file:

    shutil.copyfileobj(resp, out_file)



resp.release_conn()     # not 100% sure this is required though





f=open(FILE_KEY, "r")

if f.mode == 'r':

    contents =f.read()

    print (contents)

    #or, readlines reads the individual line into a list

    #fl =f.readlines()

    #for x in fl:

    #print(x)











credentials = service_account.Credentials.from_service_account_file(

    FILE_KEY,

    scopes=["https://www.googleapis.com/auth/cloud-platform"],

)





storage_client = storage.Client(project=PROJECT_ID, credentials=credentials)



def upload_blob(bucket_name, source_file_name, destination_blob_name):

    """Uploads a file to the bucket."""

    # bucket_name = "your-bucket-name"

    # source_file_name = "local/path/to/file"

    # destination_blob_name = "storage-object-name"



    #storage_client = storage.Client(credentials)

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)



    blob.upload_from_filename(source_file_name)



    print(

        "File {} uploaded to {}.".format(

            source_file_name, destination_blob_name

        )

    )

    

    

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        upload_blob(BUCKET_NAME, os.path.join(dirname, filename), os.path.join(DIRNAME_DISTANT, filename) )








