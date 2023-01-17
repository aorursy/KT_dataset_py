import os 

import pandas as pd

import traceback
# Set your own project id here

PROJECT_ID = 'mlflow-sample'

# from google.cloud import bigquery

# bigquery_client = bigquery.Client(project=PROJECT_ID)

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)
def create_bucket(dataset_name):

    """Creates a new bucket. https://cloud.google.com/storage/docs/ """

    bucket = storage_client.create_bucket(dataset_name)

    print('Bucket {} created'.format(bucket.name))



def upload_blob(bucket_name, source_file_name, destination_blob_name):

    """Uploads a file to the bucket. https://cloud.google.com/storage/docs/ """

    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(

        source_file_name,

        destination_blob_name))

    

def list_blobs(bucket_name):

    """Lists all the blobs in the bucket. https://cloud.google.com/storage/docs/"""

    blobs = storage_client.list_blobs(bucket_name)

    for blob in blobs:

        print(blob.name)

        

def download_to_kaggle(bucket_name,destination_directory,file_name):

    """Takes the data from your GCS Bucket and puts it into the working directory of your Kaggle notebook"""

    os.makedirs(destination_directory, exist_ok = True)

    full_file_path = os.path.join(destination_directory, file_name)

    blobs = storage_client.list_blobs(bucket_name)

    for blob in blobs:

        blob.download_to_filename(full_file_path)
bucket_name = 'mlflow-sample-curry'

try:

    create_bucket(bucket_name)   

except:

    traceback.print_exc()
local_data = '/kaggle/input/titanic/train.csv'

file_name = 'train.csv' 

upload_blob(bucket_name, local_data, file_name)

print('Data inside of',bucket_name,':')

list_blobs(bucket_name)
destination_directory = '/kaggle/working/'

file_name = 'train.csv'

download_to_kaggle(bucket_name,destination_directory,file_name)
!pwd
ls
new_file = pd.read_csv('train.csv')

new_file.head()