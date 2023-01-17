import os 

import pandas as pd

import pandas_profiling as pp
wisconsin = '/kaggle/input/breast-cancer-wisconsin-data/data.csv'
from google.cloud import storage

storage_client = storage.Client(project='kaggle-playground-170215')



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
bucket_name = 'wisconsinbreastcancer_test'         

try:

    create_bucket(bucket_name)   

except:

    pass
local_data = '/kaggle/input/breast-cancer-wisconsin-data/data.csv'

file_name = 'data.csv' 

upload_blob(bucket_name, local_data, file_name)

print('Data inside of',bucket_name,':')

list_blobs(bucket_name)
destination_directory = '/kaggle/working/breastcancerwisconsin/'       

file_name = 'data.csv'

download_to_kaggle(bucket_name,destination_directory,file_name)
os.listdir('/kaggle/working/breastcancerwisconsin/')
full_file_path = os.path.join(destination_directory, file_name)

new_file = pd.read_csv(full_file_path)

pp.ProfileReport(new_file)