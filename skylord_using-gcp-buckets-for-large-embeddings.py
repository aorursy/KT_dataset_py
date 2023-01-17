import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# System packages 

import os

import sys

# Set your own project id here

PROJECT_ID = 'project-x-262017' # This needs to be replaced with your own project name 



from google.cloud import bigquery # Import bigquery API client library 

bigquery_client = bigquery.Client(project=PROJECT_ID)



from google.cloud import storage  # Import storage client library 

storage_client = storage.Client(project=PROJECT_ID)
bucket_name = 'tpu-aakash' # select the GCP bucket 

bucket = storage_client.get_bucket(bucket_name)
%%time

blob_name = "DrugVisData_small.csv" # Select filename & print its meta information

blob = bucket.get_blob(blob_name)



print("Name: {}".format(blob.id))

print("Size: {} bytes".format(blob.size))

print("Content type: {}".format(blob.content_type))

print("Public URL: {}".format(blob.public_url))
output_file_name = "DrugVisData_small.csv" # select local filename to save file

blob.download_to_filename(output_file_name)



print("Downloaded blob {} to {}.".format(blob.name, output_file_name))
# Read the csv file & print out its header

drugData =pd.read_csv(output_file_name)

print(drugData.shape)

drugData.head()
# This is an additional code block if you want to identify the list of files <- Uncomment to run

# blobs = bucket.list_blobs()



# print("Blobs in {}:".format(bucket.name))

#for item in blobs:

#    print("\t" + item.name)
#This is the public url which is available once you add access to *allUsers* 

url = "https://storage.googleapis.com/tpu-aakash/DrugVisData_small.csv"
%%time

# csv files can be directly read using pandas *read_csv* 

drugData = pd.read_csv(url)

print(drugData.shape)

drugData.head()