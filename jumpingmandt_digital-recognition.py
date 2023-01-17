# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

import os

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# check the files inside the folder

arr = os.listdir('../input/digit-recognizer/')

print (arr)
# Load the data

train_df = pd.read_csv("../input/digit-recognizer/train.csv")

test_df = pd.read_csv("../input/digit-recognizer/test.csv")
# view the train dataset

train_df.head()
# view the test dataset

test_df.head()
# get an exemple

train_df.drop(labels = ["label"],axis = 1).head(1)
# How many numbers exist in the train dataset. Is there only 0?



plt.hist(train_df.drop(labels = ["label"],axis = 1).head(1))
# the min value

print ('The minimum value in the train dataset is',train_df.drop(labels = ["label"],axis = 1).head(1).values.min())



# the max value

print ('The maximum value in the train dataset is',train_df.drop(labels = ["label"],axis = 1).head(1).values.max())

# Check the train dataset

train_df.isnull().sum().describe()
# Check the test dataset

test_df.isnull().sum().describe()
# create X train, y_train

X_train = train_df.drop(labels = ["label"],axis = 1).copy()

y_train = train_df["label"].copy()
# Normalize the data

X_train = X_train / 255.0

test_df = test_df / 255.0
# the max value

print ('The maximum value in the X_train dataset is',X_train.head(1).values.max())

# concat the X_train and y_train

train_df_clean = pd.concat([X_train, y_train],axis =1)

train_df_clean
test_df_clean = test_df

test_df_clean
# add a new column named 'ID'

train_df_clean['Id'] = np.arange(1,42001)

test_df_clean['Id'] = np.arange(42001,70001)
train_df_clean
test_df_clean
# save data into a temporary file

train_df_clean.to_csv(path_or_buf = 'train.csv', index = False)

test_df_clean.to_csv(path_or_buf = 'test.csv', index = False)
# GCP environment preparation

 

#REPLACE THIS WITH YOUR OWN GOOGLE PROJECT ID

PROJECT_ID = 'optimal-chimera-279914'

#REPLACE THIS WITH A NEW BUCKET NAME. NOTE: BUCKET NAMES MUST BE GLOBALLY UNIQUE

BUCKET_NAME = 'optimal-chimera-279914'

#Note: the bucket_region must be us-central1.

BUCKET_REGION = 'us-central1'
from google.cloud import storage, automl_v1beta1 as automl



storage_client = storage.Client(project=PROJECT_ID)

tables_gcs_client = automl.GcsClient(client=storage_client, bucket_name=BUCKET_NAME)

automl_client = automl.AutoMlClient()

# Note: AutoML Tables currently is only eligible for region us-central1. 

prediction_client = automl.PredictionServiceClient()

# Note: This line runs unsuccessfully without each one of these parameters

tables_client = automl.TablesClient(project=PROJECT_ID, region=BUCKET_REGION, client=automl_client, gcs_client=tables_gcs_client, prediction_client=prediction_client)
# Create your GCS Bucket with your specified name and region (if it doesn't already exist)

bucket = storage.Bucket(storage_client, name=BUCKET_NAME)

if not bucket.exists():

    bucket.create(location=BUCKET_REGION)
# moving data from kaggle to GCS

def upload_blob(bucket_name, source_file_name, destination_blob_name):

    """Uploads a file to the bucket. https://cloud.google.com/storage/docs/ """

    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(

        source_file_name,

        destination_blob_name))

    

def download_to_kaggle(bucket_name,destination_directory,file_name,prefix=None):

    """Takes the data from your GCS Bucket and puts it into the working directory of your Kaggle notebook"""

    os.makedirs(destination_directory, exist_ok = True)

    full_file_path = os.path.join(destination_directory, file_name)

    blobs = storage_client.list_blobs(bucket_name,prefix=prefix)

    for blob in blobs:

        blob.download_to_filename(full_file_path)
# upload the train and test csv files



upload_blob(BUCKET_NAME, 'train.csv', 'train.csv')

upload_blob(BUCKET_NAME, 'test.csv', 'test.csv')
# each time you should choose a new dataset_display_name (very important) 

dataset_display_name = 'digital_recog'

new_dataset = False

try:

    dataset = tables_client.get_dataset(dataset_display_name=dataset_display_name)

except:

    new_dataset = True

    dataset = tables_client.create_dataset(dataset_display_name)
# gcs_input_uris have the familiar path of gs://BUCKETNAME//file



if new_dataset:

    gcs_input_uris = ['gs://' + BUCKET_NAME + '/train.csv']



    import_data_operation = tables_client.import_data(

        dataset=dataset,

        gcs_input_uris=gcs_input_uris

    )

    print('Dataset import operation: {}'.format(import_data_operation))



    # Synchronous check of operation status. Wait until import is done.

    import_data_operation.result()

#print(dataset)
model_display_name = 'digital_recognition_model'

TARGET_COLUMN = 'label'

ID_COLUMN = 'Id'



# TODO: File bug: if you run this right after the last step, when data import isn't complete, you get a list index out of range

# There might be a more general issue, if you provide invalid display names, etc.



tables_client.set_target_column(

    dataset=dataset,

    column_spec_display_name=TARGET_COLUMN

)
# Make all columns nullable (except the Target and ID Column)

for col in tables_client.list_column_specs(PROJECT_ID,BUCKET_REGION,dataset.name):

    if TARGET_COLUMN in col.display_name or ID_COLUMN in col.display_name:

        continue

    tables_client.update_column_spec(PROJECT_ID,

                                     BUCKET_REGION,

                                     dataset.name,

                                     column_spec_display_name=col.display_name,

                                     type_code=col.data_type.type_code,

                                     nullable=True)
# Train the model. This will take hours (up to your budget). AutoML will early stop if it finds an optimal solution before your budget.

# On this dataset, AutoML usually stops around 2000 milli-hours (2 hours)



TRAIN_BUDGET = 2000 # (specified in milli-hours, from 1000-72000)

model = None

try:

    model = tables_client.get_model(model_display_name=model_display_name)

except:

    response = tables_client.create_model(

        model_display_name,

        dataset=dataset,

        train_budget_milli_node_hours=TRAIN_BUDGET,

        exclude_column_spec_names=[ID_COLUMN, TARGET_COLUMN]

    )

    print('Create model operation: {}'.format(response.operation))

    # Wait until model training is done.

    model = response.result()

# print(model)
gcs_input_uris = 'gs://' + BUCKET_NAME + '/test.csv'

gcs_output_uri_prefix = 'gs://' + BUCKET_NAME + '/predictions'



batch_predict_response = tables_client.batch_predict(

    model=model, 

    gcs_input_uris=gcs_input_uris,

    gcs_output_uri_prefix=gcs_output_uri_prefix,

)

print('Batch prediction operation: {}'.format(batch_predict_response.operation))

# Wait until batch prediction is done.

batch_predict_result = batch_predict_response.result()

batch_predict_response.metadata
# The output directory for the prediction results exists under the response metadata for the batch_predict operation

# Specifically, under metadata --> batch_predict_details --> output_info --> gcs_output_directory

# Then, you can remove the first part of the output path that contains the GCS Bucket information to get your desired directory

gcs_output_folder = batch_predict_response.metadata.batch_predict_details.output_info.gcs_output_directory.replace('gs://' + BUCKET_NAME + '/','')

download_to_kaggle(BUCKET_NAME,'/kaggle/working','tables_1.csv', prefix=gcs_output_folder)
preds_df = pd.read_csv("tables_1.csv")
# the probabilities of each test vector and Id (Id is disordered)

preds_df.iloc[:,-11:]
np.argmax(preds_df.loc[0][-10:])
predicted_label = []

for i in np.arange(len(preds_df)):

    ind = np.argmax(preds_df.loc[i][-10:])

    if ind == 0:

        predicted_label.append(1)

    elif ind == 1:

        predicted_label.append(7)

    elif ind == 2:

        predicted_label.append(3)

    elif ind == 3:

        predicted_label.append(9)

    elif ind == 4:

        predicted_label.append(2)

    elif ind == 5:

        predicted_label.append(6)

    elif ind == 6:

        predicted_label.append(0)

    elif ind == 7:

        predicted_label.append(4)

    elif ind == 8:

        predicted_label.append(8)

    else:

        predicted_label.append(5)
preds_df['predicted_label'] = predicted_label

preds_df
submission_df = preds_df[['Id', 'predicted_label']]

submission_df
submission_df.columns = ['ImageId', 'Label']

submission_df
submission_df.sort_values(by=['ImageId'], inplace=True)

submission_df
submission_df['ImageId'] = np.arange(1,28001)

submission_df
submission_df.to_csv('submission.csv', index=False)