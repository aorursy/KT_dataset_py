# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
# This dataset has some missing values, which we set to the median of the column for the purpose of this tutorial. 

cleaned_train_df = train_df.fillna(train_df.median())

cleaned_test_df = test_df.fillna(test_df.median())
# Any results you write to the current directory are saved as output.

# Write the dataframes back out to a csv file, which we can more easily upload to GCS. 

cleaned_train_df.to_csv(path_or_buf='train.csv', index=False)

cleaned_test_df.to_csv(path_or_buf='test.csv', index=False)
#REPLACE THIS WITH YOUR OWN GOOGLE PROJECT ID

PROJECT_ID = 'kaggle-playground-170215'

#REPLACE THIS WITH A NEW BUCKET NAME. NOTE: BUCKET NAMES MUST BE GLOBALLY UNIQUE

BUCKET_NAME = 'automl-tutorial'

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
upload_blob(BUCKET_NAME, 'train.csv', 'train.csv')

upload_blob(BUCKET_NAME, 'test.csv', 'test.csv')
dataset_display_name = 'housing_prices'

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

# print(dataset)
model_display_name = 'tutorial_model'

TARGET_COLUMN = 'SalePrice'

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

submission_df = preds_df[['Id', 'predicted_SalePrice']]

submission_df.columns = ['Id', 'SalePrice']

submission_df.to_csv('submission.csv', index=False)