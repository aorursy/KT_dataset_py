import numpy as np

import pandas as pd

import time

from datetime import datetime



from sklearn.model_selection import train_test_split



from google.cloud import storage

from google.cloud import automl_v1beta1 as automl



from automlwrapper import AutoMLWrapper
# https://cloud.google.com/natural-language/automl/docs/quickstart



# Set your own values for these. bucket_name should be the project_id + '-lcm'.

PROJECT_ID = 'kaggle-real-or-not'

bucket_name = PROJECT_ID + '-lcm'



region = 'us-central1' # Region must be us-central1

dataset_display_name = 'kaggle_tweets_aug'

model_display_name = 'kaggle_starter_model_aug'



storage_client = storage.Client(project=PROJECT_ID)

client = automl.AutoMlClient()



print('successfully connected to GCP AutoML')
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#nlp_train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

#nlp_test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')



# this is the only change as compared to the original AutoML Getting Started notebook

nlp_train_df = pd.read_csv('/kaggle/input/offline-download-of-gpt2-augmented/train_df_combined.csv')

nlp_test_df = pd.read_csv('/kaggle/input/tweet-cleaner/test_df_clean.csv')



def callback(operation_future):

    result = operation_future.result()

    

print('loaded data')
nlp_test_df.tail()
def upload_blob(bucket_name, source_file_name, destination_blob_name):

    """Uploads a file to the bucket. https://cloud.google.com/storage/docs/ """

    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}'.format(

        source_file_name,

        'gs://' + bucket_name + '/' + destination_blob_name))

    

def download_to_kaggle(bucket_name,destination_directory,file_name,prefix=None):

    """Takes the data from your GCS Bucket and puts it into the working directory of your Kaggle notebook"""

    os.makedirs(destination_directory, exist_ok = True)

    full_file_path = os.path.join(destination_directory, file_name)

    blobs = storage_client.list_blobs(bucket_name,prefix=prefix)

    for blob in blobs:

        blob.download_to_filename(full_file_path)
bucket = storage.Bucket(storage_client, name=bucket_name)

if not bucket.exists():

    bucket.create(location=region)



print('GCP bucket created')
# Select the text body and the target value, for sending to AutoML NL

nlp_train_df[['text','target']].to_csv('train.csv', index=False, header=False)
print('starting data upload')



training_gcs_path = 'uploads/kaggle_getstarted/full_train.csv'

upload_blob(bucket_name, 'train.csv', training_gcs_path)



print('data upload completed')
amw = AutoMLWrapper(client=client, 

                    project_id=PROJECT_ID, 

                    bucket_name=bucket_name, 

                    region='us-central1', 

                    dataset_display_name=dataset_display_name, 

                    model_display_name=model_display_name)



print('AutoML wrapper created')
if not amw.get_dataset_by_display_name(dataset_display_name):

    print('dataset not found, creating new one')

    amw.create_dataset()

    # this part took me around 30 minutes

    amw.import_gcs_data(training_gcs_path)



amw.dataset
print('starting model train')

# started at 5:27pm - finished around 9pm

if not amw.get_model_by_display_name():

    # took me around 3-4 hours to train the model

    amw.train_model()

    

print('train complete')

print('starting model deploy')

amw.deploy_model() # took me around 10 min

print('model deploy complete')

amw.model
amw.model_full_path
# Create client for prediction service.

prediction_client = automl.PredictionServiceClient()

amw.set_prediction_client(prediction_client)



print('starting predictions')



# takes about 20-30 min

predictions_df = amw.get_predictions(nlp_test_df, 

                                     input_col_name='text', 

#                                      ground_truth_col_name='target', # we don't have ground truth in our test set

                                     limit=None, 

                                     threshold=0.5,

                                     verbose=False)

print('predictions complete')
print('starting model undeploy')

amw.undeploy_model()

print('undeploy complete')
submission_df = pd.concat([nlp_test_df['id'], predictions_df['class']], axis=1)

submission_df.head()
submission_df = submission_df.rename(columns={'class':'target'})

submission_df.head()
submission_df.to_csv("submission.csv", index=False, header=True)