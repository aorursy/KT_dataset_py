# Importing libraries
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, confusion_matrix
import pandas as pd
import category_encoders as ce

# set a seed for reproducability
random.seed(42)

# read in our data
df_2018 = pd.read_csv("/kaggle/input/data_jobs_info_2018.csv")
df_2019 = pd.read_csv("/kaggle/input/data_jobs_info_2019.csv")
# split into predictor & target variables
X = df_2018.drop("job_title", axis=1)
y = df_2018["job_title"]

# Splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20)

# save out the split training data to use with Cloud AutoML
with open("train_data_2018.csv", "+w") as file:
    pd.concat([X_train, y_train], axis=1).to_csv(file, index=False)
with open("test_data_2018.csv", "+w") as file:
    pd.concat([X_test, y_test], axis=1).to_csv(file, index=False)
from google.cloud import automl_v1beta1 as automl
from kaggle.gcp import KaggleKernelCredentials
from kaggle_secrets import GcpTarget
from google.cloud import storage

# don't change this value!
REGION = 'us-central1' # don't change: this is the only region that works currently

# these you'll change based on your GCP project/data
PROJECT_ID = 'auto-ml-kaggle-21224' # this will come from your specific GCP project
DATASET_DISPLAY_NAME = 'kaggle_survey_2018_example' # name of your uploaded dataset (from GCP console)
TARGET_COLUMN = 'job_title' # column with feature you're trying to predict

# these can be whatever you like
MODEL_DISPLAY_NAME = 'glib_automl_example_2018_model' # what you want to call your model
TRAIN_BUDGET = 1000 # max time to train model in milli-hours, from 1000-72000

storage_client = storage.Client(project=PROJECT_ID, credentials=KaggleKernelCredentials(GcpTarget.GCS)) 
tables_gcs_client = automl.GcsClient(client=storage_client, credentials=KaggleKernelCredentials(GcpTarget.GCS)) 
tables_client = automl.TablesClient(project=PROJECT_ID, region=REGION, gcs_client=tables_gcs_client, credentials=KaggleKernelCredentials(GcpTarget.AUTOML))
# first you'll need to make sure your model is predicting the right column
tables_client.set_target_column(
    dataset_display_name=DATASET_DISPLAY_NAME,
    column_spec_display_name=TARGET_COLUMN,
)
# let our model know that input columns may have missing values
for col in tables_client.list_column_specs(project=PROJECT_ID,
                                           dataset_display_name=DATASET_DISPLAY_NAME):
    if TARGET_COLUMN in col.display_name:
        continue
    tables_client.update_column_spec(project=PROJECT_ID,
                                     dataset_display_name=DATASET_DISPLAY_NAME,
                                     column_spec_display_name=col.display_name,
                                     nullable=True)
# and then you'll need to kick off your model training
response = tables_client.create_model(MODEL_DISPLAY_NAME, dataset_display_name=DATASET_DISPLAY_NAME, 
                                      train_budget_milli_node_hours=TRAIN_BUDGET, 
                                      exclude_column_spec_names=[TARGET_COLUMN])

# check if it's done yet (it won't be)
response.done()
def download_to_kaggle(bucket_name,destination_directory,file_name,prefix=None):
    """Takes the data from your GCS Bucket and puts it into the working directory of your Kaggle notebook"""
    os.makedirs(destination_directory, exist_ok = True)
    full_file_path = os.path.join(destination_directory, file_name)
    blobs = storage_client.list_blobs(bucket_name,prefix=prefix)
    for blob in blobs:
        blob.download_to_filename(full_file_path)
%%time
# name of the bucket to store your results & data in
BUCKET_NAME = "glib_automl_example"
# url of the data you're using to test
gcs_input_uris = "gs://glib_automl_example/2018 data/test_data_2018-2019-12-05T10:40:14.053Z.csv"
# folder to store outputs in (you should create this folder)
gcs_output_uri_prefix = 'gs://glib_automl_example/2018 data/predictions'

# predict
cloud_predictions = tables_client.batch_predict(
    model_display_name=MODEL_DISPLAY_NAME, 
    gcs_input_uris=gcs_input_uris,
    gcs_output_uri_prefix=gcs_output_uri_prefix
)
# from here we need to download our result file
# you can find the file path in the GCP console in the buckets for your project
RESULT_FILE_PATH = "gs://glib_automl_example/2018\ data/predictions/prediction-glib_automl_example_2018_model-2019-12-05T20:21:33.520Z/tables_1.csv"

# save to working directory
with open('cloud_automl_results.csv', "wb") as file_obj:
     storage_client.download_blob_to_file(RESULT_FILE_PATH,
                                  file_obj)
        
# load predictions into dataframe
cloud_predictions_df =  pd.read_csv("cloud_automl_results.csv")