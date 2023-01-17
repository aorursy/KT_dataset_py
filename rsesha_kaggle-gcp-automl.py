# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
TRAIN_FILEPATH = '/kaggle/input/melbourne-housing-train-test/melbourne_housing_train.csv'
import pandas as pd
import numpy as np
df = pd.read_csv(TRAIN_FILEPATH)
print(df.shape)
df.head(1)
TEST_FILEPATH = '/kaggle/input/melbourne-housing-train-test/melbourne_housing_test.csv'
df = pd.read_csv(TEST_FILEPATH)
print(df.shape)
df.head(1)
####  ENVIRONMENT VARIABLES FOR GCP HERE !!!###################
# don't change this value! 
REGION = 'us-central1' # don't change: this is the only region that works currently

# these you'll change based on your GCP project/data
## But project_ID can be changed to whatever project u are working on
PROJECT_ID = 'ram-kaggle' ### create a project name that is unique to GCP - remember that
BUCKET_NAME = 'ram-data' ### You must have already created this bucket in project above
DATASET_DISPLAY_NAME = 'housing2' # name of your uploaded dataset (from GCP console)
TARGET_COLUMN = 'Price' # column with feature you're trying to predict
ID_COLUMN = 'ID'  ## unfortunately every row must have a unique ID in your data set so keep in mind

# these can be whatever you like
MODEL_DISPLAY_NAME = 'kaggle_model3' # what you want to call your new model that you will build
TRAIN_BUDGET = 1000 # max time to train model in milli-hours, from 1000-72000

# Set your own project id here
from google.cloud import automl_v1beta1 as automl
tables_client = automl.AutoMlClient()
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)
from google.cloud import bigquery
bigquery_client = bigquery.Client(project=PROJECT_ID)

# Import the class defining the wrapper
from automl_tables_wrapper import AutoMLTablesWrapper
# Create an instance of the wrapper
amw = AutoMLTablesWrapper(project_id=PROJECT_ID,
                          bucket_name=BUCKET_NAME,
                          dataset_display_name=DATASET_DISPLAY_NAME,
                          train_filepath=TRAIN_FILEPATH,
                          test_filepath=TEST_FILEPATH,
                          target_column=TARGET_COLUMN,
                          id_column=ID_COLUMN,
                          model_display_name=MODEL_DISPLAY_NAME,
                          train_budget=TRAIN_BUDGET)
# Create and train the model - this does not tell u how many hours it took to train major miss!
amw.train_model()
# Get predictionsj - the problem is  that if you restart the notebook the model is lost
### This needs to be changed to re-use an existing model that is deplyed to make predictions
amw.get_predictions()