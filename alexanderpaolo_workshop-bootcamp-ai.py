import pandas as pd

# Importing libraries

import random

from sklearn.model_selection import train_test_split

from sklearn.metrics import auc, accuracy_score, confusion_matrix

import pandas as pd

import category_encoders as ce



# set a seed for reproducability

random.seed(42)

df_2018 = pd.read_csv("../input/data_jobs_info_2018.csv")

df_2019 = pd.read_csv("../input/data_jobs_info_2019.csv")
# split into predictor & target variables

X = df_2018.drop("job_title", axis=1)

y = df_2018["job_title"]



# Splitting data into training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20)



# save out the split training data to use with Cloud AutoML

with open("train_data.csv", "+w") as file:

    pd.concat([X_train, y_train], axis=1).to_csv(file, index=False)
# encode all features using ordinal encoding

encoder_x = ce.OrdinalEncoder()

X_encoded = encoder_x.fit_transform(X)



# you'll need to use a different encoder for each dataframe

encoder_y = ce.OrdinalEncoder()

y_encoded = encoder_y.fit_transform(y)



# split encoded dataset

X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X_encoded, y_encoded,

                                                    train_size=0.80, test_size=0.20)
from xgboost import XGBClassifier



# train XGBoost model with default parameters

my_model = XGBClassifier()

my_model.fit(X_train_encoded, y_train_encoded, verbose=False)



# and save our model

my_model.save_model("xgboost_baseline.model")
from google.cloud import automl_v1beta1 as automl

from kaggle.gcp import KaggleKernelCredentials

from kaggle_secrets import GcpTarget

from google.cloud import storage
from google.cloud import automl_v1beta1 as automl

from kaggle.gcp import KaggleKernelCredentials

from kaggle_secrets import GcpTarget

from google.cloud import storage





# don't change this value!

REGION = 'us-central1' # don't change: this is the only region that works currently



# these you'll change based on your GCP project/data

PROJECT_ID = 'translate-1563489517223' # this will come from your specific GCP project

DATASET_DISPLAY_NAME = '2018_automl' # name of your uploaded dataset (from GCP console)

TARGET_COLUMN = 'job_title' # column with feature you're trying to predict



# these can be whatever you like

MODEL_DISPLAY_NAME = 'paolo_automl_example_model' # what you want to call your model

TRAIN_BUDGET = 1000 # max time to train model in milli-hours, from 1000-72000



storage_client = storage.Client(project=PROJECT_ID, credentials=KaggleKernelCredentials(GcpTarget.GCS)) 

tables_gcs_client = automl.GcsClient(client=storage_client, credentials=KaggleKernelCredentials(GcpTarget.GCS)) 

tables_client = automl.TablesClient(project=PROJECT_ID, region=REGION, gcs_client=tables_gcs_client, credentials=KaggleKernelCredentials(GcpTarget.AUTOML))
# Nos aseguramos de que la columna que se va a predecir sea la correcta

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
response = tables_client.create_model(MODEL_DISPLAY_NAME, dataset_display_name=DATASET_DISPLAY_NAME, 

                                      train_budget_milli_node_hours=TRAIN_BUDGET, 

                                      exclude_column_spec_names=[TARGET_COLUMN])



# check if it's done yet (it won't be)

response.done()
# Creamos el modelo

response = tables_client.create_model(MODEL_DISPLAY_NAME, dataset_display_name=DATASET_DISPLAY_NAME, 

                                      train_budget_milli_node_hours=TRAIN_BUDGET, 

                                      exclude_column_spec_names=[TARGET_COLUMN])



# Revisamos si ya está entrenado el modelo

response.done()
import h2o

from h2o.automl import H2OAutoML



# initilaize an H20 instance running locally

h2o.init()
# convert our data to h20Frame, an alternative to pandas datatables

train_data = h2o.H2OFrame(X_train)

test_data = h2o.H2OFrame(list(y_train))



train_data = train_data.cbind(test_data)



# Run AutoML for 20 base models (limited to 1 hour max runtime by default)

aml = H2OAutoML(max_models=20, seed=1)

aml.train(y="C1", training_frame=train_data)
# View the top five models from the AutoML Leaderboard

lb = aml.leaderboard

lb.head(rows=5)



# The leader model can be access with `aml.leader`
h2o.save_model(aml.leader)