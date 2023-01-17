

# Save CSV file with first 2 million rows only

import pandas as pd

train_df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", nrows = 2_000_000)

train_df.to_csv("train_small.csv", index=False)
PROJECT_ID = 'kaggle-playground-170215'

BUCKET_NAME = 'automl-tutorial-alexis'



DATASET_DISPLAY_NAME = 'taxi_fare_dataset'

TRAIN_FILEPATH = "../working/train_small.csv" 

TEST_FILEPATH = "../input/new-york-city-taxi-fare-prediction/test.csv"



TARGET_COLUMN = 'fare_amount'

ID_COLUMN = 'key'



MODEL_DISPLAY_NAME = 'tutorial_model'

TRAIN_BUDGET = 4000



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
# Create and train the model

amw.train_model()



# Get predictions

amw.get_predictions()
submission_df = pd.read_csv("../working/submission.csv")

submission_df.head()