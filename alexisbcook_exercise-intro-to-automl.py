# TODO: Fill in your project ID and bucket name

PROJECT_ID = 'your-project-id-here'

BUCKET_NAME = 'your-bucket-name-here'



# Do not change: Fill in the remaining variables

DATASET_DISPLAY_NAME = 'house_prices_dataset'

TRAIN_FILEPATH = "../input/house-prices-advanced-regression-techniques/train.csv"

TEST_FILEPATH = "../input/house-prices-advanced-regression-techniques/test.csv"

TARGET_COLUMN = 'SalePrice'

ID_COLUMN = 'Id'

MODEL_DISPLAY_NAME = 'house_prices_model'

TRAIN_BUDGET = 2000



# Do not change: Create an instance of the wrapper

from automl_tables_wrapper import AutoMLTablesWrapper



amw = AutoMLTablesWrapper(project_id=PROJECT_ID,

                          bucket_name=BUCKET_NAME,

                          dataset_display_name=DATASET_DISPLAY_NAME,

                          train_filepath=TRAIN_FILEPATH,

                          test_filepath=TEST_FILEPATH,

                          target_column=TARGET_COLUMN,

                          id_column=ID_COLUMN,

                          model_display_name=MODEL_DISPLAY_NAME,

                          train_budget=TRAIN_BUDGET)
# Do not change: Create and train the model

amw.train_model()



# Do not change: Get predictions

amw.get_predictions()