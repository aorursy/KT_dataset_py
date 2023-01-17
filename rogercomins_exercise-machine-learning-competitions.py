# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor


# Path of the file to read. We changed the directory structure to simplify submitting to a competition
train_file_path = '../input/train.csv'
# path to file you will use for predictions on submission data
submission_data_path = '../input/test.csv'

data = pd.read_csv(train_file_path)
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
submission_data = pd.read_csv(submission_data_path)

y = data.SalePrice
X = data.drop(['SalePrice'], axis=1)
X = pd.get_dummies(X)
submission_X = pd.get_dummies(submission_data)

final_train, final_submission = X.align(submission_X, join='left', axis=1)

train_X, test_X, train_y, test_y = train_test_split(final_train.as_matrix(), y.as_matrix(), test_size=0.20)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
final_submission = my_imputer.transform(final_submission)

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# Typical n_estimators values range from 100-1000, though this depends a lot on the learning rate. In general, a small learning rate (and large number of estimators) will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle.
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
# early_stopping_rounds specify a number for how many rounds of straight deterioration to allow before stopping
# n_jobs: On larger datasets where runtime is a consideration, you can use parallelism to build your models faster. It's common to set the parameter n_jobs equal to the number of cores on your machine.

# make predictions
predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))


# make predictions which we will submit. 
submission_preds = my_model.predict(final_submission)



# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': submission_data.Id,
                       'SalePrice': submission_preds})

output.to_csv('submission.csv', index=False)