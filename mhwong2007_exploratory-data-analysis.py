import dask.dataframe as dd

from pathlib import Path
from plotly.offline import iplot
# path to dataset
dataset_dir_path = Path('../input')
training_data_path = dataset_dir_path / 'train.csv'
testing_data_path = dataset_dir_path / 'test.csv'
sample_submission_path = dataset_dir_path / 'sample_submission.csv'
training_data = dd.read_csv(training_data_path, dtype=str)
testing_data = dd.read_csv(testing_data_path, dtype=str)
training_data.head()
testing_data.head()
training_data_columns = list(training_data.columns)
testing_data_columns = list(testing_data.columns)

print(f'Number of training data columns: {len(training_data_columns)}')
print(f'Number of testing data columns: {len(testing_data_columns)}')
print('Column names:')
print(training_data_columns)
print()
print(f'Target column: {set(training_data_columns) - set(testing_data_columns)}')
for column in testing_data_columns:
    unique_val_train = sorted(training_data[column].astype(str).drop_duplicates().compute())
    unique_val_test = sorted(testing_data[column].astype(str).drop_duplicates().compute())
    first_val_train = unique_val_train[0]
    last_val_train = unique_val_train[-1]
    only_in_training = list(set(unique_val_train) - set(unique_val_test))
    only_in_testing = list(set(unique_val_test) - set(unique_val_train))
    train_unique_val_count = len(unique_val_train)
    test_unique_val_count = len(unique_val_test)
    
    print(f'{column}:')
    print(f'unique: {unique_val_train[0:5]}{"..." if len(unique_val_train) > 5 else ""}')
    print(f'first val: {first_val_train}')
    print(f'last val: {last_val_train}')
    print(f'train unique count: {train_unique_val_count}')
    print(f'test unique count: {test_unique_val_count}')
    print(f'only in train: {only_in_training[0:5]}{"..." if len(only_in_training) > 5 else ""}')
    print(f'only in test: {only_in_testing[0:5]}{"..." if len(only_in_testing) > 5 else ""}')
    print('=================================================================')
    print()