import pandas
import pandas



# Creating one chunk of 1000 row from each data set and write it to file to work on

chunk_size = 10 ** 3

print('Starting')

for data in pandas.read_csv('../input/train_date.csv', iterator=True, chunksize=chunk_size):

    data.to_csv('date_sample.csv')

    break

for data in pandas.read_csv('../input/train_categorical.csv', iterator=True, chunksize=chunk_size):

    data.to_csv('categorical_sample.csv')

    break

for data in pandas.read_csv('../input/train_numeric.csv', iterator=True, chunksize=chunk_size):

    data.to_csv('numeric_sample.csv')

    break