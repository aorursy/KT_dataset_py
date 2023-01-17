import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Import csv data as DataFrame objects.
train_df = pd.read_csv('../input/train.csv');
test_df = pd.read_csv('../input/test.csv');
# Display the first 5 samples (passengers) of our dataset.
train_df.head()
# Display the last 5 samples of our dataset.
train_df.tail()
train_df.info()
