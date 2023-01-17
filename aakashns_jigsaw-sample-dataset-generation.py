!ls ../input
import pandas as pd
train_df = pd.read_csv('../input/train.csv')
train_df
size = len(train_df)

sample_size = size // 10

size, sample_size
train_test_sample_df = train_df.sample(2*sample_size, random_state=999)
train_sample_df = train_test_sample_df[:sample_size]

test_sample_df = train_test_sample_df[sample_size:]
train_sample_df
test_sample_df
train_sample_df.to_csv('train_sample.csv', index=False)
!head train_sample.csv
test_sample_df.to_csv('test_sample.csv', index=False)
!head test_sample.csv
!cp ../input/test.csv .

!cp ../input/sample_submission.csv .

!ls 