import os



import numpy as np

import pandas as pd

import pyarrow.parquet as pq



import matplotlib.pyplot as plt

%matplotlib inline



print(os.listdir('../input'))

print(os.listdir('../input/phase-subs'))



data_dir = '../input'
metadata_train = pd.read_csv(data_dir + '/vsb-power-line-fault-detection/metadata_train.csv')
metadata_train['target_phase_sum'] = metadata_train.groupby('id_measurement')['target'].transform(np.sum)

metadata_train.head()
plt.hist(metadata_train['target_phase_sum'].values)

plt.show()
print(len(metadata_train[metadata_train['target_phase_sum'] == 1]))

print(len(metadata_train[metadata_train['target_phase_sum'] == 2]))

print(len(metadata_train[metadata_train['target_phase_sum'] == 3]))
metadata_train[metadata_train['target_phase_sum'] == 1].head(10)
metadata_train[metadata_train['target_phase_sum'] == 2].head(10)
# Submission from LSTM model trained on each phase seperately

# Source https://www.kaggle.com/braquino/5-fold-lstm-attention-fully-commented-0-694 with modification to train and predict on each phase

sub_phase = pd.read_csv(data_dir + '/phase-subs/lstm_5fold_phase_564_sub.csv')

sub_phase_n_faults = sub_phase.target.sum()

print(sub_phase_n_faults)
# Submission from LSTM model trained on concatenated phases

# Source https://www.kaggle.com/braquino/5-fold-lstm-attention-fully-commented-0-694

sub_phase_concat = pd.read_csv(data_dir + '/phase-subs/lstm_5fold_phase_concat_648_sub.csv')

sub_phase_concat_n_faults = sub_phase_concat.target.sum()

print(sub_phase_concat_n_faults)
diff_faults = sub_phase_n_faults - sub_phase_concat_n_faults

print(diff_faults)
meta_test = pd.read_csv(data_dir + '/vsb-power-line-fault-detection/metadata_test.csv')
# Merge meta test meta data so we can compare predictions from each submission

sub_merge = sub_phase.copy()

sub_merge = sub_merge.drop(columns=['target'])

sub_merge['id_measurement'] = meta_test.id_measurement.values

sub_merge['phase'] = meta_test.phase.values

sub_merge['target_sub_phase'] = sub_phase['target'].values

sub_merge['target_sub_phase_concat'] = sub_phase_concat['target'].values

sub_merge.head()
sub_merge[sub_merge['target_sub_phase_concat'] == 1].head(50)
sub_merge['target_sub_phase_group_sum'] = sub_merge.groupby('id_measurement')['target_sub_phase'].transform(np.sum)

sub_merge[sub_merge['target_sub_phase_concat'] == 1].head(20)
sub_merge['target'] = sub_merge.target_sub_phase_concat.values

sub_merge.loc[(sub_merge['target_sub_phase'] == 0) & (sub_merge['target_sub_phase_concat'] == 1) & (sub_merge['target_sub_phase_group_sum'] == 2), 'target'] = 0
sub_merge.head()
final_sub = sub_merge[['signal_id', 'target']]

final_sub.head()
final_sub.to_csv('submission.csv', index=False)