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
import plotly.express as px
import tensorflow as tf 
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
train_features_data = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')
test_features_data = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
train_targets_scored_data = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')
train_targets_nonscored_data = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')
sample_submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
train_features_data.shape
train_features_data.head()
train_targets_scored_data.shape
train_targets_scored_data.head()
train_targets_nonscored_data.shape
train_targets_nonscored_data.head()
# Any intersection between scored & unscored
[col for col in train_targets_scored_data.columns if col in train_targets_nonscored_data]
test_features_data.shape
test_features_data.head()
# See intersection between train and test features
[col for col in train_features_data if col not in test_features_data]
sample_submission.shape
sample_submission.head()
cols = ['cp_type','cp_time','cp_dose']

print('For train data')
for col in cols:
    print(col, ':', train_features_data[col].unique().tolist())

print('For test data')
for col in cols:
    print(col, ':', test_features_data[col].unique().tolist())
train_master_data = train_features_data.merge(train_targets_scored_data, on = 'sig_id', how = 'left')
train_master_data = train_master_data.merge(train_targets_nonscored_data, on = 'sig_id', how = 'left')
train_master_data.head()
moa_scored_cols = train_targets_scored_data.columns.tolist()[1:]
moa_nonscored_cols = train_targets_nonscored_data.columns.tolist()[1:]

all_moa_cols = moa_scored_cols + moa_nonscored_cols
print('# All moa columns = # Scored moa columns + # Nonscored moa columns')
print(len(all_moa_cols),' = ',len(moa_scored_cols),' + ', len(moa_nonscored_cols))
train_master_data['Number of MoAs']  = train_master_data[all_moa_cols].sum(axis = 1)
num_moa_across_samples = train_master_data.groupby(['Number of MoAs']).agg({'sig_id':'count'}).reset_index().rename(columns = {'sig_id':'# samples'})
fig = px.bar(num_moa_across_samples, y='# samples', x = 'Number of MoAs',text= '# samples')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', title_text="# MoA across samples",title_x=0.5)
fig.show()
num_moa_across_samples = train_master_data[train_master_data['cp_type']=='ctl_vehicle'].reset_index(drop = True).groupby(['Number of MoAs']).agg({'sig_id':'count'}).reset_index().rename(columns = {'sig_id':'# samples'})
fig = px.bar(num_moa_across_samples, y='# samples', x = 'Number of MoAs',text= '# samples')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', title_text="# No MoA across samples (ctl_vehicle)",title_x=0.5)
fig.show()
# Change dosage to one-hot:
train_master_data['cp_dose'] = np.where(train_features_data['cp_dose']=='D1',0,1)
train_master_data.head()
len(all_moa_cols)
len(train_master_data.columns.tolist()[2:-len(all_moa_cols)-1])
num_features = train_master_data[train_master_data['cp_type']=='trt_cp'].reset_index(drop = True).iloc[:,2:-len(all_moa_cols)-1].shape[1]

def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(num_features,)))

    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Dense(len(all_moa_cols), activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
train_master_data[train_master_data['cp_type']=='trt_cp'].reset_index(drop = True).iloc[:,2:-len(all_moa_cols)-1].head()
train_master_data[train_master_data['cp_type']=='trt_cp'].reset_index(drop = True)[all_moa_cols].head()
# fit model
model = build_model()
train_features_input_data = train_master_data[train_master_data['cp_type']=='trt_cp'].reset_index(drop = True).iloc[:,2:-len(all_moa_cols)-1]
train_target_data = train_master_data[train_master_data['cp_type']=='trt_cp'].reset_index(drop = True)[all_moa_cols]
history = model.fit(train_features_input_data, train_target_data, epochs=100, batch_size = 32, verbose=2)
model.summary()
test_features_data.shape
test_features_data[test_features_data['cp_type']=='trt_cp'].shape
test_master_data = test_features_data.copy()
test_master_data = test_master_data[test_master_data['cp_type']=='trt_cp'].reset_index(drop = True)
# Change dosage to one-hot:
test_master_data['cp_dose'] = np.where(test_master_data['cp_dose']=='D1',0,1)
test_master_data.iloc[:,2:].head()
predictions = model.predict(test_master_data.iloc[:,2:])
predictions.shape
predictions = pd.DataFrame(predictions)
predictions.columns = all_moa_cols
predictions.head()
test_data_with_pred = pd.concat([test_master_data[['sig_id']],predictions], axis = 1)
test_data_with_pred.head()
# Test max for 1 MoA:
print(predictions['acetylcholine_receptor_antagonist'].max(), test_data_with_pred['acetylcholine_receptor_antagonist'].max())
sample_submission.head()
final_submission = sample_submission[['sig_id']].merge(test_data_with_pred, how= 'left', on = 'sig_id')
final_submission.head()
final_submission.fillna(0, inplace = True)

prob_threshold = 0.5
for col in all_moa_cols:
    final_submission[col] = np.where(final_submission[col] >= prob_threshold, 1, 0)
final_submission[all_moa_cols].sum().sum()
print(final_submission.iloc[:,0:1 + len(moa_scored_cols)].shape, sample_submission.shape)
submission_csv = final_submission.iloc[:,0:1 + len(moa_scored_cols)].to_csv('submission.csv', index = False)
final_submission.iloc[:,0:1 + len(moa_scored_cols)].head()