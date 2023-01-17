# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test_id.csv')

sample = pd.read_csv('../input/sample_submission.csv')

print(train.shape, test.shape, sample.shape)
print(train['BANDWIDTH_TOTAL'].mean())
import matplotlib.pyplot as plt

def plot_eval_result(train_hist, figsize=(20,8), title='Evaluation Result'):

    plt.figure(figsize=figsize)

    for i in train_hist.columns():

        plt.plot(train_hist.columns[i])

    plt.xlabel('Epoch')

    plt.ylabel('Metric')

    plt.show()



# plot_eval_result(train_hist = train)
# train['UPDATE_TIME'] = train['UPDATE_TIME'].astype('datetime64[ns]')

test['UPDATE_TIME'] = test['UPDATE_TIME'].astype('datetime64[ns]')

print(train.head())
train_df= train.loc[train.UPDATE_TIME >= '2019-01-09']



groupby_min = train_df.groupby(['ZONE_CODE','HOUR_ID']).min().reset_index()

groupby_max = train_df.groupby(['ZONE_CODE','HOUR_ID']).max().reset_index()

groupby_mean = train_df.groupby(['ZONE_CODE','HOUR_ID']).mean().reset_index()



groupby_mean['mape_bandwidth'] = (groupby_max['BANDWIDTH_TOTAL'] - groupby_min['BANDWIDTH_TOTAL'])/groupby_min['BANDWIDTH_TOTAL'] * 100

groupby_mean['mape_user'] = (groupby_max['MAX_USER'] - groupby_min['MAX_USER'])/groupby_min['MAX_USER'] * 100

groupby_mean['min_of_BANDWIDTH'] = groupby_min['BANDWIDTH_TOTAL']

groupby_mean['min_of_MAXUSER'] = groupby_min['MAX_USER']

groupby_mean.head()
valid = train_df.drop(['BANDWIDTH_TOTAL', 'MAX_USER'], axis=1)

valid = valid.join(groupby_mean.set_index(['ZONE_CODE','HOUR_ID']),

                     on=['ZONE_CODE','HOUR_ID'])

test_df = test.join(groupby_mean.set_index(['ZONE_CODE','HOUR_ID']),

                     on=['ZONE_CODE','HOUR_ID'])

# valid.head()

# test_df.head()



# 1. Bao nhiêu zone

# 2. Vẽ đồ thị theo thời gian của từng zone

# 3. min, max, mean, median

# 4. Chọn thời gian nào

# 5. Chia train-valid

# 6. submit

train.groupby(['UPDATE_TIME','HOUR_ID']).mean().plot()
train_chart = train

train_chart = train_chart.sort_values(by=['ZONE_CODE'])

train_chart = train.set_index(['ZONE_CODE','HOUR_ID']).groupby(['UPDATE_TIME','ZONE_CODE'])

train_chart.head()
THRESHOLD = 10



valid.loc[(valid["mape_bandwidth"] > THRESHOLD) & (valid["min_of_BANDWIDTH"] < 500),

          "BANDWIDTH_TOTAL"] = np.nan

# test_df.loc[(test_df["mape_bandwidth"] > THRESHOLD) & (test_df["min_of_BANDWIDTH"] < 200), "BANDWIDTH_TOTAL"] = np.nan



# print(test_df['BANDWIDTH_TOTAL'].describe())

print(test_df.columns)
valid['MAX_USER'].fillna(0, inplace=True)

valid['BANDWIDTH_TOTAL'].fillna(0, inplace=True)

test_df['MAX_USER'].fillna(0, inplace=True)

test_df['BANDWIDTH_TOTAL'].fillna(0, inplace=True)

print(valid.shape)
def MAPE(y_true, y_pred):

    error = np.abs(y_true - y_pred)/ y_true

    error.replace([np.inf, -np.inf], np.nan, inplace=True)

    error.dropna(inplace=True)

    return np.mean(error)*100



def sMAPE(y_true, y_pred):

    assert len(y_true) == len(y_pred)

    loss = 0

    for i in range(31227,len(y_true)+31226):

        loss += 200 * abs(y_true[i] - y_pred[i]) / (abs(y_true[i]) + abs(y_pred[i]))

    return loss / len(y_true)





def mAPE(y_true, y_pred):

    assert len(y_true) == len(y_pred)

    loss = 0

    for i in range(len(y_true)):

        loss += 100 * abs(y_true[i] - y_pred[i]) / (abs(y_true[i]) + 1e-6)

    return loss / len(y_true)



bandwidth_mape = sMAPE(train_df['BANDWIDTH_TOTAL'], valid['BANDWIDTH_TOTAL'])

user_mape = sMAPE(train_df['MAX_USER'], valid['MAX_USER'])



print('MAPE bandwidth : ', bandwidth_mape)

print('MAPE user : ', user_mape)

print('MAPE total: ', bandwidth_mape*0.8 + user_mape*0.2)
test_df['MAX_USER'] = test_df.MAX_USER.astype(int).astype(str)

test_df['BANDWIDTH_TOTAL'] = test_df.BANDWIDTH_TOTAL.round(2).astype(str)

test_df['label'] = test_df['BANDWIDTH_TOTAL'].str.cat(test_df['MAX_USER'],sep=" ")



test_df[['id','label']].to_csv('sub_aivn.csv', index=False)

test_df.head()