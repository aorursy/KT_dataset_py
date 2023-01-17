# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras as k

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train_df = pd.read_csv('../input/train.csv', index_col='Id')

test_df = pd.read_csv('../input/test.csv', index_col='Id')

seed = 7

np.random.seed(seed)

master_df = pd.concat([train_df, test_df])
def _convert_multi_element_to_one_hot(series, master_series):

    return_df = pd.DataFrame(series)

    columns = master_series.dropna().drop_duplicates().values

    for col in columns:

        return_df[col] = series == col

    return return_df[columns].astype(int)



def _z_score(series, master_series):

    std_dev = master_series.std()

    mean = master_series.mean()

    return ((series - mean) / std_dev).fillna(0)



def parse_data(df, master_df, randomize=True, ignored_cols=['SalePrice']):

    saved_data = []

    for col in df.columns:

        if col in ignored_cols:

            saved_data.append(df[col])

        elif df[col].dtype in [np.float64, np.int64]:

            saved_data.append(_z_score(df[col], master_df[col]))

        else:

            saved_data.append(_convert_multi_element_to_one_hot(df[col], master_df[col]))

    for i, data in enumerate(saved_data):

        if i == 0:

            parsed_df = pd.DataFrame(data)

        else:

            parsed_df = pd.merge(parsed_df, pd.DataFrame(data), left_index=True, right_index=True)

    if randomize:

        return parsed_df.reindex(np.random.permutation(parsed_df.index))

    return parsed_df
parsed_train = parse_data(train_df, master_df)

parsed_test = parse_data(test_df, master_df, False)
train_cutoff = int(.95 * train_df['SalePrice'].count())

train_x = parsed_train[0:train_cutoff].drop('SalePrice', 1)

test_x = parsed_train[train_cutoff:].drop('SalePrice', 1)



train_y = parsed_train[0:train_cutoff]['SalePrice'].values

test_y = parsed_train[train_cutoff:]['SalePrice'].values
from sklearn.linear_model import LinearRegression
model = LinearRegression(train_x.shape)

model.fit(train_x, train_y)
pred = model.predict(test_x)

mean_y = train_y.mean()
from matplotlib import pyplot as plt

plt.scatter(test_y, pred)

plt.ylim(0 ,test_y.max())

plt.xlim(0 ,test_y.max())
submit_x = parsed_test.values
prediction = model.predict(submit_x)
mask = (prediction > max_y) | (prediction < 0)

prediction[mask] = mean_y
parsed_test['SalePrice'] = prediction

submit = parsed_test.reset_index()[['Id', 'SalePrice']]

submit.to_csv('submission.csv', index=False)