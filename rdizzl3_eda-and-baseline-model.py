# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Import widgets

from ipywidgets import widgets, interactive, interact

import ipywidgets as widgets

from IPython.display import display



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

calendar_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

submission_file = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
days = range(1, 1913 + 1)

time_series_columns = [f'd_{i}' for i in days]



ids = np.random.choice(train_sales['id'].unique().tolist(), 1000)



series_ids = widgets.Dropdown(

    options=ids,

    value=ids[0],

    description='series_ids:'

)



def plot_data(series_ids):

    df = train_sales.loc[train_sales['id'] == series_ids][time_series_columns]

    df = pd.Series(df.values.flatten())



    df.plot(figsize=(20, 10), lw=2, marker='*')

    df.rolling(7).mean().plot(figsize=(20, 10), lw=2, marker='o', color='orange')

    plt.axhline(df.mean(), lw=3, color='red')

    plt.grid()
w = interactive(

    plot_data,

    series_ids=series_ids

)

display(w)
series_data = train_sales[time_series_columns].values

pd.Series((series_data != 0).argmax(axis=1)).hist(figsize=(25, 5), bins=100)
pd.Series((series_data == 0).sum(axis=1) / series_data.shape[1]).hist(figsize=(25, 5), color='red')
pd.Series(series_data.max(axis=1)).value_counts().head(20).plot(kind='bar', figsize=(25, 10))
pd.Series(series_data.max(axis=1)).value_counts().tail(20)
forecast = pd.DataFrame(series_data[:, -28:]).mean(axis=1)

forecast = pd.concat([forecast] * 28, axis=1)

forecast.columns = [f'F{i}' for i in range(1, forecast.shape[1] + 1)]

forecast.head()
validation_ids = train_sales['id'].values

evaluation_ids = [i.replace('validation', 'evaluation') for i in validation_ids]
ids = np.concatenate([validation_ids, evaluation_ids])
predictions = pd.DataFrame(ids, columns=['id'])

forecast = pd.concat([forecast] * 2).reset_index(drop=True)

predictions = pd.concat([predictions, forecast], axis=1)
predictions.to_csv('submission.csv', index=False)