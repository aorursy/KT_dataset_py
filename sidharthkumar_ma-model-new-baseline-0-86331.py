# Some snipets coppied from https://www.kaggle.com/rdizzl3/eda-and-baseline-model

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
train_sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

submission_file = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
days = range(1, 1913 + 1)

time_series_columns = [f'd_{i}' for i in days]

time_series_data = train_sales[time_series_columns]
MA_x = 35  #play here



forecast = time_series_data.iloc[:, -MA_x:].copy()

for i in range(28):

    forecast['F'+str(i+1)] = forecast.iloc[:, -MA_x:].mean(axis=1)    

    

forecast = forecast[['F'+str(i+1) for i in range(28)]]

forecast.head(20)
validation_ids = train_sales['id'].values

evaluation_ids = [i.replace('validation', 'evaluation') for i in validation_ids]

ids = np.concatenate([validation_ids, evaluation_ids])

predictions = pd.DataFrame(ids, columns=['id'])

forecast = pd.concat([forecast] * 2).reset_index(drop=True)

predictions = pd.concat([predictions, forecast], axis=1)

predictions.to_csv('submission.csv', index=False)
predictions.head()
predictions.shape