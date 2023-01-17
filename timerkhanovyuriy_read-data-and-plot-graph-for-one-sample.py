import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
database_data = pd.read_csv("/kaggle/input/oracle-database-metrics/database_data.csv", decimal=',', index_col=False)

database_data = database_data.set_index('rollup_timestamp')
database_data.sample(5)
len(database_data['database'].unique())
input_database = '72266f8b-2cc4-4315-935b-17d0c8946381'

input_metric = 'size_gd'
sample = database_data[database_data['database'] == input_database][[input_metric]]

sample.rename(columns={input_metric: "value"}, inplace=True)

sample.index = pd.to_datetime(sample.index)
sample.head(10)
plt.figure(figsize=(15, 12))

plt.plot(sample.index,sample['value'], 'r-', label = 'value')

plt.title('Database trends')

plt.ylabel('value)');

plt.legend();

plt.show()