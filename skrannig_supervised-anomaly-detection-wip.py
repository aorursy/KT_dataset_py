import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data_raw = pd.read_csv('../input/Kaggle_Training_Dataset_v2.csv')
data_raw.head()
data_raw.info()
print("Ratio of backordered items: {0:8.6f}".format(data_raw.went_on_backorder.value_counts()['Yes'] / sum(data_raw.went_on_backorder.value_counts())))
data_raw.sku.value_counts()
data = data_raw.copy()
data = data.drop(labels = 'sku', axis=1)
data.head()
data[data.columns[(data.dtypes == 'object')]] = data[data.columns[(data.dtypes == 'object')]]  == 'Yes'
data.head()
data = data.iloc[:-1]
data.tail()
data.lead_time.isnull().sum() / data.shape[0]
print("Ratio of backordered items: {0:8.6f}".format(data_raw.went_on_backorder.value_counts()['Yes'] / sum(data_raw.went_on_backorder.value_counts())))