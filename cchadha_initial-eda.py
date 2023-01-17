import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode
init_notebook_mode()
train = pd.read_csv('../input/metadata_train.csv')
train.head(10)
train.shape
train.columns
print(train.signal_id.dtype)
print(train.id_measurement.dtype)
print(train.phase.dtype)
print(train.target.dtype)
train.phase.value_counts()
train.isna().any()
train.target.value_counts()
train['target'].value_counts().plot.bar()
train.groupby(["id_measurement"]).sum().query("target > 0").shape[0]
train['id_measurement'].unique().shape[0]
print('{} out of {} ids contain a fault in at least one of three phases. This is {:.0f}%'.format(
      train.groupby(["id_measurement"]).sum().query("target > 0").shape[0],
      train['id_measurement'].unique().shape[0],
      (train.groupby(["id_measurement"]).sum().query("target > 0").shape[0]*100)/train['id_measurement'].unique().shape[0]))
train.groupby(["id_measurement"]).sum()['target'].value_counts().plot.bar()
train.groupby(["id_measurement"]).sum()['target'].value_counts()
test = pd.read_csv('../input/metadata_test.csv')
test.head()
test.shape
test.phase.value_counts()
train_sig = pd.read_parquet('../input/train.parquet')
train_sig.head()
train_sig.shape
fig, ax = plt.subplots(figsize=(12,10))
for i in range(3):
    sns.lineplot(train_sig.index, train_sig[str(i)])
fig, ax = plt.subplots(figsize=(12,10))
for i in range(3,6):
    sns.lineplot(train_sig.index, train_sig[str(i)])