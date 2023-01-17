

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import PIL.Image



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv("/kaggle/input/bengaliai-cv19/train.csv")
train.head()
import dask.dataframe as dd
%%time

df_dd = dd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_1.parquet')
%%time

df = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_1.parquet')
class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

class_map_df.shape
class_map_df.head()
class_map_df.component_type.value_counts()
train.describe()