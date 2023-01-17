
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import os
print(os.listdir("../input"))


df_train_sample = pd.read_csv('../input/train_sample.csv')
df_train_sample.head()
df_train_sample.info()
df_train_sample['is_attributed'].value_counts().plot.bar()
df_train_sample.describe()
df_train_sample.corr()
import seaborn as sns
sns.heatmap(df_train_sample.corr())
df_train_sample.columns
df_train_sample['ip'].value_counts().sort_index().plot()
df_train_sample['app'].value_counts().sort_index().plot()
df_train_sample['attributed_time'].value_counts().sort_index().plot()
