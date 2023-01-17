import numpy as np

import pandas as pd 

import os
df_sample=pd.read_csv("/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv")
df_train=pd.read_csv("/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv")
df_sample.head()
df_train.head()
df_train['Label'].unique()
import seaborn as sns
sns.countplot(df_train.Label)