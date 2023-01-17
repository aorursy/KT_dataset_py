import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df_pima = pd.read_csv('../input/pima-indians-diabetes/pima-indians-diabetes.csv')
df_pima.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8]

df_pima.head()
df_pima.info()
df_pima.describe()
df_pima.drop(columns=[0,8], axis=1)[df_pima.eq(0).any(1)]
print((df_pima[[1, 2, 3, 4, 5]] == 0).sum())
df_pima[[1, 2, 3, 4, 5]] = df_pima[[1, 2, 3, 4, 5]].replace(0, np.NaN)
df_pima.info()
df_pima.isnull().sum()
df_remove_nan = df_pima.dropna()

df_remove_nan.info()
df_remove_nan.head()
df_pima.head()
df_pima.boxplot([1, 2, 3, 4, 5])
df_nan_media = df_pima.fillna(df_pima.mean())

df_nan_media.info()
df_nan_media.head()