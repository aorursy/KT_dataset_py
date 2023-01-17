import numpy as np 

import pandas as pd 





df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

df.columns

len(df)
df.head()
df.tail()
df_count_nan = pd.DataFrame(df.isnull().sum())

df_count_nan.rename(columns={df_count_nan.columns[0]: 'nan_count'}, inplace=True)

df_count_nan1 = df_count_nan.loc[df_count_nan['nan_count'] != 0]

df_count_notnan = df_count_nan.loc[df_count_nan['nan_count'] == 0]

df_count_notnan.index
df1 = df.copy()

df1[list(df_count_notnan.index)].hist()
df1['count'] = 1

df1.groupby('SARS-Cov-2 exam result').agg({'count': 'sum'})

df1.groupby("Patient addmited to regular ward (1=yes, 0=no)").agg({'count': 'sum'})
df1.groupby("Patient addmited to semi-intensive unit (1=yes, 0=no)").agg({'count': 'sum'})
df1.groupby("Patient addmited to intensive care unit (1=yes, 0=no)").agg({'count': 'sum'})
import seaborn as sns

sns.catplot(x='SARS-Cov-2 exam result', y="Patient age quantile", data=df1);
sns.catplot(x='Patient addmited to regular ward (1=yes, 0=no)', y="Patient age quantile", hue="SARS-Cov-2 exam result", data=df1);
sns.catplot(x='Patient addmited to semi-intensive unit (1=yes, 0=no)', y="Patient age quantile", hue="SARS-Cov-2 exam result", data=df1);
sns.catplot(x='Patient addmited to intensive care unit (1=yes, 0=no)', y="Patient age quantile", hue="SARS-Cov-2 exam result", data=df1);
df2 = df1.iloc[:,6:]

len(df2)
df2_notnan = df2.dropna(how='all')

len(df2_notnan)
df2_notnan.head()