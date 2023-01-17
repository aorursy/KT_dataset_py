# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
df.head()
df.info()
df.columns
df.isna().sum()
df["Province/State"].isna().sum() * 100 / len(df["Province/State"])
colours = ["#66FF00", "#ff2000"] # Green and red colors

sns.heatmap(df.isna(), cmap = colours, cbar = False)
df["Province/State"].unique()
df["Province/State"].value_counts()
df["Province/State"].value_counts() * 100 / len(df["Province/State"].value_counts())
df["Province/State"].replace({np.nan: "Diamond Princess cruise ship"}, inplace = True)
df.isna().sum()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].str.lower()
df.head()
df.columns = df.columns.str.lower()
np.round(df.describe(), 2)
df.drop("SNo", axis = 1, inplace = True)
df.head()
figure, axes = plt.subplots(1, 3, figsize = (20, 8))
axes = axes.flatten()
k = 0

colors = ["green", "orange", "steelblue"]

for i, col in enumerate(df.columns):
    if df[col].dtype == "float":
        sns.distplot(df[col], kde = False, color = colors[k], ax = axes[k])
        axes[k].grid(True)
        k += 1
figure, axes = plt.subplots(1, 3, figsize = (20, 8))
axes = axes.flatten()
k = 0

colors = ["green", "orange", "steelblue"]

for i, col in enumerate(df.columns):
    if df[col].dtype == "float":
        sns.lineplot(df["observationdate"], df[col], color = colors[k], ax = axes[k])
        k += 1
df_time_ser = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
df_time_ser.head()
len(df_time_ser.columns[4:])
df_time_ser.isna().sum() * 100 / len(df_time_ser.isna().sum())
df_time_ser.drop("Province/State", axis = 1, inplace = True)
df_time_ser.columns = df_time_ser.columns.str.lower()

for col in df_time_ser.columns:
    if df_time_ser[col].dtype == "object":
        df_time_ser[col] = df_time_ser[col].str.lower()
df_time_ser.head()
df_time_ser.describe()