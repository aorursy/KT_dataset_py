import numpy as np

import pandas as pd

import glob

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.concat([pd.read_csv(f) for f in glob.glob("/kaggle/input/historical-flight-and-weather-data/*.csv") ])
df.head()
df.dtypes
df.hist(figsize=(20,20)); # Tip: put a semicolon at the end of the line to avoid printing a bunch of text output.
df.shape
df.dtypes
(df.arrival_delay > 0).sum() / df.shape[0]
(df.arrival_delay > 30).sum() / df.shape[0]
(df.arrival_delay > 60).sum() / df.shape[0]
(df.departure_delay > 0).sum() / df.shape[0]
((df.arrival_delay > 0) & (df.departure_delay > 0)).sum() / df.shape[0]
df.cancelled_code.value_counts()
(df.cancelled_code != "N").sum() / df.shape[0]
df_cancel = df[df.cancelled_code != "N"]

df_cancel.hist(figsize=(20,20)); 
df_nocancel = df[df.cancelled_code == "N"]

df_nocancel.hist(figsize=(20,20)); 
print(df_cancel.HourlyWindSpeed_x.mean(), df_cancel.HourlyWindSpeed_x.median(), df_cancel.HourlyWindSpeed_x.max())

print(df_nocancel.HourlyWindSpeed_x.mean(), df_nocancel.HourlyWindSpeed_x.median(), df_nocancel.HourlyWindSpeed_x.max())