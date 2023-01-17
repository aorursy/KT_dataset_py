print("let's start")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings as ws

ws.defaultaction = "ignore"
df = pd.read_csv("/kaggle/input/state-wise-power-consumption-in-india/dataset_tk.csv", parse_dates = True, index_col = "Unnamed: 0")
df.head()
df.columns
df.index
plt.figure(figsize=(30,30))

sns.heatmap(df.corr(), annot= True)

plt.show()
df.shape
df.isnull().sum()
mean_temprature = df.mean().sort_values(ascending=False).reset_index().rename(columns = {"index": "state", 0 : "avg_consumption"})

# looks like my state topped the consumption:) 

state_code = ['MH', 'GJ', 'UP', 'TN', 'RJ', 'MP', 'KA', 'TG', 'AP', 'PH', 'WB', 'HR', 'CT', 'DL', 'BR', 'OR', 'KL', 'J&K', 'UK', 'HP', 'AS', 'JH', 'DNH', 'GA', 'PY', 'ML', 'CH', 'TR', 'MN', 'NL', 'AR', 'MZ', 'SK']

mean_temprature.state = state_code
plt.figure(figsize = (30,30))

sns.barplot(x= "state", y = "avg_consumption", data = mean_temprature)

plt.show()
# After the above plot we can see that  the crowded state are consuming more amount of electricity and hence  is we look at them geologically the all quite cowded
df.head()
# Individual analysis

df.Maharashtra.shape
#  continued