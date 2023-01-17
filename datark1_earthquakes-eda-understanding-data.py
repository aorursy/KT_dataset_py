import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
train = pd.read_csv("../input/train.csv", dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64}, nrows=2000000)
print("train shape", train.shape)
pd.options.display.precision = 20
train.head()
train.isna().sum()
train["acoustic_data"].describe()
plt.figure(figsize=(12,6))
plt.title("time_to_failure histogram")
ax = plt.plot(train["time_to_failure"], train["acoustic_data"])
plt.figure(figsize=(12,6))
plt.title("Acoustic data histogram")
ax = sns.distplot(train["acoustic_data"], label='Acustic data')
upper = train["acoustic_data"].mean()+2*train["acoustic_data"].std()
lower = train["acoustic_data"].mean()-2*train["acoustic_data"].std()

train_subset = train[(train["acoustic_data"]>lower) & (train["acoustic_data"]<upper)]
plt.figure(figsize=(12,6))
plt.title("Acoustic data histogram")
ax = sns.distplot(train_subset["acoustic_data"], label='Acustic data', kde=False)
plt.figure(figsize=(10,5))
plt.title("time_to_failure histogram")
ax = sns.distplot(train["time_to_failure"], label='time_to_failure')