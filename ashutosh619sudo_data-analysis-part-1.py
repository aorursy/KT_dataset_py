import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/train.csv")
train.head()
test = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/test.csv")
test.head()
train.shape[0]
test.shape[0]
train.info()
train["isup_grade"].value_counts().plot(kind='bar')

train["gleason_score"].value_counts().plot(kind='bar')
provider_isup = train.groupby("data_provider").sum()
provider_isup.plot(kind='bar')
data_provider_gleason_score = train.groupby("data_provider")["gleason_score"].value_counts()
plt.figure(figsize=(10,6))

data_provider_gleason_score.plot(kind="bar")