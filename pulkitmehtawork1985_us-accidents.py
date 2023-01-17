# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_May19.csv")
df.shape
df.isnull().sum()
state_wise_counts = df.groupby('State')['ID'].count().reset_index()
state_wise_counts.shape
state_wise_counts.head()
state_wise_counts = state_wise_counts.sort_values(by = "ID",ascending=False)
state_wise_counts.head()
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="whitegrid")

f, ax = plt.subplots(figsize=(6, 15))

sns.barplot(y="State", x="ID", data=state_wise_counts)