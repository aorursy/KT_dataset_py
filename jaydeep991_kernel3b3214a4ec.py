import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/data.csv")

data
data.describe()
plt.figure(figsize=(15,10))

sns.heatmap(data.isnull(),cbar=False)
data.corr()
plt.figure(figsize=(25,25))

sns.heatmap(data.corr())
plt.figure(figsize=(15,10))

sns.countplot(x='Age',data=data,palette='rainbow')


sns.jointplot(x='Age',y='Stamina',data=data)
plt.figure(figsize=(15,10))

sns.violinplot('BallControl','Acceleration',data=data)