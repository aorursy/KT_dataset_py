import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output
print(check_output(["ls", "../input/santander-value-prediction-challenge/"]).decode("utf8"))
train_df = pd.read_csv("../input/santander-value-prediction-challenge/train.csv")

print("Train rows and columns : ", train_df.shape)

train_df.head()
plt.figure(figsize=(8,6))

plt.scatter(range(train_df.shape[0]), np.sort(train_df['target'].values))

plt.xlabel('index', fontsize=12)

plt.ylabel('Target', fontsize=12)

plt.title('Distribution of Target', fontsize=14)

plt.show()