import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
print(train.shape)
train.head(5)
stats = []
for col in train.columns:
    stats.append((col, train[col].nunique(), train[col].isnull().sum() * 100 / train.shape[0], train[col].value_counts(normalize=True, dropna=False).values[0] * 100, train[col].dtype))
stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Percentage of values in the biggest category', 'type'])
stats_df.sort_values('Percentage of missing values', ascending=False)[:10]
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
target_col = "target"

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.scatter(range(train.shape[0]), np.sort(train[target_col].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('target', fontsize=12)
plt.subplot(1,2,2)
sns.distplot(train[target_col].values, bins=50, kde=False, color="red")
plt.title("Histogram of target")
plt.xlabel('target', fontsize=12)
plt.show()
target_col = "target"
tab = {'1':'feature_1','2':'feature_2','3':'feature_3'}
plt.figure(figsize=(24,8))
for i,j in tab.items():
    i = int(i)
    plt.subplot(1,3,i)
    plt.scatter(train[j], train[target_col].values)
    plt.xlabel(j, fontsize=12)
    if i==1:
        plt.ylabel('target', fontsize=12)
plt.show()
sns.pairplot(train, hue='feature_1', vars=['feature_2', 'feature_3', 'target'], palette='viridis')
