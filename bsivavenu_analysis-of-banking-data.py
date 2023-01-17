import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')
from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print("Train rows and columns : ", train.shape)
print("Test rows and columns : ", test.shape)
train.head()
train.dtypes
train.info()
train.describe()
plt.hist(train.target.values)
train.target.sort_values(ascending=False)
train.target.value_counts().sort_values(ascending=False)
sns.distplot(train.target)
plt.title('Target histogram.');

train.target = np.log10(train.target)
sns.distplot(train.target)
plt.title('Logarithm transformed target histogram.');
plt.figure(figsize=(8,6))
plt.scatter(range(train.shape[0]), np.sort(train['target'].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Target', fontsize=12)
plt.title("Target Distribution", fontsize=14)
plt.show()
unique_df = train.nunique().reset_index()
unique_df
unique_df.columns = ["col_name", "unique_count"]
constant_df = unique_df[unique_df["unique_count"]==1]
constant_df.shape