# Disclaimer: I'm a Kaggle beginner, and this may not necessarily be a good way to treat categorical variables.
# If you have any suggestions or corrections, please let me know.

%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython import embed
from pylab import rcParams
rcParams['figure.figsize'] = 15, 7
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
target = train['target']
train.drop('target', axis=1, inplace=True)
print(train.columns[np.where(train.isnull().any().values == False)])
non_numeric_columns = list(set(train.columns) - set(train.select_dtypes(include=[np.number]).columns))
print(non_numeric_columns)
D = {}
for column in non_numeric_columns:
    col = getattr(train, column)
    D[column] = len(col.value_counts())

plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys())
plt.show()
print(max(D.keys(), key=lambda key: D[key]))
train.v22[target == 1].value_counts()[:50].plot()
train.v22[target == 0].value_counts()[:50].plot()
target.value_counts()
train.drop('v22', axis=1, inplace=True)
test.drop('v22', axis=1, inplace=True)

del D['v22']
plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys())
plt.show()
train.v56[target == 1].value_counts()[:100].plot()
train.v56[target == 0].value_counts()[:100].plot()
def get_common_values(df, var, number):
    return (set(df[var][target == 1].value_counts().keys()[:number]) |
                   set(df[var][target == 0].value_counts().keys()[:number]))

common_v56 = get_common_values(train, 'v56', 2)
print(common_v56)
def encode_value(df, val, col):
    positives = df[df[col] == val][col].index
    df["{}_{}".format(col, val)] = [1 if i in positives else 0 for i in range(len(df))]
    return df
for value in common_v56:
    train = encode_value(train, value, "v56")
    test = encode_value(test, value, "v56")
train.drop('v56', axis=1, inplace=True)
test.drop('v56', axis=1, inplace=True)

del D['v56']
plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys())
plt.show()
train.v125[target == 1].value_counts()[:100].plot()
train.v125[target == 0].value_counts()[:100].plot()
train.v125[target == 1].value_counts()[50:70].plot()
train.v125[target == 0].value_counts()[50:70].plot()
common_v125 = get_common_values(train, 'v125', 10)
print(common_v125)
for value in common_v125:
    train = encode_value(train, value, "v125")
    test = encode_value(test, value, "v125")
print(train.shape)
print(test.shape)
