# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

#these are the 999999 values, we just do not take them into account
the_low, the_high = data.min().min(), data.max().max()
data.replace([the_low, the_high], np.nan, inplace=True)

# Drop cols with std == 0
cols_to_drop = [i for i in data.columns if data[i].std() == 0]
data.drop(cols_to_drop, axis=1, inplace=True)

# Drop dulicate columns
cols = [i for i in data.columns]                                                                                            
dups = []
for i in range(len(cols) - 1):
    this_col_values = data[cols[i]].values
    for j in range(i+1, len(cols)):
        if np.array_equal(this_col_values,
                          data[cols[j]].values):
            dups.append(cols[j])
data.drop(dups, axis=1, inplace=True)

cols = [i for i in data.columns if i not in ['TARGET', 'ID']]
data['outlier_count'] = 0
data['column_outlier'] = 0
for col in cols:
    is_integer = np.all(np.equal(np.mod(data[col].unique(), 1), 0))
    if not is_integer:
        count = 'outlier_count'
    else:
        count = 'column_outlier'
    M, s = data[col].mean(), data[col].std()
    data.loc[np.abs(data[col]) > (M + (3 * s)),
             count] += 1
data.info()
sns.barplot(data.outlier_count, data.TARGET)
ct = pd.crosstab(data.TARGET, data.outlier_count).apply(lambda x:x/x.sum(), axis=0)
X, Y = ct.columns, ct.values[1,:]
plt.plot(X, Y, '.-')
plt.figure(figsize=(15, 10))
sns.barplot(data.column_outlier, data.TARGET)
ct = pd.crosstab(data.TARGET, data.column_outlier).apply(lambda x:x/x.sum(), axis=0)
X, Y = ct.columns, ct.values[1,:]
plt.plot(X, Y, '.-')