# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

from matplotlib import pyplot as plt
sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission.head()
train_data = pd.read_csv('../input/train.csv')

train_data.head(10)
train_data.describe()
missing_values_cols = [i for i in train_data.columns if sum(train_data[i].isnull()) !=0 ]

train_data[missing_values_cols].info()
train_data[missing_values_cols].head(10)
# drop the columns that contain too many missing value

train_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
fig, axis = plt.subplots(4,4, figsize=(16, 10))

k = 0

for i in range(axis.shape[0]):

    for j in range(axis.shape[1]):

        if k == 15 | 14:

            break

        else:

            train_data[missing_values_cols].plot(y=missing_values_cols[i], kind='hist')

            #sns.distplot(df_train[cols[i]], kde=False, ax=axis[i, j])

        k += 1