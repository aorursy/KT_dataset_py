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
import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 99
train_df = pd.read_csv("../input/movie_metadata.csv")

train_df.shape
plt.figure(figsize=(8,6))

plt.scatter(range(train_df.shape[0]), np.sort(train_df.duration.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('duration', fontsize=12)

plt.show()
ulimit = np.percentile(train_df.duration.values, 99)



llimit = np.percentile(train_df.duration.values, 1)

print (ulimit, llimit)

train_df['duration'].ix[train_df['duration']>ulimit] = ulimit

train_df['duration'].ix[train_df['duration']<llimit] = llimit



print(train_df.duration.values)

plt.figure(figsize=(12,8))

#sns.distplot(train_df.duration.values, bins=50, kde=False)

plt.xlabel('duration', fontsize=12)

plt.show()