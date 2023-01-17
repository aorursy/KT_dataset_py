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
train_df = pd.read_csv("../input/train_2016.csv", parse_dates=["transactiondate"])

train_df.shape
train_df.head()
import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline
plt.figure(figsize=(8,6))

plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('logerror', fontsize=12)

plt.show()
ulimit = np.percentile(train_df.logerror.values, 99)

llimit = np.percentile(train_df.logerror.values, 1)

train_df['logerror'].ix[train_df['logerror']>ulimit] = ulimit

train_df['logerror'].ix[train_df['logerror']<llimit] = llimit



plt.figure(figsize=(12,8))

sns.distplot(train_df.logerror.values, bins=50, kde=False)

plt.xlabel('logerror', fontsize=12)

plt.show()
submission_pd = pd.read_csv("../input/sample_submission.csv")
submission_pd.shape
submission_pd.head()
submission_pd['201610'] = np.random.normal(0.0, 1.0, submission_pd.shape[0])
submission_pd['201611'] = np.random.normal(0.0, 1.0, submission_pd.shape[0])
submission_pd['201612'] = np.random.normal(0.0, 1.0, submission_pd.shape[0])
submission_pd['201711'] = np.random.normal(0.0, 1.0, submission_pd.shape[0])
submission_pd['201712'] = np.random.normal(0.0, 1.0, submission_pd.shape[0])
submission_pd.to_csv("submissions.csv", index= False)
out = pd.read_csv("submissions.csv")
out.shape
out.head