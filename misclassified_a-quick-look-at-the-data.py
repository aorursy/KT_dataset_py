# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Loading Train and Test Data
train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])

print(f"Train set contains {train.shape[0]} observations and {train.shape[1]} variables")
print(f"Test set contains {test.shape[0]} observations and {test.shape[1]} variables")
train.head()
dupcheck = len(train) - len(train['card_id'].unique())
print(f"The data contains {dupcheck} duplicate card ids")
plt.figure(figsize = (16, 6))

plt.subplot(121)
plt.hist(train["target"], bins = 25)
plt.title("Target Variable Histogram")

plt.subplot(122)
sns.boxplot(data = train, y = "target")
plt.title("Target Variable Boxplot")

plt.show()

train["target"].describe()
train['first_active_month'].value_counts(normalize = True).sort_index().plot(figsize = (16,4), label = 'Train')
test['first_active_month'].value_counts(normalize = True).sort_index().plot(figsize = (16,4), label = 'Test')
plt.legend()
plt.title('Frequency of cards by first active month - Train vs Test Set')
plt.show()
train[['first_active_month', 'target']].groupby('first_active_month').agg(np.mean).plot(figsize = (16,4))
plt.title("Average Target Value by Card First Active Month")


train[['first_active_month', 'target']].groupby('first_active_month'
                                               ).agg(np.mean).rolling(5).mean().plot(figsize = (16, 4))
plt.title("5 Periods Average Target Value by Card First Active Month")
feats = np.arange(1,4)

nrows = len(feats)
ncols = 2

plt.figure(figsize = (15,4.5*len(feats)))
idxs = np.arange(1,7).reshape(nrows, ncols)

for idx, i in enumerate(feats):
    
    f_name = f"feature_{i}"
    
    plt.subplot(len(feats), ncols, idxs[idx][0])
    train[f_name].value_counts(normalize=True).plot(kind = 'bar', color = 'blue', alpha = 0.5)
    plt.title(f"Train Set {f_name}")
    
    plt.subplot(len(feats), ncols, idxs[idx][1])
    train[f_name].value_counts(normalize=True).plot(kind = 'bar', color = 'green', alpha = 0.5)
    plt.title(f"Test Set {f_name}")

plt.suptitle('Features Distribution Across Train and Test Set')
plt.figure(figsize = (16,4))
sns.boxplot(data = train, x='feature_1', y ='target')
plt.figure(figsize = (16,4))
sns.boxplot(data = train, x='feature_2', y ='target')
plt.figure(figsize = (16,4))
sns.boxplot(data = train, x='feature_3', y ='target')
train.corr()['target'].head(3).plot(kind='bar', color='blue', alpha = 0.5, figsize = (10,4))
