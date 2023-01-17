# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
path = "../input"
print(os.listdir(path))
train = pd.read_csv(path + '/train-balanced-sarcasm.csv')
# Any results you write to the current directory are saved as output.
train
train['label'].hist() # 50% od každé super :)
train['comment'].sample(10)
len(train)
train[train.label == 1]["comment"].sample(10).tolist()
train.groupby(["subreddit"]).count()["comment"].sort_values()
def sample(n):
    return commonReddit.head(n).append(commonReddit.tail(n))
shortSample = sample(5)
kv = [(key, shortSample[key]) for key in shortSample.keys()]
result = {"subreddit": [], "count": []}
for col1, col2 in kv:
    result["subreddit"].append(col1)
    result["count"].append(col2)
pd.DataFrame(result).plot(x="subreddit", y="count", kind="bar", legend=False)
