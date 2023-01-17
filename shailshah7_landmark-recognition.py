# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("../input/landmark-recognition-2020/train.csv")
train.head()
print(train)
# x = train.iloc[:,0].values
# y = train.iloc[:,1].values

x = train['id']
y = train['landmark_id']

print(x)
print(y)
count_landmark = pd.DataFrame(y.value_counts())
count_landmark.reset_index(inplace=True)
count_landmark.columns = ['landmark_id', 'count']

count_landmark
plt.figure(figsize = (12, 10))

sns.set(style="whitegrid")

sns.barplot(x="landmark_id", y="count", data=count_landmark.head(10))

plt.show()