# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
print(f"Train: {len(train)}")
test = pd.read_csv('../input/test.csv')
print(f"Test: {len(test)}")
img = train[1:2].drop('label',axis=1).values[0]
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.imshow(np.array([[img[x] for x in range(i-28,i)] for i in range(28,785,28)]))
plt.imshow(train[2:3].drop('label',axis=1).values[0].reshape(28,28),cmap="terrain")
from sklearn.model_selection import train_test_split
y = train.label.values
X = train.drop("label",axis=1).values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=0)
fig1, ax1 = plt.subplots(1,15, figsize=(15,10))
for i in range(15):
    ax1[i].imshow(X_train[i].reshape((28,28)), cmap="autumn_r")
    ax1[i].axis('off')
    ax1[i].set_title(y_train[i])
import seaborn as sns
print(train.label.value_counts())
sns.countplot(train.label,palette="RdBu")
