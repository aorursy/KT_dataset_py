# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/csvfile/train_data.csv")
test=pd.read_csv("/kaggle/input/csvfile/test_data.csv")
sample=pd.read_csv("/kaggle/input/csvfile/test_data.csv")

train.head()

test.head()
print(train.shape,test.shape)
sample.head()
train.columns
train
for i in train.columns:
    print(i,train[i].corr(train['price_range'],method='pearson'))
import seaborn as sns
import matplotlib.pyplot as  plt
x_train=train.drop(columns=['price_range','id'])
y_train=train['price_range']
sns.heatmap(train)
plt.show()