# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sklearn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

train.head()
train.columns = [col.lower() for col in train.columns]



train.columns
train.category.value_counts().plot(kind='barh', figsize=(10,8))
train.describe()
train[train.y==90]
train.dates = pd.to_datetime(train.dates)

train.dtypes
train.dates.apply(lambda x: x.hour).value_counts().sort_index().plot(kind='line')
train.dayofweek.value_counts()[['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']]#.plot(kind='line')