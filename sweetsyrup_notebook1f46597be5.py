import numpy as np

import pandas as pd

from pandas import DataFrame

from matplotlib import pyplot as plt

import seaborn as sns



%matplotlib inline
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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
test.head()
train.describe()
test.describe()
plt.figure(figsize=(20,10))

sns.distplot(bins=200, a=train.accuracy)
train_head = train[:10000]



plt.figure(figsize=(20,15))

plt.scatter(x=train_head.x, y=train_head.y, c=train_head.time)
plt.figure(figsize=(20,10))

plt.scatter(x=train_head.time, y=train_head.accuracy)