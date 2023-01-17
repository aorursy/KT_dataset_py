# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.info()
train.head()
train.describe()
train.isnull().sum().sum()
cat_column = train.columns[train.dtypes == 'object']

con_column = train.columns[train.dtypes == 'float']

corr = train[con_column].corr()

lbl = list(train[con_column])



fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corr, interpolation='nearest')

fig.colorbar(cax)



ax.set_xticklabels(['']+lbl[0:16:2])

ax.set_yticklabels(['']+lbl[0:16:2])



plt.show()

#new additional columns if i use one hot encoder

sum([len(train[colname].unique()) for colname in cat_column])
sum([len(train[colname].unique()) for colname in cat_column])