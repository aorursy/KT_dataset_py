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

%pylab inline
train=pd.read_csv('../input/train.csv')
train.head()
train.info()
train.Age.describe()
sns.kdeplot(train.Age)
sns.kdeplot(train['Age'][train.Age.notnull()])
train.Age.describe()
train.Age.mean()
train['Age'][train.Age.notnull()].mean()
train.Age=train.Age.fillna(train.Age.mean())
train.columns