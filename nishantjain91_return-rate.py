# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#invite people for the Kaggle party

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats



%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_dataset = pd.read_csv('../input/train.csv')

train_dataset.rename(columns={"return": "returnrate"}, inplace=True)
train_dataset.info()
sns.distplot(train_dataset.returnrate)
df = train_dataset[train_dataset['returnrate']<0.1]
df.info()
sns.distplot(df.returnrate)