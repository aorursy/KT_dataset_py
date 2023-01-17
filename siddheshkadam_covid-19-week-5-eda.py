# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

train.head()
train.describe()
cat=train.dtypes[train.dtypes=='object'].index

train[cat].describe()
train.hist()
train['Population'].value_counts().plot(kind='density')
train['Weight'].value_counts().plot(kind='density')
train['TargetValue'].value_counts().plot(kind='density')
train['Population'].value_counts().plot(kind='box')
train['Weight'].value_counts().plot(kind='box')
train['TargetValue'].value_counts().plot(kind='box')
corr=train.corr()
print(corr)
sb.heatmap(corr,vmax=1,square=True,annot=True)
sb.barplot(x='Target',y='Population',data=train)
sb.barplot(x='Target',y='TargetValue',data=train)
sub=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/submission.csv")

sub.to_csv('submission.csv',index=False)