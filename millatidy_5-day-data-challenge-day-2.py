# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



data = pd.read_csv('../input/archive.csv')
data.describe()
data['February Average Temperature']
x = data['February Average Temperature']

x.dropna(inplace=True)

sns.distplot(x, kde=False).set_title('February Average Temperature (Northeast)')