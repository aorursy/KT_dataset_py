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
crime = pd.read_csv('../input/crime.csv',encoding='latin-1')

offence_codes = pd.read_csv('../input/offense_codes.csv',encoding='latin-1')
crime.head()
crime.columns
from matplotlib import pyplot as plt
plt.hist(crime.YEAR)

plt.show()
crime.DAY_OF_WEEK.value_counts().plot(kind='bar')
plt.figure(figsize=(13,8))

crime.OFFENSE_CODE_GROUP.value_counts().plot(kind='bar')
crime.info()
crime.SHOOTING.unique()
crime.SHOOTING = crime.SHOOTING.fillna('N')
crime.SHOOTING.unique()
crime.isnull().sum()
crime.dropna(inplace=True)
crime.shape
len(crime.STREET.unique())
offence_codes.head()