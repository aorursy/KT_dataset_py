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
data = pd.read_csv(r'../input/crime.csv',encoding='latin-1')
data.head()
data_shooting = data[data.SHOOTING == 'Y']
data_shooting.head()
print(len(data_shooting)/len(data)*100)
print(len(data_shooting))
print(len(data))
data_non_shooting = data[data.SHOOTING != 'Y']
data_non_shooting.head()
data_non_shooting['DISTRICT'].value_counts()
df = pd.DataFrame(data_non_shooting['OFFENSE_CODE_GROUP'].value_counts())
from IPython.display import display
pd.options.display.max_rows = 30
#display(df)
#df.to_csv('MajorOffenseNonShooting.csv')
print(df)
data_non_shooting['HOUR'].value_counts()
#I do my visualisation either in Tableau or locally and just paste the image in Kernels sometimes.It is easy this way
data[data.SHOOTING=='Y']['OFFENSE_CODE_GROUP'].value_counts()
data_shooting['HOUR'].value_counts()
data_shooting['DAY_OF_WEEK'].value_counts()
data_shooting['MONTH'].value_counts()
