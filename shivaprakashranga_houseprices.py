# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Loading libraries 

import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0)

import seaborn as sns

from scipy import stats

from scipy.stats import norm
import matplotlib as plt
train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.shape[1]
train.info()
#check missing values

train.columns[train.isnull().any()]
#missing value counts in each of these columns

miss = train.isnull().sum()/len(train)

miss = miss[miss > 0]

miss.sort_values(inplace=True)

miss
miss = train.isnull().sum()/len(train)

miss = miss[miss > 0]

miss.sort_values(inplace=True)

miss

#visualising missing values

miss = miss.to_frame()

miss.columns = ['count']

miss.index.names = ['Name']

miss['Name'] = miss.index
import matplotlib.pyplot as plt 
#plot the missing value count

sns.set(style="whitegrid", color_codes=True)

sns.barplot(x = 'Name', y = 'count', data=miss)

plt.xticks(rotation = 90)

plt.show()