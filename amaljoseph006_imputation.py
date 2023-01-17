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
import numpy as np

import pandas as pd

data = pd.read_csv("/kaggle/input/retaildataset/Features data set.csv")

data.columns
#to check misssing values

data['MarkDown1'].isnull().values.any()
data['MarkDown1'].fillna((data['MarkDown1'].mean()), inplace=True)
#to check misssing values

data['MarkDown1'].isnull().values.any()
data['MarkDown1']

#similarly we can do for all other variables 
# lets see what others did
# special findings

# here there are 3 csvs , so they combined them

features=pd.read_csv('../input/retaildataset/Features data set.csv')

sales=pd.read_csv('../input/retaildataset/sales data-set.csv')

stores=pd.read_csv('../input/retaildataset/stores data-set.csv')
df=pd.merge(sales,features, on=['Store','Date', 'IsHoliday'], how='left')

df.head()

df=pd.merge(df,stores, on=['Store'], how='left')

df.head()