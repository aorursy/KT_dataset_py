

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.ticker as ticker

df = pd.read_csv('../input/Automate360_Assignment - Data.csv',parse_dates = ["eventDateTime"])



df['hour'] = pd.DatetimeIndex(df['eventDateTime']).hour

to_keep=['hour','stdEventName','isExcluded']

new=pd.DataFrame()

new=df[df.columns.intersection(to_keep)]

new.head()









new.drop(new[new['stdEventName']=='purchase'].index,inplace=True)

cat_vars=['isExcluded']

data1=pd.DataFrame()

for var in cat_vars:

    cat_list='var'+'_'+var

    cat_list=pd.get_dummies(new[var],prefix=var)

    new=new.join(cat_list)

    data=new

cat_vars=['isExcluded']

data_vars=data.columns.values.tolist()

to_keep=[i for i in data_vars if i not in cat_vars]

data_final1=data[to_keep]

b=data_final1.groupby(['hour'])

print(b.sum())

a=b.sum().plot()

a.xaxis.set_major_locator(ticker.MultipleLocator(1))


