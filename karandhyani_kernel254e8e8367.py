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
import pandas as pd

from pandas import datetime

import numpy as np



import matplotlib

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



import seaborn as sns



from matplotlib.pylab import rcParams

rcParams['figure.figsize']=10,10
df=pd.read_csv('../input/temperature-readings-iot-devices/IOT-temp.csv',parse_dates=['noted_date'])

df.name='iot'
df.head(10)
cols_drop=['id','room_id/id']
df=df.drop(cols_drop,axis=1)
df.head(10)
df.dtypes
print('the dataset has shape={}'.format(df.shape))
rows_drop=['temp','out/in','noted_date']

df.drop_duplicates(subset=rows_drop,keep=False,inplace=True)
df.describe()
def features_build(df):

    df['Date']=pd.to_datetime(df['noted_date'])

    df['Year']=df['Date'].dt.year

    df['Month']=df.Date.dt.month

    df['Day']=df.Date.dt.day

    df['Weekofyear']=df.Date.dt.weekofyear

features_build(df)
ax=sns.stripplot(x='Month',y='temp',hue='out/in',data=df)