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
df=pd.read_csv('/kaggle/input/nsw-suburb-median-price-years-20072020/nsw_suburb_median_price -2007-2020.csv')
df['qtr'] = pd.to_datetime(df['qtr'])

df.set_index('qtr', inplace=True)

df.tail(10)
df.columns
#Plot the data for houses

%matplotlib inline 

df[df['property_type']=='house'].groupby('suburb')['median_pice'].plot(figsize=(15,5))
#Plot the data for units

%matplotlib inline 

df[df['property_type']=='unit'].groupby('suburb')['median_pice'].plot(figsize=(15,5))