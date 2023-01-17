# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np
data = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

print(data.info())
print(data.shape)
print(data['company'].isnull().sum()) #Verifies with info
data['company'].fillna(0,inplace=True)
data.isnull().sum() #null value counts
data['country'].fillna(method="ffill",inplace=True)

data['children'].fillna(0,inplace=True)

data.isnull().sum()
mean= (data['agent'].mean())

data['agent'].fillna(mean,inplace=True)
data.isnull().sum()
data['customer_type'].unique()
import seaborn as sns

import matplotlib.pyplot as plt


fig,ax = plt.subplots(figsize=(20,10))

sns.countplot(data=data[['babies','is_canceled']],hue='babies',x='is_canceled')