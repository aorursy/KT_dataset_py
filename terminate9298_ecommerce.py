# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/ecommerce/payment_transactions.csv')
data.info()
data.describe()
data.sample(10)
# gmv -  Gross merchandise volume

data[['gmv' , 'courier_cost', 'platform' , 'courier','weight' , 'from_city']].corr()
plt.figure(figsize=(16,16))

# sns.distplot(data['weight'],fit=norm , label='Weight')

# sns.distplot(data['courier'],fit=norm, label='Courier')

# sns.distplot(data['courier_cost'],fit=norm, label='Courier Cost')

sns.distplot(data['weight'], label='Weight')

sns.distplot(data['courier'], label='Courier')

sns.distplot(data['courier_cost'], label='Courier Cost')

sns.distplot(data['gmv'] , label = 'GMV')

# sns.distplot(data['from_city'] , label= 'From City')

plt.ylabel('Y - Axis')

plt.xlabel('X - Axis')

plt.legend()

plt.show()
joint_plots = ['gmv' , 'platform'  ,'courier' , 'weight' , 'from_city']

plt.figure(figsize=(8,8))

for i in joint_plots:

    sns.jointplot( x="courier_cost",y=i, data=data, height=10, ratio=3 , color='y')

    plt.show()