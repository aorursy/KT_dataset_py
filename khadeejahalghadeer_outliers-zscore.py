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
from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns


Data=pd.read_csv("/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv")

Data.head()
# Use log Scale

plt.hist(Data.price, bins=20, rwidth=0.8)

plt.xlabel('Price')

plt.ylabel('Count')

plt.yscale('log')

plt.show()
#Detect the outliers

maxlimit=Data["price"].quantile(0.999)

maxlimit
#Remove the outliers

Data[Data["price"]>maxlimit]
# Detect the outliers

minlimit=Data["price"].quantile(0.0001)

minlimit
# Remove the outliers

Data[Data["price"]<minlimit]
Data_outliers=Data[(Data["price"]>minlimit)|(Data["price"]<maxlimit)]

Data_outliers.head()
#Data with no outliers by using Percentile outliers removal

Data2=Data[(Data["price"]>minlimit)& (Data["price"]<maxlimit)]

Data2.head()
Data.shape[0] - Data2.shape[0]
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.hist(Data2.price, bins=20, rwidth=0.8)

plt.xlabel('price')

plt.ylabel('Count')

plt.yscale('log')

plt.show



plt.subplot(1,2,2)

sns.distplot(Data2.price)

from scipy.stats import norm

import numpy as np

plt.hist(Data2.price, bins=20, rwidth=0.8, density=True)

plt.xlabel('Price')

plt.ylabel('Count')



rng = np.arange(-1000, Data2.price.max(), 100)

plt.plot(rng, norm.pdf(rng,Data2.price.mean(),Data2.price.std()))

Data2.describe()
Data2.price.mean()
Data2.price.std()
upper_limit = Data2.price.mean() + 4*Data2.price.std()

upper_limit
lower_limit =Data2.price.mean()-4*Data2.price.std()

lower_limit
#the outliers that are beyond 4 std dev from mean

Data2[(Data2.price>upper_limit) | (Data2.price<lower_limit)]
#Outliers removal by using Standard Deviation Data without outliers

Data3=Data2[(Data2.price<upper_limit) & (Data2.price>lower_limit)]

Data3.head()
Data.shape
Data2.shape


Data3.shape
from scipy.stats import norm

import numpy as np

plt.hist(Data3.price, bins=20, rwidth=0.8, density=True)

plt.xlabel('Price')

plt.ylabel('Count')



rng = np.arange(-600, Data3.price.max(), 100)

plt.plot(rng, norm.pdf(rng,Data3.price.mean(),Data3.price.std()))
Data2["zscore"]=(Data2.price-Data2.price.mean())/Data2.price.std()

Data2.head()
#data points that has z score higher than 4 or lower than -4. Another way of saying same thing is get data points that are more than 4 standard deviation away



Data2[Data2["zscore"]>4]
Data2[Data2['zscore']<-4]
# outliers

Data2[(Data2.zscore<-4)|(Data2.zscore>4)]
# Remove the outliers from the Data points

Data4= Data2[(Data2.zscore>-4)&(Data2.zscore<4)]

Data4.head()
Data3.shape
Data4.shape
Data2.shape[0] - Data4.shape[0]
Data2.shape[0] - Data3.shape[0]