# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Singapore_data=pd.read_csv("../input/singapore-airbnb/listings.csv")
# visualisation of the top 5 rows



Singapore_data.head()
Singapore_data.info()
#Description of values in each of the numerical columns



Singapore_data.describe()
f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(Singapore_data.corr(),annot=True)#,linewidths=5,fmt='.1f',ax=ax)

#plt.show()
Singapore_data.columns
plt.figure(figsize=(10,6))

sns.distplot(a=Singapore_data['price'])#, kde=False)
plt.figure(figsize=(10,6))

plt.title("Average price per night for different neighbourhood group")

sns.barplot(x=Singapore_data['neighbourhood_group'], y=Singapore_data['price'])
plt.figure(figsize=(15,10))

plt.title("Average price per night for different neighbourhoods")

sns.barplot(x=Singapore_data['price'], y=Singapore_data['neighbourhood'])
plt.figure(figsize=(10,6))

plt.title("Average price per night for different room types")

sns.barplot(x=Singapore_data['room_type'], y=Singapore_data['price'])