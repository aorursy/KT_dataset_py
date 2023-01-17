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

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

airbnb = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
airbnb
airbnb.drop(['id','host_id','host_name','last_review','reviews_per_month'],axis=1,inplace=True)
airbnb.head()
airbnb.describe()
airbnb.isnull().sum()
medianPrice = airbnb['price'].median() #Considering price cannot be 0, price to be replaced with median of 'price'

medianPrice
airbnb = airbnb.replace({'price':{0:medianPrice}})

airbnb
airbnb.describe()
sns.heatmap( airbnb.corr(), annot=True) #since none of the features are correlated with each we will not be dropping any features
sns.pairplot(airbnb)

plt.show()
sns.catplot(x="room_type", y="price", data=airbnb);
plt.figure(figsize=(10,10))

sns.countplot(airbnb['room_type'],hue=airbnb['neighbourhood_group'], palette='plasma')
