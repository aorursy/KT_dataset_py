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

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#reading the data

df = pd.read_csv('/kaggle/input/zomato-restaurants-data/zomato_restaurants.csv')
#filtering for a particular city

#city = 'Bangalore'

df_blr = df#df[df['city'] == city]
#head

df_blr.head(5)
#info

df_blr.info()
#describe

df_blr.describe()
#remove rows with null values

remove_null_rating = df_blr.dropna()
remove_null_rating.info()
remove_null_rating.describe()
remove_null_rating.head()
# start some eda

#rating distribution

sns.distplot(remove_null_rating['aggregate_rating'], bins= 50)

# #city vs ratings

# citywise_mean_rating = remove_null_rating.groupby(['city'], as_index=False).mean() 

# sns.barplot(x = citywise_mean_rating['city'], y = citywise_mean_rating['aggregate_rating'])

#ditribution of rating against average cost

sns.jointplot(x = 'aggregate_rating', y = 'average_cost_for_two', data = remove_null_rating)
sns.pairplot(remove_null_rating)