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
airbnb = pd.read_csv('/kaggle/input/london_airbnb copy.csv')

airbnb.head()
airbnb.shape
airbnb.sort_values('Price($)',ascending=False).iloc[:5]
airbnb[airbnb.Ratings==airbnb.Ratings.min()]
airbnb.groupby('Host_Name').Ratings.sum().sort_values(ascending = False).head()
airbnb.groupby('Bedrooms')['Price($)'].mean().round(2)
airbnb.Cancellation_policy.value_counts().sort_values(ascending = False)

airbnb.groupby('Accommodates')['Price($)'].mean().plot()
airbnb.Neighborhood.value_counts().sort_values(ascending=False)
host = airbnb.groupby('Host_Name').Reviews_count.sum()

host[host==host.max()]