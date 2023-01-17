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

data = pd.read_csv("/kaggle/input/berlin-airbnb-data/calendar_summary.csv")

data.head(6)
#the column available is discrete so i do one hot encoding

one_hot = pd.get_dummies(data['available'])

one_hot
# Drop column available as it is now encoded

data = data.drop('available',axis = 1)

# Join the encoded df

data = data.join(one_hot)

data.head(6)
# price is string

data.price = data.price.str.replace('$', '').str.replace(',', '').astype(float)
type(data.price[5])
#mean imputation

data['price'].fillna((data['price'].mean()), inplace=True)
data.head(5)
# now standardization for columns 2:4

from sklearn.preprocessing import StandardScaler

data[['price', 'f','t']] = StandardScaler().fit_transform(data[['price', 'f','t']])

data.head(5)
from sklearn.model_selection import train_test_split

xtrain, xtest  = train_test_split(data,test_size = 0.5)
xtrain
# extra findings

# since no one has taken this dataset no kernels founded