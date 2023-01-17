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

import pandas as pd

import matplotlib.pyplot as plt
dataset =  pd.read_csv('../input/singapore-airbnb/listings.csv', index_col='id')

dataset.head()
# 1. Find out what categories exist and how many neighbourhood belong to each category by using the value_counts() method

dataset['neighbourhood'].value_counts()
# 2. Shows a summary of the numerical attributes

dataset.describe()
# 3. handling missing data (Replacing missing data with the mean value or others)

missing_values = dataset.isna().sum( axis = 0)

print(missing_values[missing_values > 0])
from sklearn.impute import SimpleImputer



# Drop rows where name is null

dataset = dataset.dropna(axis = 0, how ='any') 



# Replace missing data with mean

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose=0)

dataset['reviews_per_month'] = imputer.fit_transform(dataset[['reviews_per_month']]).ravel()



# Replace missing data with most_frequent

imputer2 = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent', verbose=0)

dataset['last_review'] = imputer2.fit_transform(dataset[['last_review']]).ravel()
# 4. Plot a histogram for each numerical attribute

fig = plt.figure(figsize = (20,20))

ax = fig.gca()

dataset.hist(ax = ax)
# 5. Splitting the Dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values

X_train, X_text, y_train, y_test = train_test_split(X, y, test_size=.2, random_state = 0)