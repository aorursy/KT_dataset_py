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
housing=pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")
housing.head()
from sklearn.impute import SimpleImputer

imputer= SimpleImputer(strategy='median')
from sklearn.impute import SimpleImputer

imputer= SimpleImputer(strategy='median')

housing_num=housing.drop("ocean_proximity",axis=1)

imputer.fit(housing_num)

imputer.statistics_

x= imputer.transform(housing_num)
from sklearn.preprocessing import OneHotEncoder

cat_encoder=OneHotEncoder()

housin_num=housing[['ocean_proximity']]

housing_cat_1hot=cat_encoder.fit_transform(housing_num)

housing_cat_1hot.toarray()
