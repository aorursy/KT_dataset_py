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
data=pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")

data2=pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

X=data2[['MSSubClass','LotFrontage','LotArea','YrSold']]

X= my_imputer.fit_transform(X)

data['LotFrontage'].fillna(data['LotFrontage'].mode()[0],inplace=True) 

#data2=['LotFrontage'].fillna(data['LotFrontage'].mode()[0],inplace=True) 

#data2.isnull().sum()

data


x=data[['MSSubClass','LotFrontage','LotArea','YrSold']]



y=data[['SalePrice']]



from sklearn.linear_model import LinearRegression

model=LinearRegression()

model.fit(x,y)



Y=model.predict(X)

Y=Y.flatten()

output = pd.DataFrame({'Id': data2.Id,'SalePrice':Y})

output.to_csv('submission.csv', index=False)
