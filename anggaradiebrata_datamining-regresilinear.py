# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

from sklearn import linear_model





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataframe =  pd.read_csv("../input/melbourne-housing-market/Melbourne_housing_FULL.csv")
dataset = dataframe.dropna()
dataset.head(150)
df_x=pd.DataFrame(dataset.Landsize)

df_y=pd.DataFrame(dataset.Price)
df_y.describe()
reg=linear_model.LinearRegression()

reg.fit(df_x, df_y)
predi = reg.predict(df_x)

predi[2]
df_x
df_y
reg.score(df_x,df_y)
reg.coef_
reg.intercept_
Landsize = 120;

Price = 1073340.60 + (37.36 * Landsize);

print(Price)