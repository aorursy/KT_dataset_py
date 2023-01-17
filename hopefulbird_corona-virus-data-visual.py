# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df  = pd.read_csv('/kaggle/input/corona-confirmed-cases-15-feb-al-jajeera/Update Corona.csv')

df2 =  pd.read_csv('/kaggle/input/bd-corona-data-28-03-2020/bd.csv')
df
df2
df.dtypes
df.describe()
df['country']
df.columns
df
df.columns
plt.scatter(df['continent'] , df['count'] , s = 100, alpha = 0.2)

plt.show()
italy = [3, 3, 3, 3, 4, 21, 79, 157, 229, 323, 470, 655, 889, 1128, 1701, 2036, 2502, 3089, 3858, 4636, 5883, 7375,

         9172, 10149, 12462, 15113, 17660, 21157, 24747, 27980, 31506, 35713, 41035, 47021, 53578, 59138, 63927, 69176, 80122]



usa = [15, 15, 15, 15, 15, 35, 35, 35, 53, 57, 60, 60, 63, 68, 75, 100, 124, 158, 221, 319, 435, 541, 704,

       994, 1301, 1630, 2183, 2770, 3613, 4596, 6344, 9197, 13779, 19367, 24192, 33592, 43781, 54881, 85712]



ban = [1, 3, 3, 6, 8, 8, 11, 14, 24, 27, 33, 39, 44, 48]





plt.plot(italy, label="Italy")

plt.plot(usa, label="USA")

plt.plot(ban, label="Bangladesh")



plt.xlabel("Number of Days")

plt.ylabel("Number Of Cases")

plt.title("Corona Data Visual")

plt.legend()



plt.show()