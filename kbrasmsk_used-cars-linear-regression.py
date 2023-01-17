# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/autos.csv", encoding="windows-1252")
df.head()
df.tail()
df.columns
df.info()
df.describe()
df.shape
print("Seller count : \n",df.seller.value_counts())

print("OfferType count : \n",df.offerType.value_counts())

print("Model count : \n",df.model.value_counts())

df.drop(['dateCrawled','name','abtest','notRepairedDamage','dateCreated','vehicleType','lastSeen','postalCode','monthOfRegistration','nrOfPictures'], axis=1, inplace=True)
df.head()
x=df.iloc[:,[3,5,7]].values

y=df.price.values.reshape(-1,1)

reg=LinearRegression()

reg.fit(x,y)

print("b0:",reg.intercept_)

print("b1 b2 b3:",reg.coef_)

reg.predict(np.array([[1993,0,150000]]))