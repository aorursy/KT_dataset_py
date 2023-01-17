# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#reading data(CSV format)



df=pd.read_csv('/kaggle/input/house-price-dataset-for-simple-regression/house_prices.csv')
#checking first 5 rows

df.head()
#plotting scatter plot to see relation between LotArea and SalePrice



plt.scatter(df.LotArea,df.SalePrice)

plt.xlabel('LotArea')

plt.ylabel("SalePrice")

plt.title('house pricing based on LotArea')

#fitting simple regression to training set



reg=LinearRegression()

X=df[['LotArea']]

y=df.SalePrice

reg.fit(X,y)
#Predicting the test set results

y_pr=reg.predict(X)
plt.scatter(X,y)

plt.plot(X,y_pr,"r-")

plt.title('Housing Price ')

plt.xlabel('Area')

plt.ylabel('Price')

plt.show()

#plotting best fitted line and vertical line(area=14000)



plt.scatter(X,y)

plt.plot(X,y_pr,"r-")

#plotting vertical line from area=14000

plt.axvline(x=14000,color='black')

plt.title('Housing Price ')

plt.xlabel('Area')

plt.ylabel('Price')

plt.show()
#predicting the price of house for Area=14000



X=14000

reg.predict([[X]])
