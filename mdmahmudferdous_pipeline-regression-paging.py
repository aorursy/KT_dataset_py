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
data=pd.read_csv('/kaggle/input/telco-paging-a-interface/Paging_Analysis_A_interface.csv')

data.head()
data.shape
data1=data[['Success_Rate','Response','Failures']]

data1.head()
import matplotlib.pyplot as plt

%matplotlib inline

plt.plot(data1['Response'], data1['Success_Rate'])

plt.show()
plt.plot(data1['Failures'], data1['Success_Rate'])

plt.show()
import seaborn as sns

sns.pairplot(data1)
X=data1[['Response','Failures']]

y=data1[['Success_Rate']]

print(X.shape)

print(y.shape)
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
pipe=Pipeline([

        ('Scaler',StandardScaler()),

        ('Poly', PolynomialFeatures(3)),

        ('Predictor', LinearRegression())      

])
model_lr=pipe.fit(X,y)

model_lr.score(X,y)