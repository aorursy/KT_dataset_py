# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


file =pd.read_csv("../input/PlasticSales.csv")

import matplotlib.pylab as plt

get_ipython().run_line_magic('matplotlib', 'inline')

file.head()

from datetime import datetime

con=file['Month']



file['Month']=pd.to_datetime(file['Month'])

file.set_index('Month', inplace=True)





ts = file['Sales']

ts.head(10)



from statsmodels.graphics.tsaplots import plot_acf

sales_diff =file.diff(periods=1)



sales_diff=sales_diff[1:]

sales_diff

plot_acf(sales_diff)
plt.plot(sales_diff)
x=file.Sales

train=x[0:45]

test=x[44:60]
from statsmodels.tsa.ar_model import AR

from sklearn.metrics import mean_squared_error
armodel=AR(train)

armodel_fit =armodel.fit()

train.shape

a=armodel_fit.predict(start=45 ,end=60)

plt.plot(test)

plt.plot(a,color="red")
mean_squared_error(a,test)

from statsmodels.tsa.arima_model  import ARIMA

modelarima =ARIMA(train,order=(3,1,1))

arimafit =modelarima.fit()

print(arimafit.aic)

predictarim =arimafit.forecast(steps=16)[0]

plt.plot(predictarim)



predictarim.shape
plt.plot(test)

mean_squared_error(predictarim,test)
import sys

import itertools

p=range(0,10)

d=q=range(0,5)

pdf=list(itertools.product(p,d,q))



import warnings

warnings.filterwarnings("ignore")

for allthevalue in pdf:

    try:

      model_arima=ARIMA(train,order=allthevalue)

      fitarimaa=model_arima.fit()

      print(allthevalue,fitarimaa.aic)

    except:

        continue

# (7,2,2) gave us the least aic value so we will choose it to build the model

from statsmodels.tsa.arima_model  import ARIMA

modelarima1 =ARIMA(train,order=(7,2,2))

arimafit1 =modelarima1.fit()

prediction1 =arimafit1.forecast(steps=16)[0]

print(arimafit1.aic)