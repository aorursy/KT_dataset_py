# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/corona-virus-brazil/brazil_covid19.csv")

df.head()
df_region = df

regio_group = df_region.groupby(["region"])['date',"cases"].sum().reset_index()

regio_group =regio_group.sort_values(by='cases', ascending=False)

regio_group
df_ceara = df[df["state"]=="Ceará"]

df_ceara
dgc = df_ceara[df_ceara["cases"]>0].groupby(["date"])["cases"].sum().reset_index()

dg_c = dgc.sort_values(by='date', ascending=True)

dg_c.head(2)



df_ce = pd.DataFrame(dg_c.set_index('date').diff()).reset_index()

df_ce = df_ce.sort_values('date', ascending = True) 

df_ce = df_ce[1:]

df_ce.tail(10).style.background_gradient(cmap='OrRd')
plt.plot(df_ce["date"], df_ce["cases"])

plt.show()
dg = df[df["cases"]>0].groupby(["date"])["cases"].sum().reset_index()

dg_n = dg.sort_values(by='date', ascending=True)

dg_n.head(2)
plt.plot(dg_n["date"], dg_n["cases"])

plt.show()
bra = pd.DataFrame(dg_n.set_index('date').diff()).reset_index()

bra = bra.sort_values('date', ascending = True) 

bra = bra[1:]

bra.tail(20).style.background_gradient(cmap='OrRd')
bra.sum()
plt.plot(bra["date"], bra["cases"])

plt.show()
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.ar_model import AR

from sklearn.metrics import mean_squared_error
bra.set_index("date", inplace=True)

bra.head(2)
X = bra.values

train = X[:len(bra)-5]

test = X[len(bra)-5:]

pred = []

X.size
import itertools

p=range(0,7)

q = range(0,3)

d = range(0,4)

pdq = list(itertools.product(p,d,q))

import warnings

warnings.filterwarnings('ignore')

for param in pdq:

    try:        

        model_arima = ARIMA(train,order=param)

        model_arima_fit = model_arima.fit()

        print(param,model_arima_fit.aic)

        predictions= model_arima_fit.forecast(steps=5)[0]

        mse = mean_squared_error(test,predictions)        

        print(mse)

    except:

        continue    

        
model_Arima = ARIMA(train, order=(2,0,0))

model_Arima_fit = model_Arima.fit()

print(model_Arima_fit.aic)
predictions = model_Arima_fit.forecast(steps=5)[0]

predictions
plt.plot(test)

plt.plot(predictions, color="red")

plt.show()

print(predictions.sum())

print(test.sum())
import itertools

p=range(0,7)

q = range(0,3)

d = range(0,4)

pdq = list(itertools.product(p,d,q))

import warnings

warnings.filterwarnings('ignore')

for param in pdq:

    try:        

        model_arima = ARIMA(X,order=param)

        model_arima_fit = model_arima.fit()

        print(param,model_arima_fit.aic)

        predictions= model_arima_fit.forecast(steps=5)[0]

        mse = mean_squared_error(test,predictions)        

        print(mse)

    except:

        continue    

        
model_Arima = ARIMA(X, order=(0,1,1))

model_Arima_fit = model_Arima.fit()

print(model_Arima_fit.aic)
predictions = model_Arima_fit.forecast(steps=10)[0]

predictions
plt.plot(predictions, color="red")

plt.show()
round(predictions.sum())