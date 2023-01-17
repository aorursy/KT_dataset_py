# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/testee/copiadoras.csv")

df = df.drop('Unnamed: 2', 1)

df.columns = ['Y', 'X']

df["Div"] = df["Y"]/df["X"]

df.head()
df.describe()
plt.hist(df["Y"], bins= 10)

plt.show()
plt.hist(df["X"], bins= 10)

plt.show()
plt.scatter(df["X"], df["Y"])

plt.show()
plt.hist(df["Div"], bins = 10)

plt.show()
sns.jointplot(x="X", y="Y", data=df, kind="reg")
# é necessário adicionar uma constante a matriz X

X_sm = sm.add_constant(df["X"])

# OLS vem de Ordinary Least Squares e o método fit irá treinar o modelo

results = sm.OLS(df["Y"], X_sm).fit()

# mostrando as estatísticas do modelo

results.summary()

# mostrando as previsões para o mesmo conjunto passado

#results.predict(X_sm)
#E

#H0 x = 14

#H1 x > 14

from scipy.stats import t



mean = np.mean(df["Div"])

std = np.std(df["Div"])

num = len(df["Div"])

x = 14

hp = (mean - x)/(std/np.sqrt(num))





alpha = 0.05

cv = t.ppf(1.0 - alpha, num-1)

p = (1 - t.cdf(abs(hp), num)) * 2

print("Região Crítica do teste é ", cv, "         ", "Estatística do teste é ", hp)

print("P-Valor é ", p )
from sklearn.linear_model import LinearRegression



x = np.array(df["X"]).reshape(-1, 1)

y = np.array(df["Y"])



rg = LinearRegression()

rg.fit(x,y)

print(rg.predict(np.array(11).reshape(-1,1)))
from pandas import read_excel

from pandas import datetime

from matplotlib import pyplot

 

"""def parser(x):

	return datetime.strptime('190'+x, '%Y-%m')"""

 

series = read_excel('/kaggle/input/testes/assinantes.xlsx', header=0, parse_dates=[0], index_col=0, squeeze=True)

print(series.head())

series.plot()

pyplot.show()
path = "/kaggle/input/testes/assinantes.xlsx" 

dataset = pd.read_excel(path)



dataset['Período'] = pd.to_datetime(dataset['Período'],infer_datetime_format=True) #convert from string to datetime

indexedDataset = dataset.set_index(['Período'])

indexedDataset.head()
#Determine rolling statistics

rolmean = indexedDataset.rolling(window=6).mean() 

rolstd = indexedDataset.rolling(window=6).std()

print(rolmean,rolstd)
from statsmodels.tsa.arima_model import ARIMA



# 1,1,2 ARIMA Model

model = ARIMA(dataset["Assinantes"], order=(1,1,1))

model_fit = model.fit(disp=0)

print(model_fit.summary())
# Plot residual errors

residuals = pd.DataFrame(model_fit.resid)

fig, ax = plt.subplots(1,2)

residuals.plot(title="Residuals", ax=ax[0])

residuals.plot(kind='kde', title='Density', ax=ax[1])

plt.show()
# Actual vs Fitted

model_fit.plot_predict(dynamic=False)

plt.show()
dataset_forecast = pd.read_excel("/kaggle/input/forecast/Test forecast.xlsx")

type(dataset_forecast)

from statsmodels.tsa.stattools import acf

data = pd.concat([dataset,dataset_forecast], ignore_index=True)

# Create Training and Test

train = data[:21]["Assinantes"]

test = data[21:]["Assinantes"]
# Build Model

#model = ARIMA(train, order=(3,2,1))  

model = ARIMA(train, order=(1, 1, 1))  

fitted = model.fit(disp=-1)  



# Forecast

fc, se, conf = fitted.forecast(5, alpha=0.05)  # 95% conf



# Make as pandas series

fc_series = pd.Series(fc, index=test.index)

lower_series = pd.Series(conf[:, 0], index=test.index)

upper_series = pd.Series(conf[:, 1], index=test.index)



# Plot

plt.figure(figsize=(20,5), dpi=100)

plt.plot(train, label='training')

plt.plot(test, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()