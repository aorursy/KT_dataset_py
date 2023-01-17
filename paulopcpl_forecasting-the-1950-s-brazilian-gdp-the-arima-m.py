import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.stattools import adfuller





%matplotlib inline

plt.rcParams["figure.figsize"] = (15,7)
dataset = pd.read_csv("../input/ipeadata22-02-2020-03-34.csv.csv")
dataset.describe()
dataset.drop('Unnamed: 2', axis = 1, inplace = True)

dataset.columns = ['year', 'GDP']
#Parse strings to datetime type

dataset = dataset[dataset['year'] <= 1950].copy()

dataset['year'] = pd.to_datetime(dataset['year'], format='%Y').dt.year
dataset.tail()
dataset['GDP_Change'] = (dataset['GDP'] - dataset['GDP'].shift())/dataset['GDP'].shift() * 100
dataset.head()
plt.subplot(2, 1, 1)

plt.plot(dataset['year'], dataset['GDP']/(10**3))

plt.title('Brazil GDP (1900 - 1950)')

plt.ylabel('Billions (R$)')



plt.subplot(2, 1, 2)

plt.plot(dataset['year'], dataset['GDP_Change'])

plt.title('Brazil yearly GDP Growth (1900 - 1950)')

plt.ylabel('%')

plt.hlines(y = 0, xmin = 1900, xmax = 1950, color = 'red', linestyle = 'dashed')
result = adfuller(dataset['GDP'])

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])
result = adfuller(dataset['GDP'].diff().dropna())

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])
result = adfuller(dataset['GDP'].diff().diff().dropna())

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])
plot_pacf(dataset['GDP'].diff().diff().dropna());
plot_acf(dataset['GDP'].diff().diff().dropna());
# fitting the models.

AR = [0, 1, 4]

MA = [0, 1, 5]

models = pd.DataFrame(columns = ['ARIMA', 'AIC', 'BIC'])

for p in range(len(AR)):

    for q in range(len(MA)):

        try:

            model = ARIMA(dataset['GDP'], order = (AR[p], 2, MA[q]))

            model_fit = model.fit(disp=0)

            values = pd.Series([(AR[p], 2, MA[q]), model_fit.aic, model_fit.bic], index = ['ARIMA', 'AIC', 'BIC'])

            models = models.append(values, ignore_index = True)

        except: pass

models
model = ARIMA(dataset['GDP'], order=(0,2,1))

model_fit = model.fit(disp=0)

print(model_fit.summary())
# Plot residual errors

residuals = pd.DataFrame(model_fit.resid)

residuals.set_index(pd.Series(list(range(1901, 1950))), inplace = True)

fig, ax = plt.subplots(1,2)

residuals.plot(title="Residuals", ax=ax[0]).hlines(y = 0, xmin = 1900, xmax = 1950, color = 'red', linestyles = 'dashed')

#plot residuals density

residuals.plot(kind='kde', title='Density', ax=ax[1])

plt.show()
residuals.describe()
dataset = pd.read_csv("../input/ipeadata22-02-2020-03-34.csv.csv")

dataset.drop('Unnamed: 2', axis = 1, inplace = True)

dataset.columns = ['year', 'GDP']

dataset = dataset[dataset['year'] <= 1960].copy()
next10 = model_fit.forecast(steps = 10, alpha = 0.05)

ci1 = []

ci2 = []

for i in range(len(next10[2])):

    ci1.append(next10[2][i][0])

    ci2.append(next10[2][i][1])
inSample = model_fit.predict(start=2, end=50, exog=None, typ='levels', dynamic=False)
plt.plot(dataset['year'], dataset['GDP'], color = 'orange', label = 'Real GDP Value')

plt.plot(range(1902, 1951), inSample, color = 'blue', label = 'In Sample Prediction')

plt.plot(range(1951, 1961), next10[0], color = 'purple', label = 'Out of Sample Prediction')

plt.plot(range(1951, 1961), ci1, linestyle = 'dashed', color = 'gray')

plt.plot(range(1951, 1961), ci2, linestyle = 'dashed', color = 'gray', label = '95% Confidence Interval')

plt.title('Brazilian GDP - Real Value and Forecast')

plt.ylabel('BRL')

plt.xlabel('Year')

plt.legend()

plt.show()
for i in range(len(next10[0])):

    if dataset['GDP'][51 + i] > next10[2][i][1]:

        print('The Real GDP is outside of the 95% confidence interval at the year 195' + str(i) + ' and forward.')

        break