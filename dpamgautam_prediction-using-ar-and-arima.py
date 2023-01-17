import numpy as np

import pandas as pd

from pandas import datetime

import matplotlib.pyplot as plt
def parser(x):

    return datetime.strptime(x, "%Y-%m")
data = pd.read_csv("../input/sales-cars.csv", index_col = 0, parse_dates=[0], date_parser = parser)
data.shape
data.head()
data.columns
data.plot()
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(data)
data.shift(1)
diff_data = data.diff(periods=1)
diff_data.head()
diff_data = diff_data[1:]
plot_acf(diff_data)
diff_data.head()
diff_data.plot()
x = data.values

train = x[:27]

test = x[27:]
train.size, test.size
from statsmodels.tsa.ar_model import AR

from sklearn.metrics import mean_squared_error

ar_model = AR(train)

ar_fit = ar_model.fit()
predictions = ar_fit.predict(start=27, end=36)
plt.plot(test)

plt.plot(predictions, color='red')
from statsmodels.tsa.arima_model import ARIMA

arima_model = ARIMA(train, order=(2,1,0))

arima_fit = arima_model.fit()
print(arima_fit.aic)
predictions = arima_fit.forecast(steps=9)[0]
print(predictions)
plt.plot(test)

plt.plot(predictions, color='red')
import itertools

p=d=q = range(0,5)

pdq = list(itertools.product(p,d,q))
pdq
import warnings

warnings.filterwarnings('ignore')
for param in pdq:

    try:

        arima_model = ARIMA(train, order=param)

        arima_fit = arima_model.fit()

        aic = arima_fit.aic

        print(param, aic)

    except:

        continue
arima_model = ARIMA(train, order=(3,2,4))

arima_fit = arima_model.fit()

predictions = arima_fit.forecast(steps=9)[0]

print(predictions)
plt.plot(test)

plt.plot(predictions, color='red')