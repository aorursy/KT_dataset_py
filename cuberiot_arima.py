! wget https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from statsmodels.tsa.arima.model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
data = pd.read_csv('time_series_covid19_confirmed_global.csv').iloc[:, 4:].sum(0)
train = data[:-7]

test = data[-7:]
train_ = train.to_numpy()

train_ = train_[1:] - train_[:-1]

train_ = train_[1:] - train_[:-1]

train_ = train_[1:] - train_[:-1]



plt.plot(train_)

plt.show()



plot_pacf(train_);
# p = 9, d = 2

arima = ARIMA(train, order = (9, 3, 0))



res = arima.fit()
plt.plot(res.resid.to_numpy())

plt.title('Residual Error')

plt.show()
x_test = pd.date_range(pd.to_datetime('8/18/20'), pd.to_datetime('8/24/20'))

# Both inclusive



predictions = np.empty(7, dtype = int)

for i, x in enumerate(x_test):

    y_hat = np.round(res.predict(x)[0]).astype(int)

    print('Date: x\t', y_hat)

    predictions[i] = y_hat
print(f'MSE: {np.mean(np.square(predictions - test.to_numpy())):.4e}')
plt.plot(predictions, label = 'pred')

plt.plot(test.to_numpy(), label = 'true')

plt.legend(loc = 'best')

plt.show()