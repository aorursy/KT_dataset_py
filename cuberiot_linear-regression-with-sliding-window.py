! wget https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv





import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression
data = pd.read_csv('time_series_covid19_confirmed_global.csv')



data = data.iloc[:, 4:].sum(0)
data = data.to_numpy()
# indices = [0, 1, 2, 3, ...]

indices = np.arange(data.shape[0])
x, y = indices, data
y_train = y[:-7]

y_test = y[-7:]
window_size = 10



x_train = []



for i in range(y_train.shape[0] - window_size):

    x_train.append(y_train[i : i + window_size])

x_train = np.array(x_train)



y_train = y_train[window_size - y_train.shape[0]:]
x_train.shape, y_train.shape
model = LinearRegression()



model.fit(x_train, y_train)
x = y_train[-window_size:]



for i in range(7):

    # all of x -> the next day

    y_hat = model.predict([x[-window_size:]])[0]

    x = np.append(x, y_hat)



predictions = np.round(x[-7:]).astype(int)
predictions
import matplotlib.pyplot as plt



plt.plot(predictions, label = 'Pred')

plt.plot(y_test, label = 'True')

plt.legend(loc = 'best')

plt.show()
plt.plot(np.concatenate((y_train, predictions)), label = 'Pred')

plt.plot(np.concatenate((model.predict(x_train), y_test)), label = 'True')

plt.legend(loc = 'best')

plt.show()
print((y_test - predictions) ** 2)

mse = np.mean(

        (y_test - predictions) ** 2

)
print(f'MSE: {mse:.4e}')