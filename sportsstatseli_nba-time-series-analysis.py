# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd



data = pd.read_csv("/kaggle/input/lowryp/3.csv")

data2 = pd.read_csv("/kaggle/input/lebron/lbj.csv")



pts = data[["Season", "PTS"]]

lj = data2[["Season", "PTS"]]

display(lj.tail())

display(pts.tail())
import matplotlib.pyplot as plt

%matplotlib inline



pts['Rolling_Mean'] = pts['PTS'].rolling(window = 2).mean()

lj['Rolling_Mean'] = lj['PTS'].rolling(window = 2).mean()



fig = plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(111)

ax1.scatter(lj["Season"], lj["PTS"], c="blue", label = "Lebron")

ax1.scatter(pts["Season"], pts["PTS"], c="red", label= "Lowry")

plt.plot(pts["Season"], pts["Rolling_Mean"], c="red")

plt.plot(lj["Season"], lj["Rolling_Mean"], c="blue")

plt.legend(loc='lower right')



plt.title('PPG v.s Season for Lebron & Lowry (Rolling Mean)')

plt.xlabel('Season')

plt.ylabel('PPG')

plt.show()
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt



fit1 = SimpleExpSmoothing(lj["PTS"]).fit(smoothing_level=0.5,optimized=True)

fcast1 = fit1.forecast(5)



fig = plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(111)

ax1.scatter(lj["Season"], lj["PTS"], c="blue", label = "Lebron")

ax1.scatter(pts["Season"], pts["PTS"], c="red", label= "Lowry")

plt.plot(pts["Season"], pts["Rolling_Mean"], c="red")

plt.plot(lj["Season"], lj["Rolling_Mean"], c="blue")

fcast1.plot(marker='o', color='purple', legend=True)

plt.legend(loc='lower right')



plt.title('PPG v.s Season for Lebron & Lowry (Exponential Smoothing)')

plt.xlabel('Season')

plt.ylabel('PPG')

plt.show()
from pylab import *

from sklearn.metrics import r2_score



xp = np.linspace(1, 17, 17)

z = np.linspace(1, 16, 16)

p4_lebron = np.poly1d(np.polyfit(xp, lj["PTS"], 5))

p4_lowry = np.poly1d(np.polyfit(z, pts["PTS"], 5))





fig = plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(111)

ax1.scatter(lj["Season"], lj["PTS"], c="blue", label = "Lebron")

ax1.scatter(pts["Season"], pts["PTS"], c="red", label= "Lowry")

plt.plot(xp, p4_lebron(xp), c='blue')

plt.plot(z, p4_lowry(z), c='red')

plt.legend(loc='lower right')



plt.title('PPG v.s Season for Lebron & Lowry (Polynomial Regression)')

plt.xlabel('Season')

plt.ylabel('PPG')

plt.show()



r2_lebron = r2_score(lj["PTS"], p4_lebron(xp))

r2_lowry = r2_score(pts["PTS"], p4_lowry(z))



print("R-Squared error for Lebron & Lowry fits respectively:")

print(r2_lebron)

print(r2_lowry)

optimal_n = None

best_mse = None

db = lj[['PTS']].values.astype('float32')

mean_results_for_all_possible_n_values = np.zeros(int(len(db) / 2 - 2))

for n in range(3, int(len(db) / 2 + 1)):

    mean_for_n = np.zeros(len(db) - n)

    for i in range(0, len(db) - n):

        mean_for_n[i] = np.power(np.mean(db[:, 0][i:i+n]) - db[i + n][0], 2)

    mean_results_for_all_possible_n_values[n - 3] = np.mean(mean_for_n)

optimal_n = np.argmin(mean_results_for_all_possible_n_values) + 3

best_mse = np.min(mean_results_for_all_possible_n_values)

print("MSE = %s" % mean_results_for_all_possible_n_values)

print("Best MSE = %s" % best_mse)

print("Optimal n = %s" % optimal_n)



print("MA = %s" % np.mean(db[:, 0][len(db) - optimal_n:len(db)]))



forecast = np.zeros(len(db) + 1)

for i in range(0, optimal_n):

    forecast[i] = db[i][0]

for i in range(0, len(db) - optimal_n + 1):

        forecast[i+optimal_n] = np.mean(db[:, 0][i:i+optimal_n])

plt.plot(db[:, 0],label = 'real data: LBJ', c="blue")

plt.plot(forecast, label = 'forecast', c="purple")

plt.title('PPG v.s Season for Lebron (Moving Average)' )

plt.legend()

plt.show()

optimal_n = None

best_mse = None

db = pts[['PTS']].values.astype('float32')



mean_results_for_all_possible_n_values = np.zeros(int(len(db) / 2 - 2))

for n in range(3, int(len(db) / 2 + 1)):

    mean_for_n = np.zeros(len(db) - n)

    for i in range(0, len(db) - n):

        mean_for_n[i] = np.power(np.mean(db[:, 0][i:i+n]) - db[i + n][0], 2)

    mean_results_for_all_possible_n_values[n - 3] = np.mean(mean_for_n)

optimal_n = np.argmin(mean_results_for_all_possible_n_values) + 3

best_mse = np.min(mean_results_for_all_possible_n_values)

print("MSE = %s" % mean_results_for_all_possible_n_values)

print("Best MSE = %s" % best_mse)

print("Optimal n = %s" % optimal_n)

print("MA = %s" % np.mean(db[:, 0][len(db) - optimal_n:len(db)]))



forecast = np.zeros(len(db) + 1)

for i in range(0, optimal_n):

    forecast[i] = db[i][0]

for i in range(0, len(db) - optimal_n + 1):

        forecast[i+optimal_n] = np.mean(db[:, 0][i:i+optimal_n])

plt.plot(db[:, 0],label = 'real data: Lowry', c="red")

plt.plot(forecast, label = 'forecast', c="orange")

plt.title('PPG v.s Season for Lowry (Moving Average)')



plt.legend()

plt.show()
!pip install pmdarima
import pmdarima as pm



model = pm.auto_arima(lj["PTS"],

                        start_p=1, start_q=1,

                        test='adf',      

                        max_p=3, max_q=3, 

                        m=1,              

                        d=None,           

                        seasonal=False, 

                        start_P=0, D=0, trace=True, 

                        error_action='ignore', suppress_warnings=True, 

                        stepwise=True)



forecast = model.predict(n_periods=5) 



forecast = pd.DataFrame(forecast,columns=["PTS"])

total = lj.append(forecast, ignore_index=False)

x = np.linspace(1, len(lj)+5,len(lj)+5 )



print("Forecast of PPG for Lebron for the next 5 seasons:" )

print(forecast)



fig = plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(111)

ax1.scatter(lj["Season"], lj["PTS"], c="blue", label = "Real data")

plt.plot(lj["Season"], lj["Rolling_Mean"], c="purple", linestyle='dashed', label = "Rolling Mean")

plt.plot(x, total["PTS"], c="green", label = "ARIMA")

plt.legend(loc='lower right')

plt.title('PPG v.s Season for Lebron (ARIMA)')

plt.xlabel('Season')

plt.ylabel('PPG')

plt.show()
from statsmodels.tsa.arima_model import ARIMA

from matplotlib import pyplot



xp = np.linspace(1, 17, 17)

y = lj["PTS"]



model = ARIMA(y, order=(5,2,0))

#ORDER: p,d,q

#sets the lag value to 5 for autoregression, 

#uses a difference order of 2 to make the time series stationary, 

#and uses a moving average model of 0.

model_fit = model.fit(disp=1)

print(model_fit.summary())



residuals = pd.DataFrame(model_fit.resid)

residuals.plot()



residuals.plot(kind='kde')

print(residuals.describe())
%matplotlib inline



log_e = np.log(np.sqrt(np.log(pts[["PTS"]])))

pts[["PTS"]] = log_e

pts['Rolling_Mean'] = pts['PTS'].rolling(window = 2).mean()





log_d = np.log(np.log(lj[["PTS"]]))

lj[["PTS"]] = log_d

lj['Rolling_Mean'] = lj['PTS'].rolling(window = 2).mean()



fig = plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(111)

ax1.scatter(lj["Season"], lj["PTS"], c="blue", label = "Lebron (log)")

ax1.scatter(pts["Season"], pts["PTS"], c="red", label= "Lowry (sqrt)")

plt.plot(pts["Season"], pts["Rolling_Mean"], c="red")

plt.plot(lj["Season"], lj["Rolling_Mean"], c="blue")

plt.legend(loc='lower right')



plt.title('PPG v.s Season for Lebron & Lowry')

plt.xlabel('Season')

plt.ylabel('PPG')

plt.show()