import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_log_error
def RMSLE(y_true, y_pred):

    

    if len(y_pred[y_pred<0])>0:

        y_pred = np.clip(y_pred, 0, None)

    

    return np.sqrt(mean_squared_log_error(y_true, y_pred))









def MAPE(y_true, y_pred):

    

    if len(y_true[y_true==0])>0: # Use WAPE if there are zeros

        return sum(np.abs(y_true - y_pred)) / sum(y_true) * 100

        

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv', usecols=['meter_reading'])
rmsle = list()

mape = list()



y = df['meter_reading'].values



for dev in np.arange(0, 1, 0.01):

    

    y_noise = np.random.normal(y, dev*y)

    

    rmsle.append(RMSLE(y, y_noise))

    mape.append(MAPE(y, y_noise))
df_results = pd.DataFrame([mape, rmsle], index=['MAPE', 'RMSLE']).T

df_results.sort_values('MAPE', inplace=True)
plt.figure(figsize=(16,16))

plt.plot(df_results['MAPE'].values, df_results['RMSLE'].values)

plt.xlabel('MAPE %')

plt.ylabel('RMSLE')

plt.grid(True, axis='both')

plt.show()