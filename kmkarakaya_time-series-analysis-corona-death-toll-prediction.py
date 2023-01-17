import pandas as pd

# prophet by Facebook

from fbprophet import Prophet

from sklearn.metrics import mean_absolute_error

import warnings; warnings.simplefilter('ignore')

import matplotlib.pyplot as plt



#if you are running from COLAB copy the corona.csv file from  the link above in NOTES to your drive

#from google.colab import drive

#drive.mount('/content/drive')
import os

print(os.listdir("../input"))

df = pd.read_csv('../input/corona-turkey/corona.csv', sep=',')

df.head()
df= df[['DAY','DEAD_TODAY','DEAD_TOMORROW']].copy()

df.tail()
df['ds'] = pd.to_datetime(df['DAY'],  dayfirst = True)

df.plot(x='ds',   figsize=(10, 5))

newdf = df[['ds', 'DEAD_TODAY']].copy()

df.drop(['ds'], axis=1, inplace=True)

newdf.rename(columns={'DEAD_TODAY': 'y'}, inplace=True)



newdf.tail()
m = Prophet( )

m.fit(newdf)

horizon= 1

future = m.make_future_dataframe(periods=horizon)

forecast = m.predict(future)

forecast[['ds',  'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
#fig2 = m.plot_components(forecast)
MAE={}



MAE['Prophet'] =  mean_absolute_error(newdf['y'], forecast[:-horizon]['yhat'])



print("MAE : {}".format(MAE))
comparison= pd.DataFrame()

comparison['ds']=newdf['ds'].copy()

comparison['DEAD_TOMORROW']=df['DEAD_TOMORROW'].copy()

comparison['Prediction_Prophet'] = forecast[:-1].yhat

comparison.plot(title="comparison",x='ds',figsize=(20, 6))
window= 3

df['Prediction_ SMA_3'] = df.iloc[:,1].rolling(window=window).mean()

df.head()
MAE['SMA_3'] =  mean_absolute_error(df[2:-1]['DEAD_TOMORROW'], df[2:-1]['Prediction_ SMA_3'])

print("MAE : {}".format(MAE))
rstd = df.iloc[:,2].rolling(window=window).std()

bands = pd.DataFrame()

bands['DAY']= df['DAY'].copy()

bands['lower'] = df['Prediction_ SMA_3'] - 2 * rstd

bands['upper'] = df['Prediction_ SMA_3'] + 2 * rstd





bands = bands.join(df['DEAD_TOMORROW']).join(df['Prediction_ SMA_3'])

fig = plt.figure(figsize=(20, 6))

ax = bands.plot(title='Prediction_ SMA_3', figsize=(20, 6))

ax.fill_between(bands.index, bands['lower'], bands['upper'], color='#ADCCFF', alpha=0.4)

ax.set_xlabel('date')

ax.set_ylabel('DEAD TOMORROW')

ax.grid()



plt.show()
comparison['Prediction_SMA_3'] = df['Prediction_ SMA_3']

print(comparison.tail())

comparison.plot(title="comparison",x='ds',figsize=(20, 6))
df['Prediction_EMA_3'] = df.iloc[:,1].ewm(span=window,adjust=False).mean()

df.head()

MAE['EMA_3'] =  mean_absolute_error(df[1:-1]['DEAD_TOMORROW'], df[1:-1]['Prediction_EMA_3'])

print("MAE : {}".format(MAE))
rstd = df.iloc[:,2].rolling(window=window).std()

bands = pd.DataFrame()

bands['DAY']= df['DAY'].copy()

bands['lower'] = df['Prediction_EMA_3'] - 2 * rstd

bands['upper'] = df['Prediction_EMA_3'] + 2 * rstd

bands = bands.join(df['DEAD_TOMORROW']).join(df['Prediction_EMA_3'])

fig = plt.figure(figsize=(20, 6))

ax = bands.plot(title='Prediction_EMA_3', figsize=(20, 6))

ax.fill_between(bands.index, bands['lower'], bands['upper'], color='#ADCCFF', alpha=0.4)

ax.set_xlabel('date')

ax.set_ylabel('DEAD TOMORROW')

ax.grid()

plt.show()
comparison['Prediction_EMA_3'] = df['Prediction_EMA_3']

comparison.plot(title="comparison",x='ds',figsize=(20, 6))
print('Mean Absolute Errors (MAE): {}'.format(MAE))
errorsDF = pd.DataFrame(MAE, index=['MAE']) 

ax = errorsDF.plot.bar(rot=0, figsize=(10, 7))
rstd.tail()
bands.tail()