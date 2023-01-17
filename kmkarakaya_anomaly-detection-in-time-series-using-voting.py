import pandas as pd
# prophet by Facebook
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error
import warnings; warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
from IPython.display import HTML
import os


pd.set_option('display.max_columns', None)
url='../input/Min-Max Daily Analyse.csv'
df = pd.read_csv(url, sep=',')
df.head()

#url = 'https://raw.githubusercontent.com/kmkarakaya/ML_tutorials/master/data/Min-Max%20Daily%20Analyse.csv'
#df = pd.read_csv(url, sep=';')
#df.head()
df['Range'] = df['Max']-df['Min']
df.head()
import datetime 
day_name= ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
df['Day'] = [ day_name[i] for i in pd.to_datetime(df['Date']).dt.dayofweek]
df= df[['Date','Day','Min','Max','Range']]
df.head()
df.describe()
#df['ds'] = pd.to_datetime(df['Day'],  dayfirst = True)
df.plot(x='Date',   figsize=(15, 5))

def prediction_Prophet(feature):
  dfNew = pd.DataFrame()
  dfNew['ds'] = pd.to_datetime(df['Date'],  dayfirst = True)
  dfNew['y'] = df[[feature]].copy()
  
  #print(dfNew.tail())

  m = Prophet(daily_seasonality=True )
  m.fit(dfNew)
  horizon= 1
  future = m.make_future_dataframe(periods=horizon)
  forecast = m.predict(future)
  print('\nForcasted  {} values \n {}\n'.format(feature, forecast[['ds',  'yhat', 'yhat_lower', 'yhat_upper']].tail()))
  fig1 = m.plot(forecast)
  #fig2 = m.plot_components(forecast)
  return forecast
pred=prediction_Prophet('Range')

df['Range_By_Prophet']=pred['yhat_upper']

print('Anamolies for range values\n', df[df['Range']>df['Range_By_Prophet']][['Date','Day','Range','Range_By_Prophet']])
pred=prediction_Prophet('Min')

df['Min_By_Prophet']=pred['yhat_lower']
print('Anamolies for min values\n', df[df['Min']<df['Min_By_Prophet']][['Date','Day','Min','Min_By_Prophet']])

pred=prediction_Prophet('Max')

df['Max_By_Prophet']=pred['yhat_upper']
print('Anamolies for Max values\n', df[df['Max']>df['Max_By_Prophet']][['Date','Day','Max','Max_By_Prophet']])
df.plot(title="comparison",x='Date',y=['Min','Max', 'Min_By_Prophet','Max_By_Prophet'],figsize=(20, 6))
df.plot(title="comparison",x='Date',y=['Range','Range_By_Prophet'],figsize=(20, 6))
print('Mean of Min', df['Min'].mean())
print('Standart Deviation of Min', df['Min'].std())
print('Expected minimum value for of Min', df['Min'].mean()-2*df['Min'].std())
df['Min_Calculated']=df['Min'].mean()-2*df['Min'].std()
print('Anamolies for Min values\n', df[df['Min']<df['Min_Calculated']][['Date','Day','Min','Min_Calculated']])
print('Mean of Max', df['Max'].mean())
print('Standart Deviation of Max', df['Max'].std())
print('Expected minimum value for of Max', df['Max'].mean()+2*df['Max'].std())
df['Max_Calculated']=df['Max'].mean()+2*df['Max'].std()
print('Anamolies for Max values\n', df[df['Max']>df['Max_Calculated']][['Date','Day','Max','Max_Calculated']])
print('Mean of Range', df['Range'].mean())
print('Standart Deviation of Range', df['Range'].std())
maxRange=df['Range'].mean()+2*df['Range'].std()
print('Expected maximum value for of Range', maxRange)
df['Range_Calculated']=maxRange
print('Anamolies for Range values\n', df[df['Range']>df['Range_Calculated']][['Date','Day','Range','Range_Calculated']])
df.plot(title="comparison",x='Date',y=['Min','Max', 'Min_Calculated','Max_Calculated'],figsize=(20, 6))

df.plot(title="Range",x='Date',y=['Range','Range_Calculated'],figsize=(20, 6))
CodesOfInterest=['anomaly']
def hover(hover_color="#ffff99"):
    return dict(selector="tr:hover",
                props=[("background-color", "%s" % hover_color)])
def showSummary(fontSize='12px'):
  summary =pd.DataFrame()
  
  summary= anomaly[(anomaly.isin(CodesOfInterest)==True).any(1)]
  styles = [
    hover(),
    dict(selector="th", props=[("font-size", fontSize),
                               ("text-align", "center")]),
    dict(selector="tr", props=[("font-size", fontSize),
                               ("text-align", "center")]),      
    dict(selector="caption", props=[("caption-side", "bottom")])
  ]
  html = (summary.style.set_table_styles(styles)
          .set_caption("Hover to highlight."))
  print(' Number of detected anomalies: ', len(summary) )
  return html
anomaly = pd.DataFrame()
anomaly = df[['Date','Day','Min','Max','Range']].copy()
anomaly['Min_anomaly_Prophet']= df['Min']
anomaly['Max_anomaly_Prophet']= df['Max']
anomaly['Range_anomaly_Prophet']=df['Range']

anomaly['Min_anomaly_Calculated']= df['Min']
anomaly['Max_anomaly_Calculated']= df['Max']
anomaly['Range_anomaly_Calculated']= df['Range']

df.columns
anomaly['Min_anomaly_Prophet'][df['Min']<df['Min_By_Prophet']]= 'anomaly'
anomaly['Min_anomaly_Prophet'][df['Min']>=df['Min_By_Prophet']]= ''

anomaly['Max_anomaly_Prophet'][df['Max']>df['Max_By_Prophet']]= 'anomaly'
anomaly['Max_anomaly_Prophet'][df['Max']<=df['Max_By_Prophet']]= ''

anomaly['Range_anomaly_Prophet'][df['Range']>df['Range_By_Prophet']]= 'anomaly'
anomaly['Range_anomaly_Prophet'][df['Range']<=df['Range_By_Prophet']]= ''

anomaly['Min_anomaly_Calculated'][df['Min']<df['Min_Calculated']]= 'anomaly'
anomaly['Min_anomaly_Calculated'][df['Min']>=df['Min_Calculated']]= ''

anomaly['Max_anomaly_Calculated'][df['Max']>df['Max_Calculated']]= 'anomaly'
anomaly['Max_anomaly_Calculated'][df['Max']<=df['Max_Calculated']]= ''

anomaly['Range_anomaly_Calculated'][df['Range']>df['Range_Calculated']]= 'anomaly'
anomaly['Range_anomaly_Calculated'][df['Range']<=df['Range_Calculated']]= ''



showSummary('11px')

def predict_SMA(feature):
  window= 7
  sma = df[feature].rolling(window=window).mean()
  rstd = df[feature].rolling(window=window).std()
  bands = pd.DataFrame()
  bands['Date']=  (df['Date']).copy()
  bands['Date'] = pd.to_datetime(bands['Date'], dayfirst=True)
  bands['sma'] = sma 
  bands['lower'] = sma - 2 * rstd
  bands['upper'] = sma + 2 * rstd
  bands = bands.join(df[feature])
  bands = bands.set_index('Date')
  fig = plt.figure(figsize=(20, 6))
  ax = bands.plot(title=feature,  figsize=(20, 6))
  ax.fill_between(bands.index, bands['lower'], bands['upper'], color='#ADCCFF', alpha=0.4)
  ax.set_xlabel('Date')
  ax.set_ylabel(feature)
  ax.grid()
  plt.show()
  return bands
bands = predict_SMA('Min')
bands.reset_index(inplace=True)
min= df['Min'].min()
bands['lower'].fillna(min , inplace=True)
df['Min_SMA']= bands['lower'].copy()
print('Anamolies for SMA_Min values\n', df[df['Min']<df['Min_SMA']][['Date','Min', 'Min_SMA']])


bands = predict_SMA('Max')
bands.reset_index(inplace=True)
max= df['Max'].max()
bands['upper'].fillna(max , inplace=True)

df['Max_SMA']= bands['upper'].copy()
print('Anamolies for Max_SMA values\n', df[df['Max']>df['Max_SMA']][['Date','Max', 'Max_SMA']])
bands = predict_SMA('Range')
bands.reset_index(inplace=True)
max= df['Range'].max()
bands['upper'].fillna(max , inplace=True)
df['Range_SMA']= bands['upper'].copy()
print('Anamolies for Range_SMA values\n', df[df['Range']>=df['Range_SMA']][['Date','Range', 'Range_SMA']])
anomaly['Min_anomaly_SMA']= df['Min']
anomaly['Max_anomaly_SMA']= df['Max']
anomaly['Range_anomaly_SMA']= df['Range']

anomaly['Min_anomaly_SMA'][df['Min']<df['Min_SMA']]= 'anomaly'
anomaly['Min_anomaly_SMA'][df['Min']>=df['Min_SMA']]= ''

anomaly['Max_anomaly_SMA'][df['Max']>df['Max_SMA']]= 'anomaly'
anomaly['Max_anomaly_SMA'][df['Max']<=df['Max_SMA']]= ''

anomaly['Range_anomaly_SMA'][df['Range']>df['Range_SMA']]= 'anomaly'
anomaly['Range_anomaly_SMA'][df['Range']<=df['Range_SMA']]= ''

showSummary('10px')
def predict_EMA(feature):
  window= 3
  ema = df[feature].ewm(span=window,adjust=False).mean()
  rstd = df[feature].rolling(window=window).std()
  bands = pd.DataFrame()
  bands['Date']=  (df['Date']).copy()
  bands['Date'] = pd.to_datetime(bands['Date'], dayfirst=True)
  bands['ema'] = ema 
  bands['lower'] = ema - 2 * rstd
  bands['upper'] = ema + 2 * rstd
  bands = bands.join(df[feature])
  bands = bands.set_index('Date')
  fig = plt.figure(figsize=(20, 6))
  ax = bands.plot(title=feature,  figsize=(20, 6))
  ax.fill_between(bands.index, bands['lower'], bands['upper'], color='#ADCCFF', alpha=0.4)
  ax.set_xlabel('Date')
  ax.set_ylabel(feature)
  ax.grid()
  plt.show()
  return bands
bands= predict_EMA('Min')
bands.reset_index(inplace=True)
min= df['Min'].min()
bands['lower'].fillna(min , inplace=True)
df['Min_EMA']= bands['lower'].copy()
print('Anamolies for EMA_Min values\n', df[df['Min']<df['Min_EMA']][['Date','Min', 'Min_EMA']])
bands = predict_EMA('Max')
bands.reset_index(inplace=True)
max= df['Max'].max()
bands['upper'].fillna(max , inplace=True)
df['Max_EMA']= bands['upper'].copy()
print('Anamolies for EMA_Max values\n', df[df['Max']>df['Max_EMA']][['Date','Max', 'Max_EMA']])
bands = predict_EMA('Range')
bands.reset_index(inplace=True)
max= df['Range'].max()
bands['upper'].fillna(max , inplace=True)
df['Range_EMA']= bands['upper'].copy()
print('Anamolies for EMA_Range values\n', df[df['Range']>df['Range_EMA']][['Date','Range', 'Range_EMA']])
anomaly['Min_anomaly_EMA']= df['Min']
anomaly['Max_anomaly_EMA']= df['Max']
anomaly['Range_anomaly_EMA']= df['Range']

anomaly['Min_anomaly_EMA'][df['Min']<df['Min_EMA']]= 'anomaly'
anomaly['Min_anomaly_EMA'][df['Min']>=df['Min_EMA']]= ''

anomaly['Max_anomaly_EMA'][df['Max']>df['Max_EMA']]= 'anomaly'
anomaly['Max_anomaly_EMA'][df['Max']<=df['Max_EMA']]= ''

anomaly['Range_anomaly_EMA'][df['Range']>df['Range_EMA']]= 'anomaly'
anomaly['Range_anomaly_EMA'][df['Range']<=df['Range_EMA']]= ''
showSummary('9px')
anomaly.info()
anomaly=anomaly[(anomaly.isin(CodesOfInterest)==True).any(1)]

#Apply pd.Series.value_counts to all the columns of the dataframe, it will give you the count of unique values for each row
voting= anomaly.iloc[:,5:14].apply(pd.Series.value_counts, axis=1)
voting.iloc[:,1:2]
anomaly['Vote_Number']=voting.iloc[:,1:2]
anomaly['Vote_Ratio']=voting.iloc[:,1:2]/9*100
anomaly.plot.bar(x='Date', y='Vote_Number')
print(anomaly[['Date','Day', 'Vote_Number']])
print("Total Number of detected anomalies: ",len(anomaly))
threshold= 50
print("Number of Anomalies over the threshold ({}%) voting: {} ".format(threshold,len(anomaly[anomaly['Vote_Ratio']>threshold] )))
print(anomaly[anomaly['Vote_Ratio']>threshold][['Date','Day','Vote_Number','Min','Max','Range']])
anomaly['Vote_Number'].describe()
anomaly.to_csv('anomaly.csv')