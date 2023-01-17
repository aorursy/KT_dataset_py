import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,date,time
import seaborn as sns
import numpy as np
import plotly.express as px

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima_model import ARIMA
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import register_matplotlib_converters  

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

import pandas as pd
Case = pd.read_csv("../input/coronavirusdataset/Case.csv")
PatientInfo = pd.read_csv("../input/coronavirusdataset/PatientInfo.csv")
Time = pd.read_csv("../input/coronavirusdataset/Time.csv", parse_dates=['date'])
TimeAge = pd.read_csv("../input/coronavirusdataset/TimeAge.csv")
TimeProvince = pd.read_csv("../input/coronavirusdataset/TimeProvince.csv")

PatientInfo['confirmed_date'] = PatientInfo['confirmed_date'].apply(pd.to_datetime)
PatientInfo['released_date'] = PatientInfo['released_date'].apply(pd.to_datetime)
PatientInfo['deceased_date'] = PatientInfo['deceased_date'].apply(pd.to_datetime)
released_people = PatientInfo[PatientInfo['released_date'].isnull() == 0]
released_people['recover_period'] = released_people['released_date'] - released_people['confirmed_date']

PatientInfo.loc[PatientInfo['age'] == '10s', 'ageGroup'] = '10~30'
PatientInfo.loc[PatientInfo['age'] == '20s', 'ageGroup'] = '10~30'
PatientInfo.loc[PatientInfo['age'] == '30s', 'ageGroup'] = '10~30'
PatientInfo.loc[PatientInfo['age'] == '40s', 'ageGroup'] = '40~60'
PatientInfo.loc[PatientInfo['age'] == '50s', 'ageGroup'] = '40~60'
PatientInfo.loc[PatientInfo['age'] == '60s', 'ageGroup'] = '40~60'
PatientInfo.loc[PatientInfo['age'] == '70s', 'ageGroup'] = '70~'
PatientInfo.loc[PatientInfo['age'] == '80s', 'ageGroup'] = '70~'
PatientInfo.loc[PatientInfo['age'] == '90s', 'ageGroup'] = '70~'



Time.tail(10)
df_cases = Time.copy().drop(['date', 'time'], axis=1)
df_t = df_cases.tail(1)
df_t["Mortality Rate (%)"] = np.round(100*df_t["deceased"]/df_t["confirmed"],2)
df_t['test'] = df_t['test'].apply(lambda x:format(x,","))
df_t['negative'] = df_t['negative'].apply(lambda x:format(x,","))
df_t['released'] = df_t['released'].apply(lambda x:format(x,","))
df_t['confirmed'] = df_t['confirmed'].apply(lambda x:format(x,","))
df_t.reset_index(0)
df_t.style.background_gradient(cmap='summer',axis=1).set_properties(**{'font-size': '12pt'})

recentDate = TimeProvince['date'].max()
df_t = TimeProvince[TimeProvince['date']== recentDate]
df_t = df_t.copy().drop(['date', 'time'], axis=1)
df_t["Mortality Rate (%)"] = np.round(100*df_t["deceased"]/df_t["confirmed"],2)
#df_t['released'] = df_t['released'].apply(lambda x:format(x,","))
#df_t['confirmed'] = df_t['confirmed'].apply(lambda x:format(x,","))
df_t.reset_index(0)
df_t.style.background_gradient(cmap='summer',axis=1)\
      .background_gradient(cmap='YlOrBr',subset=["confirmed"])\
     .background_gradient(cmap='Reds',subset=["deceased"])\
     .background_gradient(cmap='Purples',subset=["Mortality Rate (%)"]).set_properties(**{'font-size': '10pt'})
recentDate = TimeProvince['date'].max()
df_t = TimeProvince[TimeProvince['date']== recentDate]
f = plt.figure(figsize=(10,5))
f.add_subplot(111)
df_t.set_index("province", inplace=True)
plt.axes(axisbelow=True)
plt.barh(df_t.sort_values('confirmed')["confirmed"].index[-10:],df_t.sort_values('confirmed')["confirmed"].values[-10:],color="skyblue")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Confirmed Cases",fontsize=18)
plt.title("Top 10 province (Confirmed Cases)",fontsize=20)
plt.grid(alpha=0.3)
recentDate = TimeProvince['date'].max()
df_t = TimeProvince[TimeProvince['date']== recentDate]
f = plt.figure(figsize=(10,5))
f.add_subplot(111)
df_t.set_index("province", inplace=True)
plt.axes(axisbelow=True)
plt.barh(df_t.sort_values('deceased')["deceased"].index[-10:],df_t.sort_values('deceased')["deceased"].values[-10:],color="red")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("deceased Cases",fontsize=18)
plt.title("Top 10 province (deceased Cases)",fontsize=20)
plt.grid(alpha=0.3)
recentDate = TimeProvince['date'].max()
df_t = TimeProvince[TimeProvince['date']== recentDate]
f = plt.figure(figsize=(10,5))
f.add_subplot(111)
df_t['active'] = df_t['confirmed'] - df_t['released'] - df_t['deceased']
df_t.set_index("province", inplace=True)
plt.axes(axisbelow=True)
plt.barh(df_t.sort_values('active')["active"].index[-10:],df_t.sort_values('active')["active"].values[-10:],color="pink")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("active Cases",fontsize=18)
plt.title("Top 10 province (active Cases)",fontsize=20)
plt.grid(alpha=0.3)
recentDate = TimeProvince['date'].max()
df_t = TimeProvince[TimeProvince['date']== recentDate]
f = plt.figure(figsize=(10,5))
f.add_subplot(111)
df_t.set_index("province", inplace=True)
plt.axes(axisbelow=True)
plt.barh(df_t.sort_values('released')["released"].index[-10:],df_t.sort_values('released')["released"].values[-10:],color="yellow")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("released Cases",fontsize=18)
plt.title("Top 10 province (released Cases)",fontsize=20)
plt.grid(alpha=0.5)
recentDate = TimeProvince['date'].max()
df_t = TimeProvince[TimeProvince['date']== recentDate]
df_t = df_t.copy().drop(['date', 'time'], axis=1)
df_t["Mortality Rate (%)"] = np.round(100*df_t["deceased"]/df_t["confirmed"],2)
df_t.iloc[:,:-1].corr().style.background_gradient(cmap='Reds')
df = PatientInfo.groupby([PatientInfo['confirmed_date'].dt.date, 'ageGroup']).count()['patient_id']
df_temp = pd.DataFrame(data=df[:])    
df_temp = df_temp.reset_index()
df_temp.rename(columns = {'patient_id' : 'count'}, inplace = True)
plt.figure(figsize=(20,8))
sns.barplot(x='confirmed_date', y='count', hue='ageGroup',  data=df_temp) # default : dodge=True
plt.title('Confirmed Case / AgeGroup', fontsize=20)
plt.xticks( rotation=90)
plt.legend(fontsize=12)
plt.show()
temp = Case['infection_case'].str.split()
for i in range(len(temp)):
    Case['infection_case'][i] = temp[i][-1]

cg = Case.groupby('infection_case').agg(sum)['confirmed']
data_pie = []
data_pie_index = []
for a in range(len(cg)):
    if cg[a] > 20:
        data_pie.append(cg[a])
        data_pie_index.append(cg.index[a])
number_case = len(data_pie)
explode=np.zeros(number_case)

fig = plt.gcf() 
fig.set_size_inches(10,10)
plt.pie(data_pie,autopct='%1.1f%%',shadow=True)
title = "Top "+str(number_case) +" infection Case" 
plt.title(title,fontsize=20, fontweight="bold")
plt.legend(data_pie_index, loc="best") # bbx required to place legend without overlapping
plt.show()
released_people.head(10)

temp2 = released_people.groupby('confirmed_date')['recover_period'].sum().reset_index()
temp3 = released_people.groupby('confirmed_date')['recover_period'].count().reset_index()
aa = pd.merge(temp2, temp3, on='confirmed_date')

aa.set_index('confirmed_date', inplace=True)
aa['recover_mean'] = aa['recover_period_x'] // aa['recover_period_y']
aa['recover_mean'] = aa['recover_mean'].astype('timedelta64[D]').astype(int)
marker_style = dict(c="crimson",linewidth=5, linestyle='-', marker='o',markersize=8, markerfacecolor='#ffffff')
#aa['recover_mean'].plot(figsize=(8,5), color='blue', zorder=1)
aa['recover_mean'].plot(figsize=(15,5))
aa.rolling(window=5).mean()['recover_mean'].plot(**marker_style)
plt.title('Recovery Time', size=30)
plt.xlabel('Days Since 1/22/2020', fontsize=18)
plt.ylabel('means of Recovery Time', fontsize=18)
plt.grid(alpha = 0.5)

temp2 = released_people.groupby('province')['recover_period'].sum().reset_index()
temp3 = released_people.groupby('province')['recover_period'].count().reset_index()
aa = pd.merge(temp2, temp3, on='province')

aa['recover_mean'] = aa['recover_period_x'].dt.days / aa['recover_period_y']
aa.sort_values(by='recover_mean', ascending=True, inplace=True)
fig, ax = plt.subplots(1,1,figsize=(10,6))
plt.barh(aa['province'], aa['recover_mean'] ,height=0.6,alpha=0.5)
plt.title('# of Covid-19 Confirmed Cases in Countries/Regions', size=12)
plt.xlabel('Recovery Days', size=12)
ax.set_xlim([5,22])
fig.autofmt_xdate()

released_people.head()
temp2 = released_people.groupby('age')['recover_period'].sum().reset_index()
temp3 = released_people.groupby('age')['recover_period'].count().reset_index()
aa = pd.merge(temp2, temp3, on='age')
aa['recover_mean'] = aa['recover_period_x'].dt.days / aa['recover_period_y']

fig, ax = plt.subplots(1,1,figsize=(10,6))
aa.sort_values(by='recover_mean', ascending=True, inplace=True)
plt.barh(aa['age'], aa['recover_mean'] ,height=0.6,alpha=0.5)
plt.title('# of Covid-19 Confirmed Cases in Countries/Regions', size=12)
plt.xlabel('Recovery Days', fontsize=12,fontweight="bold")
ax.set_xlim([10,28])
fig.autofmt_xdate()
data = pd.DataFrame({'ds':Time['date'], 'y': Time['confirmed']})
m=Prophet()
m.fit(data)
f=m.make_future_dataframe(periods=40)
prop_forecast=m.predict(f)
forecast = prop_forecast[['ds','yhat']].tail()
fig = plot_plotly(m, prop_forecast)
fig = m.plot(prop_forecast,xlabel='Date',ylabel='Confirmed Cases')
import datetime

a = ARIMA(data['y'], order=(5, 1, 0))
a = a.fit(trend='c', full_output=True, disp=True)
forecast = a.forecast(steps= 15)
pred = list(forecast[0])

start_date = data['ds'].max()
prediction_dates = []
for i in range(15):
    date = start_date + datetime.timedelta(days=1)
    prediction_dates.append(date)
    start_date = date
plt.figure(figsize= (15,5))
plt.xlabel("Dates",fontsize = 20)
plt.ylabel('Confirmed cases',fontsize = 20)

plt.plot_date(y= pred,x= prediction_dates,label = 'Predicted');
plt.plot_date(y=data['y'],x=data['ds'],linestyle = '-',label = 'Actual');
plt.legend();
plt.xticks(rotation=90)
register_matplotlib_converters()     
df = pd.DataFrame(columns = ['date','confirmed'])
df['date'] = Time['date']
df['confirmed'] = Time['confirmed']
df.set_index('date', inplace=True)
   
def plot_decompose(decomposeresult):
    fig,(ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(15,8))   
    decomposeresult.observed.plot(legend=False, ax=ax1)
    ax1.set_ylabel('Observed')
    decomposeresult.trend.plot(legend=False, ax=ax2)
    ax2.set_ylabel('Trend')
    decomposeresult.seasonal.plot(legend=False, ax=ax3)
    ax3.set_ylabel('Seasonal')
    decomposeresult.resid.plot(legend=False, ax=ax4)
    ax4.set_ylabel('Resid')
    
result = seasonal_decompose(df, model='additive', freq=1)

plot_decompose(result)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import datetime
#import keras.callbacks

data_temp = pd.DataFrame(columns = ['date','confirmed'])
data_temp['date'] = Time['date']
data_temp['confirmed'] = Time['confirmed']
data_temp.head(20)

seq_len = 5
sequence_length = seq_len + 1

result = []
for index in range(len(data_temp) - sequence_length):
    result.append(data_temp['confirmed'].values[index: index + sequence_length])

def normalize_windows(data):
    normalized_data = []
    for window in data:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    return np.array(normalized_data)

result = normalize_windows(result)  
row = int(round(result.shape[0] * 0.6))
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

x_train.shape, x_test.shape
model = Sequential()
print(x_train.shape)
model.add(LSTM(50, return_sequences=True, input_shape=(5, 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')
model.summary()
start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=20,
    epochs=80
    
 )
pred = model.predict(x_test)

fig = plt.figure(facecolor='white', figsize=(15, 5))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction', linewidth=5, linestyle=':')
plt.ylim([0,0.5])   

ax.legend()
plt.show()
