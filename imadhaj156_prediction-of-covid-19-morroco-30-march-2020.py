# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt
df=pd.read_excel('/kaggle/input/data30mars/data (1).xlsx')
date_parser=lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M')

df.head()
df['Cas testés positifs'].plot(style=['r-'], figsize=(30, 10))

idata=df[['Jour','Cas testés positifs','Rétablis','Morts']]
idata.rename(columns={'Jour':'Date','Cas testés positifs':'Confirmed','Rétablis':'Recovered','Morts': 'Deaths'}, inplace=True)
idata['Confirmed'] = pd.to_numeric(idata['Confirmed'], errors='coerce')
idata['Recovered'] = pd.to_numeric(idata['Recovered'] , errors='coerce')
idata['Deaths'] = pd.to_numeric(idata['Deaths'], errors='coerce')
plt.figure(figsize=(20,10))
plt.bar(idata.Date, idata.Confirmed,label="Confirm")
plt.bar(idata.Date, idata.Deaths,label="Death")
plt.bar(idata.Date, idata.Recovered,label="Recovery")

plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Confirmation vs Recoverey vs Death",fontsize=50)
plt.show()

idata.set_index("Date", inplace = True)
idata.plot()
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
pr_data_test = df.loc[:,['Jour','Cas testés positifs']]
pr_data_test.columns = ['ds','y']
pr_data_test.head()
m = Prophet()
m.fit(pr_data_test)
future=m.make_future_dataframe(periods=15)
forecast_test=m.predict(future)
forecast_test
test = forecast_test.loc[:,['ds','trend']]
test = test[test['trend']>0]
test.head()
test=test.tail(15)
test.columns = ['Date','Cas positif']
test.head()
import plotly.offline as py

fig_test = plot_plotly(m, forecast_test)
py.iplot(fig_test) 

fig_test = m.plot(forecast_test,xlabel='Date',ylabel='cas positif')
figure=m.plot_components(forecast_test)

pr_data_cm = df.loc[:,['Jour','Morts']]
pr_data_cm.columns = ['ds','y']
pr_data_cm.head()
m=Prophet()
m.fit(pr_data_cm)
future=m.make_future_dataframe(periods=15)
forecast_cm=m.predict(future)
forecast_cm
cnfrm = forecast_cm.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm=cnfrm.tail(15)
cnfrm.columns = ['Date','death']
cnfrm.head()
fig_cm = plot_plotly(m, forecast_cm)
py.iplot(fig_cm) 

fig_cm = m.plot(forecast_cm,xlabel='Date',ylabel='death')
figure=m.plot_components(forecast_cm)

pr_data_r = df.loc[:,['Jour','Rétablis']]
pr_data_r.columns = ['ds','y']
pr_data_r.head()
m=Prophet()
m.fit(pr_data_r)
future=m.make_future_dataframe(periods=15)
forecast_r=m.predict(future)
forecast_r
rec = forecast_r.loc[:,['ds','trend']]
rec = rec[rec['trend']>0]
rec=rec.tail(15)
rec.columns = ['Date','Recovery']
rec.head()
fig_r = plot_plotly(m, forecast_r)
py.iplot(fig_r) 

fig_r = m.plot(forecast_r,xlabel='Date',ylabel='Recovery Count')
figure=m.plot_components(forecast_r)

df1=df[['Cas testés positifs']]
x = np.arange(len(df1)).reshape(-1, 1)
y = df1.values
pr_data = df.loc[:,['Jour','Cumul Patients']]
pr_data.columns = ['ds','y']
pr_data.head()

m=Prophet()
m.fit(pr_data)
future=m.make_future_dataframe(periods=365)
forecast=m.predict(future)
forecast
fig = plot_plotly(m, forecast)
py.iplot(fig) 

fig = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count')
figure=m.plot_components(forecast)

arima_data=pr_data

arima_data.columns = ['confirmed_date','count']
arima_data=arima_data[:-1]
arima_data
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(arima_data['count'].values, order=(0, 2,4))
fit_model = model.fit(trend='c', full_output=True, disp=True)
fit_model.summary()
fit_model.plot_predict()
plt.title('Forecast vs Actual')
pd.DataFrame(fit_model.resid).plot()
forcast = fit_model.forecast(steps=7)
pred_y = forcast[0].tolist()
pd.DataFrame(pred_y)
dataset = arima_data
dataset.set_index("confirmed_date", inplace = True)
dataset.head()
data = np.array(dataset).reshape(-1, 1)
train_data = dataset[5:len(dataset)-5]
test_data = dataset[len(dataset)-5:]
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
#from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)
n_input =5
n_features =1
                             
generator = TimeseriesGenerator(scaled_train_data,scaled_train_data, length=n_input, batch_size=1)

lstm_model = Sequential()
lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (n_input, n_features)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50, return_sequences = True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units = 1))
lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
lstm_model.fit_generator( generator,epochs = 30)
lstm_model.history.history.keys()


losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize = (30,4))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(np.arange(0,100,1))
plt.plot(range(len(losses_lstm)), losses_lstm)
lstm_predictions_scaled = []

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)

prediction = pd.DataFrame(scaler.inverse_transform(lstm_predictions_scaled))
prediction
times = np.arange(1,31)
df=arima_data[['count']]
var=df.to_numpy()
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
clf = SVR(degree=5,C=1000)
vals=clf.fit(times.reshape(-1,1),var).predict(times.reshape(-1,1))
plt.plot(times.reshape(-1,1),vals)
plt.scatter(times.reshape(-1,1),var)
plt.show()
value = clf.predict(np.arange(1,37).reshape(-1,1))
plt.plot(np.arange(1,37),value)
plt.scatter(np.arange(1,37),value)
plt.show()
#print(f'Prediction of deaths using SVR on {int(day)} is {int(value)}')
idata.describe()


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
df=pd.read_excel('/kaggle/input/data30mars/data (1).xlsx')
date_parser=lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M')
x=df.loc[:,['Cas testés']]
y=df.loc[:,['Cas testés positifs']]
x=np.array(x)
y=np.array(y)
reg = linear_model.LinearRegression(normalize='Ture')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)


reg.fit(X_train, y_train)

print(reg.score(X_train, y_train))
reg.coef_

reg.intercept_

import seaborn as sns

sns.regplot(x=X_train,y=y_train)

