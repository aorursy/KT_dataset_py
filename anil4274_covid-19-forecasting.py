# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn import preprocessing
from sklearn import metrics
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

df = train.fillna('NA').groupby(['Country_Region','Province_State','Date'])['ConfirmedCases'].sum() \
                          .groupby(['Country_Region','Province_State']).max().sort_values() \
                          .groupby(['Country_Region']).sum().sort_values(ascending = False)

top10 = pd.DataFrame(df).head(10)
fig = px.bar(top10, x=top10.index, y='ConfirmedCases', labels={'x':'Country'},
             color="ConfirmedCases", color_continuous_scale=px.colors.sequential.Aggrnyl_r)
fig.update_layout(title_text='COVID-19 Confirmed cases by country')
fig.show()
X=pd.DataFrame(train.iloc[:,-1])
Y=pd.DataFrame(train.iloc[:,-2])
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
from sklearn.linear_model import LinearRegression
linearmodel=LinearRegression()
linearmodel.fit(X_train,Y_train)
y_pred_lin=linearmodel.predict(X_test)
df_y_pred=pd.DataFrame(y_pred_lin,columns=['Predict'])
plt.figure(figsize=(5,5))
plt.title('Actual vs Prediction')
plt.xlabel('Fatalities')
plt.ylabel('Predicted')
plt.scatter((X_test['Fatalities']),(Y_test['ConfirmedCases']),c='red')
plt.scatter((X_test['Fatalities']),(df_y_pred['Predict']),c='cyan')
plt.legend()
plt.show()
df_date = train['Date'].apply(lambda dt: datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.datetime.strptime('2020-01-21', '%Y-%m-%d')).apply(lambda x : str(x).split()[0]).astype(int)

train['Days'] = df_date
train.drop(['Date','Id'],axis=1,inplace=True)

train.fillna('0',inplace=True)
df_date = test['Date'].apply(lambda dt: datetime.datetime.strptime(dt, '%Y-%m-%d') - datetime.datetime.strptime('2020-01-21', '%Y-%m-%d')).apply(lambda x : str(x).split()[0]).astype(int)

test['Days'] = df_date
after_use = test.copy()
test.drop(['Date','ForecastId'],axis=1,inplace=True)

test.fillna('0',inplace=True)
ohe = preprocessing.OneHotEncoder()
ohe.fit(train[['Province_State','Country_Region']])
ohe_train = ohe.transform(train[['Province_State','Country_Region']]).toarray()
ohe_t = preprocessing.OneHotEncoder()
ohe_t.fit(test[['Province_State','Country_Region']])
ohe_test = ohe_t.transform(test[['Province_State','Country_Region']]).toarray()
train.drop(['Province_State','Country_Region'],axis=1,inplace=True)

train['Province_State'] = ohe_train[:,0]
train['Country_Region'] = ohe_train[:,1]

train_l_cc = train['ConfirmedCases'].to_numpy()
train_l_fa = train['Fatalities'].to_numpy()

#normed_train_data = preprocessing.normalize(ncc)
train_cc = train[['Days','Province_State','Country_Region']]
train_fa = train[['Days','Province_State','Country_Region','ConfirmedCases']]
test.drop(['Province_State','Country_Region'],axis=1,inplace=True)

test['Province_State'] = ohe_test[:,0]
test['Country_Region'] = ohe_test[:,1]

#normed_train_data = preprocessing.normalize(ncc)
test_cc = test[['Days','Province_State','Country_Region']]
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
def model_func(cc_input_size):
  model = keras.Sequential([
    layers.Dense(3, activation='relu', input_shape=cc_input_size),
    layers.Dense(3, activation='relu'),
    layers.Dense(3, activation='relu'),
    layers.Dense(2, activation='relu'),
    layers.Dense(2, activation='relu'),
    layers.Dense(2, activation='relu'),
    layers.Dense(1)
  ])

  #optimizer = tf.keras.optimizers.RMSprop(0.001)
  optimizer = tf.keras.optimizers.Adam()

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
cc_input_size = [3]
cc_model = model_func(cc_input_size)

EPOCHS = 250

history = cc_model.fit(
  train_cc, train_l_cc,
  epochs=EPOCHS, validation_split = 0.3, verbose=2)

df = pd.DataFrame(history.history)
df['epoch'] = history.epoch
display(df.tail())
df[['mae']].plot()
plt.ylabel('Confirmaed Cases / Infected ')
df[['mse']].plot()
plt.ylabel('Confirmaed Cases / Infected ')
# Train Fatality Model

cc_input_size = [4]
fa_model = model_func(cc_input_size)

EPOCHS = 250

history = fa_model.fit(
  train_fa, train_l_fa,
  epochs=EPOCHS, validation_split = 0.3, verbose=2)

df = pd.DataFrame(history.history)
df['epoch'] = history.epoch
display(df.tail())
df[['mae']].plot()
plt.ylabel('Confirmaed Cases / Infected ')
df[['mse']].plot()
plt.ylabel('Confirmaed Cases / Infected ')
test_data = test_cc

test_predictions_cc = cc_model.predict(test_data)
#Fatality
test_fa = test[['Days','Province_State','Country_Region']]
#,'ConfirmedCases'
test_fa['ConfirmedCases'] = test_predictions_cc

test_data = test_fa

test_p = fa_model.predict(test_fa)
submit = pd.DataFrame()
submit['ForecastId'] = after_use['ForecastId']
submit['ConfirmedCases'] = pd.DataFrame(test_predictions_cc)
submit['Fatalities'] = pd.DataFrame(test_p)
submit.info()
submit.to_csv('submission.csv',index=False)