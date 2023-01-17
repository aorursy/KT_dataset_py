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
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.arima_model import ARIMA



train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
!pip install pmdarima

import pmdarima as pm
train.head()
train.fillna(' ', inplace=True)

test.fillna(' ', inplace=True)

train.columns = train.columns.str.lower()

test.columns = test.columns.str.lower()

train_id = train.pop('id')

#test_id = test.pop('forecastid')

train['c_p'] = train['country_region'] + train['province_state']

test['c_p'] = test['country_region'] + test['province_state']

train.drop(['country_region','province_state'], axis=1, inplace=True)

test.drop(['country_region','province_state'], axis=1, inplace=True)
countries_list = train.c_p.unique()

train_new = []

for i in countries_list:

    train_new.append(train[train['c_p']==i])



plt.subplots(figsize =(12,8))

for i in train_new:

    data = i.confirmedcases.astype('int64').tolist()

    plt.plot(i.date, data)
plt.subplots(figsize =(12,8))

for i in train_new:

    f_data = i.fatalities.astype('int64').tolist()

    plt.plot(i.date, f_data)
#example to find the best sarima parameter for the first country

data = train_new[0].confirmedcases.astype('int64').tolist()

scmodel = pm.auto_arima(data,star_p=1,start_q=1, test='adf', max_p=3, max_q=3, m=12, start_P=0, seasonal=True,

                        D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True) 
scmodel.summary()
#example to find the best sarima parameter for the first country

data = train_new[0].fatalities.astype('int64').tolist()

sfmodel = pm.auto_arima(data,star_p=1,start_q=1, test='adf', max_p=3, max_q=3, m=12, start_P=0, seasonal=True,

                        D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True) 

sfmodel.summary()


submit_confirmed=[]

submit_fatal = []



for i in train_new:

    # confired cases predict

    data = i.confirmedcases.astype('int64').tolist()

    try :

        model_c = SARIMAX(data, order=(1,1,0), seasonal_order=(0,1,1,12), measurement_error=True)

        model_c_fit = model_c.fit(disp=False)

        predicted = model_c_fit.predict(len(data), len(data)+34)

        new = np.concatenate((np.array(data), np.array([int(num) for num in predicted])), axis=0)

        submit_confirmed.extend(list(new[-43:]))

    except:

        submit_confirmed.extend(list(data[-10:-1]))

        for j in range(34):

            submit_confirmed.append(data[-1]*2)

            

    # Fatalities predict

    

    data = i.fatalities.astype('int64').tolist()

    try :

        model_f = SARIMAX(data, order = (1,1,0), seasonal_order=(0,1,0,12), measurement_error=True)

        model_f_fit = model_f.fit(disp=False)

        predicted = model_f_fit.predict(len(data), len(data)+34)

        new = np.concatenate((np.array(data), np.array([int(num) for num in predicted])), axis = 0)

        submit_fatal.extend(list(new[-43:]))

            

    except :

        submit_fatal.extend(list(data[-10:-1]))

        for j in range(34):

            submit_fatal.append(data[-1]*2)
result_submit = pd.concat([pd.Series(np.arange(1,1+len(submit_confirmed))), pd.Series(submit_confirmed), pd.Series(submit_fatal)], axis=1)

result_submit.isnull().sum()
result_submit.rename(columns ={0:'ForecastId',1:'ConfirmedCases',2:'Fatalities'}, inplace=True)
#result_submit
#result_submit.to_csv('submission.csv', index=False)
# Try LSTM method



import pandas as pd

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv', parse_dates =['Date'])

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv',parse_dates=['Date'])

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
train.fillna(' ', inplace=True)

test.fillna(' ', inplace=True)

train.columns = train.columns.str.lower()

test.columns = test.columns.str.lower()

train_id = train.pop('id')

test_id = test.pop('forecastid')

train['c_p'] = train['country_region'] + train['province_state']

test['c_p'] = test['country_region'] + test['province_state']

train.drop(['country_region','province_state'], axis=1, inplace=True)

test.drop(['country_region','province_state'], axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train['c_p_le'] = le.fit_transform(train['c_p'])

test['c_p_le'] = le.transform(test['c_p'])
new = pd.DataFrame()

def create_time_features(data):

    new['hour'] = data['date'].dt.hour

    new['day'] = data['date'].dt.day

    new['dayofweek'] = data['date'].dt.dayofweek

    new['dayofyear'] = data['date'].dt.dayofyear

    new['quarter'] = data['date'].dt.quarter

    new['weekofyear'] = data['date'].dt.weekofyear

    new['month'] = data['date'].dt.month

    new['year'] = data['date'].dt.year

    new_feature = new[['hour','day','dayofweek','dayofyear','quarter','weekofyear','month','year']]

    

    return new_feature



add_train = create_time_features(train) 

add_test = create_time_features(test)

train_tot = pd.concat([train, add_train], axis=1)

test_tot = pd.concat([test, add_test], axis=1)



def create_add_trend(data, a, b):

    for d in data['date'].drop_duplicates():

        for i in data['c_p_le'].drop_duplicates():

            org_mask = (data['date']==d) & (data['c_p_le']==i)

            for l in range(1,8):

                mask_loc = (data['date']==(d-pd.Timedelta(days=l))) & (data['c_p_le']==i)

                            

                try:

                    data.loc[org_mask, 'cf_'+ str(l)] = data.loc[mask_loc,a].values

                    data.loc[org_mask, 'ft_'+ str(l)] = data.loc[mask_loc,b].values

                except:

                    data.loc[org_mask, 'cf_'+ str(l)] = 0.0

                    data.loc[org_mask, 'ft_'+ str(l)] = 0.0



create_add_trend(train_tot,'confirmedcases', 'fatalities')
train_tot
#train_date = train_tot.pop('date')

#test_date = test_new.pop('date')



train_tot.drop('c_p', axis=1, inplace=True)

test_tot.drop('c_p', axis=1, inplace=True)

train_tot.head()

import math

import numpy as np

import keras.backend as K



def RMSLE(predict, true):

    assert predict.shape[0]==true.shape[0]

    return K.sqrt(K.mean(K.square(K.log(predict+1) - K.log(true+1))))

#print ('My RMSLE: ' + str(RMSLE(predict,true)) )





from keras.models import Sequential

from keras.layers import Dense

from sklearn.model_selection import train_test_split

c_features=['c_p_le', 'dayofyear', 'quarter', 'weekofyear', 'month',

            'cf_1', 'cf_2', 'cf_3', 'cf_4', 'cf_5', 'cf_6','cf_7']

f_features=['c_p_le', 'dayofyear', 'quarter', 'weekofyear', 'month',

            'ft_1', 'ft_2', 'ft_3', 'ft_4', 'ft_5', 'ft_6','ft_7']

train_x_c = train_tot[c_features].copy()

train_y_c = train_tot['confirmedcases'].copy()

train_x_f = train_tot[f_features].copy()

train_y_f = train_tot['fatalities'].copy()





train_X, val_X, train_Y, val_Y = train_test_split(train_x_c, train_y_c, test_size=0.1, random_state=0)

train_X_f, val_X_f, train_Y_f, val_Y_f = train_test_split(train_x_f, train_y_f, test_size=0.1, random_state=0)


from sklearn.preprocessing import MinMaxScaler



## for the confirmedcases



x_scale = MinMaxScaler()

y_scale = MinMaxScaler()





train_X_np=train_X.values

val_X_np = val_X.values

train_Y_np = train_Y.values

train_Y_np_reshape = train_Y_np.reshape(-1,1)

val_Y_np = val_Y.values

val_Y_np_reshape = val_Y_np.reshape(-1,1)



#train_X_np_s = x_scale.fit_transform(train_X_np)

#val_X_np_s = x_scale.transform(val_X_np)

#train_Y_np_s = y_scale.fit_transform(train_Y_np)

#val_Y_np_s = y_scale.transform(val_Y_np)



train_X_np_reshape=train_X_np.reshape((train_X_np.shape[0],1,train_X_np.shape[1]))

val_X_np_reshape = val_X_np.reshape((val_X_np.shape[0],1,val_X_np.shape[1]))







## for the fatalities



x_scale_f = MinMaxScaler()

y_scale_f = MinMaxScaler()





train_X_f_np=train_X_f.values

val_X_f_np = val_X_f.values

train_Y_f_np = train_Y_f.values

train_Y_f_np_reshape = train_Y_f_np.reshape(-1,1)

val_Y_f_np = val_Y_f.values

val_Y_f_np_reshape = val_Y_f_np.reshape(-1,1)



#train_X_f_np_s = train_X_f_np #x_scale_f.fit_transform(train_X_f_np)

#val_X_f_np_s = val_X_f_np#x_scale_f.transform(val_X_f_np)

#train_Y_f_np_s = train_Y_f_np#y_scale_f.fit_transform(train_Y_f_np)

#val_Y_f_np_s = val_Y_f_np#y_scale_f.transform(val_Y_f_np)



train_X_f_np_reshape=train_X_f_np.reshape((train_X_f_np.shape[0],1,train_X_f_np.shape[1]))

val_X_f_np_reshape = val_X_f_np.reshape((val_X_f_np.shape[0],1,val_X_f_np.shape[1]))



print(train_X_f_np_reshape.shape, train_Y_f_np_reshape.shape, val_X_f_np_reshape.shape, val_Y_f_np_reshape.shape)

from keras.layers import LSTM

from keras.callbacks import EarlyStopping, ModelCheckpoint







es = EarlyStopping(monitor = 'val_loss', verbose=0, min_delta=0, patience=5, mode='auto')

mc = ModelCheckpoint('model_cf.h5',monitor='val_loss',verbose=1,save_best_only=True)

mc_f = ModelCheckpoint('model_ft.h5',monitor='val_loss',verbose=1,save_best_only=True)

def lstm(hidden_nodes,second_dim,third_dim):

    model = Sequential([LSTM(hidden_nodes, input_shape=(second_dim, third_dim),activation='relu'),

                        Dense(64, activation ='relu'),

                        Dense(32, activation = 'relu'),

                        Dense(1, activation='relu')])



    model.compile(loss=RMSLE, optimizer='adam')

    return model
# 1. learning for confirmedcases 



model_cf = lstm(10, train_X_np_reshape.shape[1], train_X_np_reshape.shape[2])



history_cf = model_cf.fit(train_X_np_reshape, train_Y_np_reshape, epochs=250, batch_size=512, validation_data=(val_X_np_reshape, val_Y_np_reshape), callbacks=[es,mc])

import matplotlib.pyplot as plt



plt.figure(figsize=(8,5))

plt.plot(history_cf.history['loss'])

plt.plot(history_cf.history['val_loss'])

plt.title('CF Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# 2. learning for fatalities



model_ft = lstm(10, train_X_f_np_reshape.shape[1], train_X_f_np_reshape.shape[2])



history_ft=model_ft.fit(train_X_f_np_reshape, train_Y_f_np_reshape, epochs=250, batch_size=512, validation_data=(val_X_f_np_reshape, val_Y_f_np_reshape), callbacks=[es,mc_f])



#_Y_f_np_reshape
plt.figure(figsize=(8,5))

plt.plot(history_ft.history['loss'])

plt.plot(history_ft.history['val_loss'])

plt.title('FT Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# test data also need to be re featuring

from sklearn.preprocessing import MinMaxScaler

test_cf_scaler = MinMaxScaler()

test_ft_scaler = MinMaxScaler()

import pandas as pd

features = ['confirmedcases','fatalities','cf_1', 'ft_1', 'cf_2', 'ft_2', 'cf_3', 'ft_3', 'cf_4', 'ft_4', 'cf_5',

       'ft_5', 'cf_6', 'ft_6', 'cf_7', 'ft_7']

c_feat = ['c_p_le','dayofyear','quarter','weekofyear','month',

          'cf_1', 'cf_2', 'cf_3', 'cf_4', 'cf_5', 'cf_6', 'cf_7']

trend_features =['c_p_le','dayofyear','quarter','weekofyear','month',

                 'cf_1', 'ft_1', 'cf_2', 'ft_2', 'cf_3', 'ft_3', 'cf_4', 'ft_4', 'cf_5',

                 'ft_5', 'cf_6', 'ft_6', 'cf_7', 'ft_7']

f_feat = ['c_p_le','dayofyear','quarter','weekofyear','month',

          'ft_1','ft_2', 'ft_3', 'ft_4', 'ft_5', 'ft_6', 'ft_7']



test_tot.dropna(inplace=True)

test_new =test_tot.copy().join(pd.DataFrame(columns=features))

test_new.head()

test_mask = (test_tot['date']<=train_tot['date'].max())

train_mask = (train_tot['date']>= test_tot['date'].min())

test_new.loc[test_mask,features]=train_tot.loc[train_mask,features].values

future_dt = pd.date_range(start=train_tot['date'].max()+pd.Timedelta(days=1), end=test_tot['date'].max(), freq='1D')

def create_add_trend_predict(data,a,b):

    for d in future_dt:

        for i in data['c_p_le'].drop_duplicates():

            org_mask = (data['date']==d) & (data['c_p_le']==i)

            for l in range(1,8):

                mask_loc = (data['date']==(d-pd.Timedelta(days=l))) & (data['c_p_le']==i)

                            

                try:

                    data.loc[org_mask, 'cf_'+ str(l)] = data.loc[mask_loc,a].values

                    data.loc[org_mask, 'ft_'+ str(l)] = data.loc[mask_loc,b].values

                

                except:

                    data.loc[org_mask, 'cf_'+ str(l)] = 0.0

                    data.loc[org_mask, 'ft_'+ str(l)] = 0.0

                

                    

                #try:

                

                #    data.loc[org_mask, 'ft_'+ str(l)] = data.loc[mask_loc,b].values

                #except:

                

                #    data.loc[org_mask, 'ft_'+ str(l)] = 0.0

                    

            

            test_X = data.loc[org_mask, trend_features]

            

            test_X_cc = test_X[c_feat]

            test_X_cc = test_X_cc.to_numpy().reshape(1,-1)

            test_cc_sc = test_X_cc#x_scale.transform(test_X_cc)

            test_cc = test_cc_sc.reshape(test_cc_sc.shape[0],1,test_cc_sc.shape[1])

            

            test_X_ft = test_X[f_feat]

            test_X_ft = test_X_ft.to_numpy().reshape(1,-1)

            test_ft_sc = test_X_ft#x_scale_f.transform(test_X_ft)

            test_ft = test_ft_sc.reshape(test_ft_sc.shape[0],1,test_ft_sc.shape[1])

            

            next_cc = model_cf.predict(test_cc)

            next_ft = model_ft.predict(test_ft)

            data.loc[org_mask, 'confirmedcases']=next_cc

            data.loc[org_mask, 'fatalities']=next_ft

            

                   

create_add_trend_predict(test_new,'confirmedcases','fatalities')
#from datetime import datetime

#date_str = '04-01-2020'

#d = datetime.strptime(date_str, '%m-%d-%Y').date()

#mask_loc = (test_new['date']==(d-pd.Timedelta(days=1))) & (test_new['c_p_le']==0)

#mask_lll = (test_new['date']==d) & (test_new['c_p_le']==0)



#mask_loc.head(15)

#aa =test_new.loc[mask_loc, 'confirmedcases'].values

#mask_lll.head(15)

#test_new.loc[13,'cf_1']=174



#test_new.loc[mask_lll, 'ft_'+ str(1)]=test_new.loc[mask_loc, 'fatalities'].values



#test_new.loc[mask_lll,'ft_'+ str(1)]

#aa=test_new.loc[13,f_feat].values

#aa=aa.reshape(1,-1)

#aa=val_X_f_np[35]

#aa=x_scale.transform(aa)

#aa=aa.reshape(aa.shape[0],1,aa.shape[1])

#test_new.loc[:,'confirmedcases':'ft_7'].head(50)

#result=model_f.predict(aa)

#result

#val_Y_f_np[2]

#test_new
test_new.loc[:,'confirmedcases':'fatalities'][1:50]


result = pd.DataFrame({'ForecastId':test_id,'ConfirmedCases':test_new['confirmedcases'], 'Fatalities': test_new['fatalities']})

result.shape
result.to_csv('submission.csv', index=False)