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
import pandas as pd

import numpy as np

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv', parse_dates=['Date'])

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv',parse_dates=['Date'])

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')
train.shape

#train.date.drop_duplicates()

#test.date.drop_duplicates()
train.columns = train.columns.str.lower()

test.columns = test.columns.str.lower()
train.fillna(' ',inplace=True)

test.fillna(' ', inplace=True)

train_id = train.pop('id')

test_id = test.pop('forecastid')



train['cp'] = train['country_region'] + train['province_state']

test['cp'] = test['country_region'] + test['province_state']



train.drop(['province_state','country_region'], axis=1, inplace=True)

test.drop(['province_state','country_region'], axis =1, inplace=True)
train.cp.nunique(), test.cp.nunique()
df = pd.DataFrame()

def create_time_feat(data):

    df['date']= data['date']

    df['hour']=df['date'].dt.hour

    df['weekofyear']=df['date'].dt.weekofyear

    df['quarter'] =df['date'].dt.quarter

    df['month'] = df['date'].dt.month

    df['dayofyear']=df['date'].dt.dayofyear

    

    x=df[['hour','weekofyear','quarter','month','dayofyear']]

    

    return x



cr_tr = create_time_feat(train)

cr_te = create_time_feat(test)
train_df = pd.concat([train,cr_tr], axis=1)

test_df = pd.concat([test, cr_te], axis =1)

train_df.shape, test_df.shape, train_df.cp.nunique(), test_df.cp.nunique(), test.shape
test_df.dropna(inplace=True)
from sklearn.preprocessing import LabelEncoder



le=LabelEncoder()

train_df['cp_le']=le.fit_transform(train_df['cp'])

test_df['cp_le']=le.transform(test_df['cp'])



train_df.drop(['cp'], axis=1, inplace=True)

test_df.drop(['cp'], axis=1, inplace=True)
'''

cl_new=[]

for i in train_df['cp_le'].drop_duplicates():

    cl_new.append(train_df[train_df['cp_le']==i])

    

import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))



for i in cl_new:

    df=i.confirmedcases.astype('int64').tolist()



    plt.plot(i.date, df)

'''
'''

#  Use stepwise to find opt-model

import pmdarima as pm



for i in cl_new:

    df=i.confirmedcases.astype('int64').tolist()

    

    scmodel = pm.auto_arima(df, start_p=1,start_q=1, test='adf', max_p=3, max_q=3, m=12, start_P=0, seasonal=True,

                              D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True) 

   # scmodel.summary()



        

#    data = i.fatalities.astype('int64').tolist()

#    sfmodel = pm.auto_arima(data,star_p=1,start_q=1, test='adf', max_p=3, max_q=3, m=12, start_P=0, seasonal=True,

#                              d=None, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True) 

#    sfmodel.summary()

'''
'''

from statsmodels.tsa.statespace.sarimax import SARIMAX





submit_confirmed = []

submit_fatal = []



for i in cl_new:

    #confirmed cases predict

    data = i.confirmedcases.astype('int64').tolist()

    try:

        model_c = SARIMAX(data, order=(1,1,0), seasonal_order=(0,1,1,12),measurement_error=True )

        model_c_fit = model_c.fit(disp=False)

        predicted = model_c_fit.predict(len(data), len(data)+32)

        new = np.concatenate((np.array(data), np.array([int(num) for num in predicted])), axis=0)

        submit_confirmed.extend(list(new[-43:]))



    except:

        submit_confirmed.extend(list(data[-10:-1]))

        for j in range(32):

            submit_confirmed.append(data[-1]*2)

    

    #fatalities predict

    data = i.fatalities.astype('int64').tolist()

    try:

        model_f = SARIMAX(data, order=(1,1,0), seasonal_order=(0,1,0,12),measurement_error=True )

        model_f_fit = model_f.fit(disp=False)

        predicted = model_f_fit.predict(len(data), len(data)+32)

        new = np.concatenate((np.array(data), np.array([int(num) for num in predicted])), axis=0)

        submit_fatal.extend(list(new[-43:]))



    except:

        submit_fatal.extend(list(data[-10:-1]))

        for j in range(32):

            submit_fatal.append(data[-1]*2)

'''
#df_submit = pd.concat([pd.Series(np.arange(1,1+len(submit_confirmed))),pd.Series(submit_confirmed), pd.Series(submit_fatal)], axis=1)

#df_submit.rename(columns={0:'ForecastId', 1:'ConfirmedCases', 2:'Fatalities'}, inplace=True)

#df_submit.to_csv('submission.csv', index=False)
# Using LSTM to see if this brings better performance

train_df.shape
#train_df.head()

test_df.shape
def create_date_feat(data, cf, ft):

    for d in data['date'].drop_duplicates():

        for i in data['cp_le'].drop_duplicates():

            org_mask = (data['date']==d) & (data['cp_le']==i)

            for lag in range(1,15):

                mask_loc = (data['date']==(d-pd.Timedelta(days=lag))) & (data['cp_le']==i)

                

                try:

                    data.loc[org_mask, 'cf_' + str(lag)]=data.loc[mask_loc, cf].values

                    data.loc[org_mask, 'ft_' + str(lag)]=data.loc[mask_loc, ft].values

                

                except:

                    data.loc[org_mask, 'cf_' + str(lag)]=0.0

                    data.loc[org_mask, 'ft_' + str(lag)]=0.0



create_date_feat(train_df,'confirmedcases','fatalities')
train_df.tail(50)
from sklearn.model_selection import train_test_split



cf_feat = ['cp_le', 'weekofyear','quarter','month','dayofyear','cf_1', 'cf_2', 'cf_3', 

          'cf_4', 'cf_5', 'cf_6', 'cf_7', 'cf_8', 'cf_9','cf_10', 'cf_11', 'cf_12', 

          'cf_13', 'cf_14']

ft_feat = ['cp_le', 'weekofyear','quarter','month','dayofyear','ft_1', 'ft_2', 'ft_3', 

          'ft_4', 'ft_5', 'ft_6', 'ft_7', 'ft_8', 'ft_9','ft_10', 'ft_11', 'ft_12', 

          'ft_13', 'ft_14']



train_x_cf = train_df[cf_feat]

print(train_x_cf.shape)

train_x_ft = train_df[ft_feat]

print(train_x_ft.shape)

train_x_cf_reshape = train_x_cf.values.reshape(train_x_cf.shape[0],1,train_x_cf.shape[1])

train_x_ft_reshape = train_x_ft.values.reshape(train_x_ft.shape[0],1,train_x_ft.shape[1])



train_y_cf = train_df['confirmedcases']

train_y_ft = train_df['fatalities']



train_y_cf_reshape = train_y_cf.values.reshape(-1,1)

train_y_ft_reshape = train_y_ft.values.reshape(-1,1)



tr_x_cf, val_x_cf, tr_y_cf, val_y_cf = train_test_split(train_x_cf_reshape, train_y_cf_reshape, test_size=0.2, random_state=0)

tr_x_ft, val_x_ft, tr_y_ft, val_y_ft = train_test_split(train_x_ft_reshape, train_y_ft_reshape, test_size=0.2, random_state=0)

train_x_cf_reshape.shape, train_y_cf_reshape.shape, train_y_ft_reshape.shape
import keras.backend as K



def rmsle(pred,true):

    assert pred.shape[0]==true.shape[0]

    return K.sqrt(K.mean(K.square(K.log(pred+1) - K.log(true+1))))



from keras.models import Sequential

from keras.layers import Dense, LSTM

from keras.callbacks import EarlyStopping, ModelCheckpoint



es = EarlyStopping(monitor='val_loss', min_delta = 0, verbose=0, patience=10, mode='auto')

mc_cf = ModelCheckpoint('model_cf.h5', monitor='val_loss', verbose=0, save_best_only=True)

mc_ft = ModelCheckpoint('model_ft.h5', monitor='val_loss', verbose=0, save_best_only=True)



def lstm_model(hidden_nodes, second_dim, third_dim):

    model = Sequential([LSTM(hidden_nodes, input_shape=(second_dim, third_dim), activation='relu'),

                        Dense(64, activation='relu'),

                        Dense(32, activation='relu'),

                        Dense(1, activation='relu')])

    model.compile(loss=rmsle, optimizer = 'adam')

    

    return model



model_cf = lstm_model(10, tr_x_cf.shape[1], tr_x_cf.shape[2])

model_ft = lstm_model(10, tr_x_ft.shape[1], tr_x_ft.shape[2])



history_cf = model_cf.fit(tr_x_cf, tr_y_cf, epochs=200, batch_size=512, validation_data=(val_x_cf,val_y_cf), callbacks=[es,mc_cf])

history_ft = model_ft.fit(tr_x_ft, tr_y_ft, epochs=200, batch_size=512, validation_data=(val_x_ft,val_y_ft), callbacks=[es,mc_ft])
history_ft.history['loss']
import matplotlib.pyplot as plt



plt.figure(figsize=(8,6))

plt.plot(history_cf.history['loss'], label='Train')

plt.plot(history_cf.history['val_loss'], label='Test')

plt.title("CF Model Loss")

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(loc="upper left")

plt.show()
plt.figure(figsize=(8,6))

plt.plot(history_ft.history['loss'], label='Train')

plt.plot(history_ft.history['val_loss'], label='Test')

plt.title('FT Model loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(loc='upper left')

plt.show()
# formatting Test data & predicting



feat = ['confirmedcases','fatalities','cf_1', 'ft_1', 'cf_2', 'ft_2', 'cf_3', 'ft_3', 

        'cf_4', 'ft_4', 'cf_5', 'ft_5', 'cf_6', 'ft_6', 'cf_7', 'ft_7', 'cf_8', 'ft_8',

        'cf_9', 'ft_9', 'cf_10', 'ft_10', 'cf_11', 'ft_11', 'cf_12', 'ft_12', 'cf_13', 'ft_13',

        'cf_14', 'ft_14']

c_feat = ['cp_le', 'weekofyear','quarter','month','dayofyear','cf_1', 'cf_2', 'cf_3', 

          'cf_4', 'cf_5', 'cf_6', 'cf_7', 'cf_8', 'cf_9','cf_10', 'cf_11', 'cf_12', 

          'cf_13', 'cf_14']

f_feat =  ['cp_le', 'weekofyear','quarter','month','dayofyear','ft_1', 'ft_2', 'ft_3', 

          'ft_4', 'ft_5', 'ft_6', 'ft_7', 'ft_8', 'ft_9','ft_10', 'ft_11', 'ft_12', 

          'ft_13', 'ft_14']

tot_feat = ['cp_le', 'weekofyear','quarter','month','dayofyear','cf_1', 'ft_1', 'cf_2', 'ft_2', 'cf_3', 'ft_3', 

        'cf_4', 'ft_4', 'cf_5', 'ft_5', 'cf_6', 'ft_6', 'cf_7', 'ft_7', 'cf_8', 'ft_8',

        'cf_9', 'ft_9', 'cf_10', 'ft_10', 'cf_11', 'ft_11', 'cf_12', 'ft_12', 'cf_13', 'ft_13',

        'cf_14', 'ft_14']



test_new = test_df.copy().join(pd.DataFrame(columns=feat))

test_mask = (test_df['date'] <= train_df['date'].max())

train_mask = (train_df['date'] >= test_df['date'].min())

test_new.loc[test_mask,feat] = train_df.loc[train_mask, feat].values

future_df = pd.date_range(start = train_df['date'].max()+pd.Timedelta(days=1),end=test_df['date'].max(), freq='1D')



def create_add_trend_pred(data, cf, ft):

    for d in future_df:

        for i in data['cp_le'].drop_duplicates():

            org_mask = (data['date']==d) & (data['cp_le']==i)

            for lag in range(1,15):

                mask_loc = (data['date']==(d-pd.Timedelta(days=lag))) & (data['cp_le']==i)

                

                try:

                    data.loc[org_mask, 'cf_' + str(lag)]=data.loc[mask_loc,cf].values

                    data.loc[org_mask, 'ft_' + str(lag)]=data.loc[mask_loc,ft].values

                    

                except:

                    data.loc[org_mask, 'cf_' + str(lag)]=0.0

                    data.loc[org_mask, 'ft_' + str(lag)]=0.0

            

            test_x = data.loc[org_mask,tot_feat]

            

            test_x_cf = test_x[c_feat]

            test_x_cf = test_x_cf.to_numpy().reshape(1,-1)

            test_x_cf_reshape = test_x_cf.reshape(test_x_cf.shape[0],1,test_x_cf.shape[1])

            

            test_x_ft = test_x[f_feat]

            test_x_ft = test_x_ft.to_numpy().reshape(1,-1)

            test_x_ft_reshape = test_x_ft.reshape(test_x_ft.shape[0],1,test_x_ft.shape[1])

            data.loc[org_mask, cf] = model_cf.predict(test_x_cf_reshape)

            data.loc[org_mask, ft] = model_ft.predict(test_x_ft_reshape)



create_add_trend_pred(test_new, 'confirmedcases', 'fatalities')
test_new.head(60)
sub_pred = pd.DataFrame({'ForecastId': test_id, 'ConfirmedCases':test_new['confirmedcases'],'Fatalities':test_new['fatalities']})

sub_pred.to_csv('submission.csv', index=False)
sub_pred.head(60)