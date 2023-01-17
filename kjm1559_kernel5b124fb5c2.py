# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



datas = []

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        datas.append(pd.read_csv(os.path.join(dirname, filename)))



datas[1] = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

datas[2] = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

# Any results you write to the current directory are saved as output.
tmp_country_label = []

for idata in datas[1].itertuples():

    try:

        tmp_country_label.append(idata.Country_Region + '_' + idata.Province_State)

    except:

        tmp_country_label.append(idata.Country_Region)

datas[1]['country_label'] = tmp_country_label

# datas[1].loc[0].Country_Region + datas[1].loc[0].Province_State

np.isnan(datas[1].loc[0].Province_State)

country_list = datas[1].country_label.unique().tolist()
import matplotlib.pylab as plt

datas[1]['Mortality_rate'] = datas[1]['Fatalities'].values / (datas[1]['ConfirmedCases'].values + 1e-8)

datas[1][datas[1]['Mortality_rate'] > 0.5]

plt.plot(datas[1][datas[1]['Country_Region'] == 'Guyana']['Mortality_rate'].values)

plt.show()
from datetime import datetime

from datetime import timedelta

import tensorflow.keras

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Input, GRU, Masking, Permute, Concatenate, LSTM, BatchNormalization, Flatten

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.regularizers import l2

from tensorflow.keras import backend as K

from tqdm import tqdm

import gzip

import pickle



def rmsle(pred,true):

    assert pred.shape[0]==true.shape[0]

    return K.sqrt(K.mean(K.square(K.log(pred+1) - K.log(true+1))))



def attention_mechanism(days, input_):

    x = Dense(days, activation='softmax')(input_)

    return x



def attention_model(input_size, days=21, batch_size=32, epochs=200, lr=1e-3):



    country_input = Input(shape=(306,), name='country_onehot')

    inputs = Input(shape=(None, input_size), name='encoder_input')    

    target_number = Input(shape=(1,), name='target_input')

    flag_input = Input(shape=(1,), name='flag_input')



    x = Masking(mask_value=0, input_shape=(None, input_size))(inputs)

    x = GRU(256, name='GRU_layer1', return_sequences=True)(inputs)



    attention_x = attention_mechanism(days, x)

    gru_out = Permute((2, 1))(x)

    attention_mul = K.batch_dot(gru_out, attention_x)

    attention_mul = Permute((2, 1))(attention_mul)       



    x = GRU(256, name='GRU_layer2', return_sequences=True)(attention_mul)

    x = Flatten()(x)

    # country onehot concatenate

    x = Concatenate(axis=-1)([country_input, x])

    x = Dense(128, activation='relu')(x)

    x = Dense(64, activation='relu')(x)

    x = Dense(32, activation='relu')(x)        

    outputs = Dense(1, activation='sigmoid')(x)

    print(outputs.shape, flag_input.shape, target_number.shape)

    

    outputs = target_number * (flag_input + outputs)



    optimizer = Adam(lr=5e-5, clipnorm=1.0, name='adam')

    model = Model([inputs, country_input, target_number, flag_input], outputs, name='gru_network')

    model.compile(optimizer=optimizer, loss=rmsle)

#     print(self.model.summary())

    return model
class corona19_predict:

    def __init__(self, df, days=21, batch_size=32, epochs=200):

        self.days = days

        self.batch_size = batch_size

        self.epochs = epochs

        self.confirmed_cases_model = attention_model(3, days, lr=1e-4)

        self.fatalities_model = attention_model(1, days, lr=1e-4)

        self.cal_increase_rate(df)

    

    def cal_increase_rate(self, df):

        # calculate increase rate & set target dataframe

        pre_ccd = 0

        pre_fd = 0

        confirmed_cases_diff = []

        fatalities_diff = []

        for idata in df.itertuples():

            if idata.ConfirmedCases < pre_ccd:

                pre_ccd = 0

                pre_fd = 0

            confirmed_cases_diff.append(idata.ConfirmedCases - pre_ccd)

            fatalities_diff.append(idata.Fatalities - pre_fd)

            pre_ccd = idata.ConfirmedCases

            pre_fd = idata.Fatalities



        df['ConfirmedCases_diff'] = confirmed_cases_diff

        df['Fatalities_diff'] = fatalities_diff 

        

        df['Fatalities_diff'] = df['Fatalities_diff'].clip(0) # dead man never live

        

        df['ConfirmedCases_diff_percent'] = df['ConfirmedCases_diff'].values / (df['ConfirmedCases'].values + 1.0e-10)

        df['Fatalities_diff_percent'] = df['Fatalities_diff'].values / (df['Fatalities'].values + 1.0e-10)        

        

        tmp_country_label = []

        for idata in df.itertuples():

            try:

                tmp_country_label.append(idata.Country_Region + '_' + idata.Province_State)

            except:

                tmp_country_label.append(idata.Country_Region)

        df['country_label'] = tmp_country_label

        

        self.target_df = df

        self.country_list = df['country_label'].unique().tolist()

        return df

    

    def get_country_onehot(self, country_str):

        # country onehot encoding

        country_onehot = np.zeros(len(self.country_list))

        country_onehot[self.country_list.index(country_str)] = 1

        return country_onehot  

    

    def encoded_data(self, country_onehot, target_date_str):        

        # get target date list

        date_list = target_date_str.split('-')

        delta = timedelta(days=1)

        date = datetime(int(date_list[0]), int(date_list[1]), int(date_list[2])) - delta        

        day_list = []

        for i in range(self.days):

            day_list.append(date.strftime('%Y-%m-%d'))

            date -= delta

        day_list = day_list[::-1]

        

        # get data

        confirmed_cases = 0

        fatalities = 0

        encoded_data = []  

        if self.country_df.country_label.values[0] != self.country_list[np.argmax(country_onehot)]:

            self.country_df = self.target_df[self.target_df.country_label == self.country_list[np.argmax(country_onehot)]]

        for date_str in day_list:

            tmp_data_df = self.country_df[self.country_df.Date == date_str]

            if len(tmp_data_df) == 0:

                'train data not exist'                

            else:                

                confirmed_cases_diff = tmp_data_df.ConfirmedCases_diff.values[0]

                fatalities_diff = tmp_data_df.Fatalities_diff.values[0]

                confirmed_cases = tmp_data_df.ConfirmedCases.values[0]

                fatalities = tmp_data_df.Fatalities.values[0]

                confirmed_cases_diff_percent = tmp_data_df.ConfirmedCases_diff_percent.values[0]

                fatalities_diff_percent = tmp_data_df.Fatalities_diff_percent.values[0]

            encoded_data.append([confirmed_cases, fatalities, confirmed_cases_diff, fatalities_diff, confirmed_cases_diff_percent, fatalities_diff_percent])

        return encoded_data

    

    def make_train_data(self):

        train_data = {'country_onehot': [], 'encoder_input': []}

        train_label = []

        p_country = ''

        for idata in tqdm(self.target_df.itertuples(), total=len(self.target_df), position=0):

            if p_country != idata.country_label:

                self.country_df = self.target_df[self.target_df.country_label == idata.country_label]

                p_country = idata.country_label

            if idata.Date > self.target_df.iloc[self.days + 1].Date:

                tmp_onehot_data = self.get_country_onehot(idata.country_label)                        

                tmp_encoded_data = self.encoded_data(tmp_onehot_data, idata.Date)

                if np.sum(np.array(tmp_encoded_data)[:, :]) != 0:

                    train_data['country_onehot'].append(tmp_onehot_data)

                    train_data['encoder_input'].append(self.encoded_data(tmp_onehot_data, idata.Date)) 

                    train_label.append([idata.ConfirmedCases, idata.Fatalities, idata.ConfirmedCases_diff, idata.Fatalities_diff, idata.ConfirmedCases_diff_percent, idata.Fatalities_diff_percent])

        

        return [np.array(train_data['encoder_input']), np.array(train_data['country_onehot'])], np.array(train_label)

    

    def train_data_fatalities(self):  

        try:

            with gzip.open('encoded_data.dat', 'rb') as f:

                X_train, y_train = pickle.load(f)

        except:        

            X_train, y_train = self.make_train_data()

            with gzip.open('encoded_data.dat', 'wb') as f:

                pickle.dump([X_train, y_train], f)

        x_train = X_train.copy()

        x_train[0] = (x_train[0][:, :, 1] / (x_train[0][:, :, 0] + 1e-8)).reshape((x_train[0].shape[0], x_train[0].shape[1], 1))        

        death_rate_index = np.where(y_train[:, 1] / (y_train[:, 0] + 1e-8) < 0.3)[0]

        

        tb_hist = tensorflow.keras.callbacks.TensorBoard(log_dir='./graph_gru_attention', histogram_freq=1, write_graph=True, write_images=True)

        model_path = './fatalities_gru_attention.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'

        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)

        early_stopping = EarlyStopping(patience=10)



        history = self.fatalities_model.fit({'encoder_input':x_train[0][death_rate_index], 'country_onehot':x_train[1][death_rate_index], 'target_input': X_train[0][death_rate_index][:, -1, 0].reshape(list(X_train[0][death_rate_index].shape[:-2]) + [1]), 'flag_input': np.zeros(len(x_train[0][death_rate_index]))}, y_train[:, 1][death_rate_index], batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True,

                              validation_split=0.2,

                               callbacks=[tb_hist, cb_checkpoint, early_stopping])  # , class_weight=class_weights) 

        y_predict = self.fatalities_model.predict({'encoder_input':x_train[0], 'country_onehot':x_train[1], 'target_input': X_train[0][:, -1, 0].reshape(list(X_train[0].shape[:-2]) + [1]), 'flag_input': np.zeros(len(x_train[0]))})

        return y_predict, y_train

    

    def train_data_confirmed_cases(self):  

        try:

            with gzip.open('encoded_data.dat', 'rb') as f:

                X_train, y_train = pickle.load(f)

        except:        

            X_train, y_train = self.make_train_data()

            with gzip.open('encoded_data.dat', 'wb') as f:

                pickle.dump([X_train, y_train], f)

                

        x_train = X_train.copy()

        x_train[0] = np.concatenate([(np.clip(x_train[0][:, :, 1] / 1e6, 0, 1)).reshape(list(x_train[0].shape[:-1]) + [1]), x_train[0][:, :, [4, 5]]], axis=2)

        

        tb_hist = tensorflow.keras.callbacks.TensorBoard(log_dir='./graph_gru_attention', histogram_freq=1, write_graph=True, write_images=True)

        model_path = './confirmed_cases_gru_attention.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'

        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)

        early_stopping = EarlyStopping(patience=10)

        

        history = self.confirmed_cases_model.fit({'encoder_input':x_train[0], 'country_onehot':X_train[1], 'target_input': X_train[0][:,-1,0].reshape(list(X_train[0].shape[:-2]) + [1]), 'flag_input':np.ones((len(x_train[0]), 1))}, y_train[:, 0], batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True,

                              validation_split=0.2,

                               callbacks=[tb_hist, cb_checkpoint, early_stopping])  # , class_weight=class_weights) 

        y_predict = self.confirmed_cases_model.predict({'encoder_input':x_train[0], 'country_onehot':X_train[1], 'target_input': X_train[0][:,-1,0].reshape(list(X_train[0].shape[:-2]) + [1]), 'flag_input':np.ones((len(x_train[0]), 1))})

        return y_predict, y_train

    

    def load_models(self, country_list):

        self.confirmed_cases_model.load_weights('/kaggle/working/confirmed_cases_gru_attention.h5')

        self.fatalities_model.load_weights('/kaggle/working/fatalities_gru_attention.h5')

        self.country_list = country_list

        

    def predict_encoder_confirmed_cases(self, day_list, country_label):

        encoded_data = []

        country_onehot = np.zeros(len(self.country_list))

        country_onehot[self.country_list.index(country_label)] = 1

        before_confirmed_case = 0

        for day in day_list:            

            tmp_data_df = self.target_df[(self.target_df.Date == day) & (self.target_df.country_label == country_label)]            

#             print(day, ps, cr, tmp_data_df)

#             print(self.target_df[self.target_df.])

            try:

                encoded_data.append([np.clip(tmp_data_df.ConfirmedCases.values[0] / 1e6, 0, 1),tmp_data_df.ConfirmedCases_diff_percent.values[0], tmp_data_df.Fatalities_diff_percent.values[0]])

            except:

                print(day, ps, cr, tmp_data_df, day_list)

                return

            if day == day_list[-1]:

                before_confirmed_case = tmp_data_df.ConfirmedCases.values[0]

        return np.array([country_onehot]), np.array([encoded_data]), before_confirmed_case

    

    def predict_encoder_fatalities(self, day_list, country_label):

        encoded_data = []        

        for day in day_list:

            tmp_data_df = self.target_df[(self.target_df.country_label == country_label) & (self.target_df.Date == day)]

            encoded_data.append([tmp_data_df.Fatalities.values[0] / (tmp_data_df.ConfirmedCases.values[0] + 1e-8)])

        return np.array([encoded_data])

        

    def predict_test(self, test_df):

        predict_confirmed_cases = []

        predict_fatalities = []

        country_list = self.country_list.copy()

        for itest in tqdm(test_df.itertuples(), total=len(test_df), position=0):

            # get target date list            

            date_list = itest.Date.split('-')

            delta = timedelta(days=1)

            date = datetime(int(date_list[0]), int(date_list[1]), int(date_list[2])) - delta        

            day_list = []

            self.country_list = country_list.copy()

            for i in range(self.days):

                day_list.append(date.strftime('%Y-%m-%d'))

                date -= delta

            day_list = day_list[::-1]

            

            try:

                country_label = itest.Country_Region + '_' + itest.Province_State

            except:

                country_label = itest.Country_Region

            

            country_onehot, encoded_data_c, bc = self.predict_encoder_confirmed_cases(day_list, country_label)

            encoded_data_f = self.predict_encoder_fatalities(day_list, country_label)

            try:

                confirmed_cases_increase_rate = self.confirmed_cases_model.predict_on_batch({'encoder_input':encoded_data_c, 'country_onehot':country_onehot, 'target_input':np.array([bc]).reshape((1,1)), 'flag_input':np.array([1])})

            except:

                print(bc, itest.Date)

            mortality_rate = self.fatalities_model.predict_on_batch({'encoder_input':encoded_data_f, 'country_onehot':country_onehot, 'target_input': np.array([bc]).reshape((1,1)), 'flag_input':np.array([0])})

            confirmed_cases_increase_rate = confirmed_cases_increase_rate.numpy().reshape(1)

            mortality_rate = mortality_rate.numpy().reshape(1)

#             confirmed_cases = int(bc * (1 + confirmed_cases_increase_rate))

#             fatalities = int(confirmed_cases * mortality_rate)

            predict_confirmed_cases.append(round(confirmed_cases_increase_rate[0]))

            predict_fatalities.append(round(mortality_rate[0]))

            if len(self.target_df[(self.target_df.Date == itest.Date) & (self.target_df.country_label == country_label)]) == 0:

#                 print(itest.Date, confirmed_cases, fatalities, confirmed_cases_increase_rate, mortality_rate)

                # add new predict data

                self.target_df.loc[len(self.target_df)] = [-1, itest.Province_State, itest.Country_Region, itest.Date, round(confirmed_cases_increase_rate[0]), round(mortality_rate[0]), 0, 0, 0, 0, 0, 0]

                self.target_df = self.target_df.sort_values(by='Date')

                self.target_df = self.target_df.sort_values(by='Province_State')

                self.target_df = self.target_df.sort_values(by='Country_Region')

                self.cal_increase_rate(self.target_df)

                

        test_df['ConfirmedCases'] = predict_confirmed_cases

        test_df['Fatalities'] = predict_fatalities

        test_df[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv('submission.csv', index=False)

        return test_df

            

            

        
test_c19 = corona19_predict(datas[1], 61, epochs=200)

y_predict_c, y_train_c = test_c19.train_data_confirmed_cases()

y_predict_f, y_train_f = test_c19.train_data_fatalities()
test_c19 = corona19_predict(datas[1], 61, epochs=200)

country_list = test_c19.country_list.copy()

test_c19.load_models(country_list)

# y_predict_f, y_train_f = test_c19.train_data_fatalities()

# y_predict_c, y_train_c = test_c19.train_data_confirmed_cases()

#rr = test_c19.predict_test(datas[2][datas[2].Country_Region == 'Afghanistan'])

# rr = test_c19.predict_test(datas[2][(datas[2].Country_Region == 'Canada') & (datas[2].Province_State == 'Yukon')])

rr = test_c19.predict_test(datas[2])
rr
datas[1][(datas[1].Country_Region == 'Canada') & (datas[1].Province_State == 'Yukon')]
ttt = pd.read_csv('/kaggle/working/submission.csv')

ttt
result_ = pd.DataFrame()

result_['y_predict'] = y_predict_f[:].reshape(len(y_predict_f))

result_['y_label'] = y_train_f[:, 1]

# print(y_predict[:20], y_train[:20, 0].reshape(20 ,1))

# print(result_[8150:8170])

# print(np.max(y_predict_f), np.max(y_train[:, ]))

plt.plot(y_predict_f[:])

plt.plot(y_train_f[:, 1])

# plt.plot((y_train_f[:, 1] / (y_train_f[:, 0] + 1e-8)))

plt.ylim(0,)

plt.show()



# import matplotlib.pylab as plt

# result_ = pd.DataFrame()

# result_['y_predict'] = y_predict[:].reshape(len(y_predict))

# result_['y_label'] = y_train[:, 1]

# print(y_predict[:20], y_train[:20, 0].reshape(20 ,1))

# print(result_[8150:8170])

print(np.max(y_predict_c), np.max(y_train_c[:, 4]))

plt.plot(y_predict_c[:][1500:2000])

plt.plot(y_train_c[:, 0][1500:2000])

# plt.plot((y_train[:, 1] / (y_train[:, 0] + 1e-8)))

plt.ylim(0,)

plt.show()
len(datas[1][datas[1].Country_Region == 'Afghanistan']) * 0.8
test_c19.target_df#[test_c19.target_df.Country_Region == 'Albania']