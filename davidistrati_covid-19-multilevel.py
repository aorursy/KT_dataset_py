!pip install pmdarima
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
from numpy.random import seed

seed(1)

import pandas as pd

import numpy as np

from keras.models import Sequential

from pmdarima import auto_arima

from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, BatchNormalization, Dropout, GaussianNoise

from keras.layers import Input, multiply

from keras.models import Model

from keras.optimizers import Adam

from keras import initializers

from datetime import datetime,timedelta

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from tqdm import tqdm

%matplotlib inline

np.random.seed(0)
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

train.head()
no_countr = train['Country_Region'].nunique()

no_province = train['Province_State'].nunique()

no_countr_with_prov = len(train[train['Province_State'].isna()==False]['Country_Region'].unique())

total_forecasting_number = no_province + no_countr - no_countr_with_prov+2

no_days = train['Date'].nunique()

print('there are ', no_countr, 'unique Countries/Regionions, each with ', no_days, 'days of data, all of them having the same dates. There are also ',no_province, 'Provinces/States which can be found on ', no_countr_with_prov, 'countries/ regions.' )
from datetime import date

max_date = train['Date'].max()

min_date = train['Date'].min()

print('dates start on: ', train['Date'].min(), ' and end on: ',max_date)

max_year = pd.to_datetime(max_date).year

min_year = pd.to_datetime(min_date).year

max_month = pd.to_datetime(max_date).month

min_month = pd.to_datetime(min_date).month

max_day = pd.to_datetime(max_date).day

min_day = pd.to_datetime(min_date).day

d0 = date(max_year, max_month, max_day)

d1 = date(min_year, min_month, min_day)

data_days = (d0-d1).days + 1

print('There are:', data_days, 'days with data')



min_date = test['Date'].min()

max_year = pd.to_datetime(max_date).year

min_year = pd.to_datetime(min_date).year

max_month = pd.to_datetime(max_date).month

min_month = pd.to_datetime(min_date).month

max_day = pd.to_datetime(max_date).day

min_day = pd.to_datetime(min_date).day

d0 = date(max_year, max_month, max_day)

d1 = date(min_year, min_month, min_day)

no_test_train_dates = (d0-d1).days + 1

no_test_dates = test['Date'].nunique()

no_test_days = no_test_dates - no_test_train_dates

print('there are ', no_test_train_dates, 'common test and train dates, ', no_test_dates, ' test dates, ', no_test_days, ' unique test dates ')
def l_regr(x,y):

    model = LinearRegression().fit(x, y)

    return model
def predict_next(model, n, X, n_nodes):

    series = [xi for xi in X]

    pred_list = []

    for i in range(n):

        data = series_to_supervised(np.array(series), n_in = n_nodes-1)

        data = data.reshape((data.shape[0], data.shape[1], 1))

        pred = model.predict(data)

        pred = pred[-1][0]

        pred_list.append(pred)

        series.append(pred)

    return pred_list
def series_to_supervised(data, n_in = 10, n_out=1):

    df = pd.DataFrame(data)

    cols = list()

    for i in range(n_in, 0, -1):

        cols.append(df.shift(i))



    for i in range(0, n_out):

        cols.append(df.shift(-i))



    agg = pd.concat(cols, axis=1)



    agg.dropna(inplace=True)

    return agg.values
n_nodes = 10
def train_model(train_x, train_y, n_nodes):

    #     =================

#     model = Sequential()

#     model.add(Conv1D(filters=250, kernel_size=3, activation='relu', input_shape=(n_nodes,1), padding='same', kernel_initializer='truncated_normal'))

#     model.add(LSTM(64, activation='relu', kernel_initializer='truncated_normal'))

#     model.add(BatchNormalization(momentum =0.0)) <====

#     model.add(Dropout(0.2))

#     model.add(Dense(32, activation='relu', kernel_initializer='truncated_normal'))

#     model.add(Dropout(0.2))

#     model.add(Dense(1))

#     model.compile(loss='mse', optimizer='adam')

#     =================

    np.random.seed(0)

    model = Sequential()

    layer_in = Input(shape=(n_nodes,1))

    layer_regr = Conv1D(filters=250, kernel_size=3, activation='relu', padding='same', kernel_initializer='truncated_normal')(layer_in)

    layer_regr = LSTM(64, activation='relu', kernel_initializer='truncated_normal')(layer_regr)

    layer_regr = Dropout(0.3)(layer_regr)

    layer_regr = Dense(32, activation='relu', kernel_initializer='truncated_normal')(layer_regr)

    layer_regr = Dropout(0.2)(layer_regr)

    layer_out = Dense(1,)(layer_regr)

    

    model = Model(inputs=layer_in, outputs=layer_out)

    

    model.compile(loss='mse', optimizer='adam')

#     =================

#     model = Sequential()

#     layer_in = Input(shape=(n_nodes,))

#     layer_regr = Dense(64, activation='relu', kernel_initializer='truncated_normal')(layer_in)

#     layer_regr = BatchNormalization()(layer_regr)

#     layer_class = Dense(32, activation='softmax', kernel_initializer='truncated_normal')(layer_regr)

#     layer_regr = Dense(32, activation='relu', kernel_initializer='truncated_normal')(layer_regr)



#     layer_regr = multiply([layer_regr, layer_class])

#     layer_out = Dense(1,)(layer_regr)

#     model = Model(inputs=layer_in, outputs=layer_out)

    

#     model.compile(loss='mse', optimizer='adam')

    for i in range(600):

        model.fit(train_x, train_y, batch_size=len(train_x), epochs = 1, verbose = 0)

        model.reset_states()

    return model
index = int(((len(test)/no_test_dates)+1)/2)

cases_pred= []

fatalities_pred = []

pbar = tqdm(total=((len(test)/no_test_dates)))

while index < ((len(test)/no_test_dates)+1):

    x = train['ConfirmedCases'].iloc[[i for i in range(no_days*(index-1),no_days*index)]].values

    z = train['Fatalities'].iloc[[i for i in range(no_days*(index-1),no_days*index)]].values

    

    index += 1

    

    no_nul_cases = pd.DataFrame(x)



    if(not no_nul_cases.empty):

        X = [xi[0] for xi in no_nul_cases.values]

        if (len(X) >= 30):

            try:



                new_pred = []

                model_er = l_regr(np.array([i for i in range(len(X))]).reshape(-1, 1), np.log1p(np.array(X)).tolist())

                er_pred = [(model_er.coef_*(len(X)+i) + model_er.intercept_).astype('int')[0] for i in range(no_test_days)]

                er_pred = np.expm1(np.array(er_pred)).tolist()



                train_set = series_to_supervised(X, n_in = n_nodes, n_out=1)

                train_set = np.log1p(train_set)



                train_x, train_y = train_set[:, :-1], train_set[:, -1]

                train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))



                for i in range(5):



                    model = train_model(train_x, train_y, n_nodes)



#                     plotme = series_to_supervised(X, n_in = n_nodes-1, n_out=1)

#                     plotme = np.log1p(plotme)

#                     plotme = plotme.reshape((plotme.shape[0], plotme.shape[1], 1))

                    

                    pred_list = predict_next(model, no_test_days, np.log1p(np.array(X)).tolist(), n_nodes)

                    pred_list = np.expm1(np.array(pred_list)).tolist()

                    

#                     plt.plot([[float('NaN')] for s in range(n_nodes)] + np.expm1(model.predict(plotme)).tolist() + np.array(pred_list).reshape(-1,1).tolist(), color='blue', label='Model fit and prediction')

#                     plt.plot(X, color='red', label='Train Data')

#                     plt.legend()

#                     plt.ylabel('no. of cases')

#                     plt.xlabel('no. of days')

#                     plt.show()

                    

                    new_pred += pred_list

                pred = np.array(new_pred[:no_test_days])

                for i in range(2, 6):

                    pred = np.add( pred, np.array(new_pred[(i-1)*no_test_days:i*no_test_days]))

                pred = pred/5

                pred = pred.tolist()

                if (pred[-1] > 2*er_pred[-1]):

                    pred = er_pred

                    



            except:

                model = l_regr(np.array([i for i in range(len(X))]).reshape(-1, 1),X)

                pred = [(model.coef_*(len(X)+i) + model.intercept_).astype('int')[0] for i in range(no_test_days)]

        else:

            try:

                model = auto_arima(X,seasonal=True, m=12)

                pred = model.predict(no_test_days)

                pred = pred.astype(int)

                pred = pred.tolist()

            except:

                model = l_regr(np.array([i for i in range(len(X))]).reshape(-1, 1),X)

                pred = [(model.coef_*(len(X)+i) + model.intercept_).astype('int')[0] for i in range(no_test_days)]

                

    else:

        pred = [0] * no_test_days

    pred = x[-no_test_train_dates:].astype(int).tolist() + pred

    cases_pred+=pred



    no_nul_fatalities = pd.DataFrame(z)



    if(not no_nul_fatalities.empty):

        Z = [zi[0] for zi in no_nul_fatalities.values]

        if (len(Z) >= 30):

            try:

                new_pred = []

                model_er = l_regr(np.array([i for i in range(len(Z))]).reshape(-1, 1), np.log1p(np.array(Z)).tolist())

                er_pred = [(model_er.coef_*(len(Z)+i) + model_er.intercept_).astype('int')[0] for i in range(no_test_days)]

                er_pred = np.expm1(np.array(er_pred)).tolist()

                train_set = series_to_supervised(Z, n_in = 10, n_out=1)

                train_set = np.log1p(train_set)



                train_x, train_y = train_set[:, :-1], train_set[:, -1]

                train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))

                

                for i in range(5):



                    model = train_model(train_x, train_y)

                    

                    pred_list = predict_next(model, no_test_days, np.log1p(np.array(Z)).tolist(), n_nodes)

                    pred_list = np.expm1(np.array(pred_list)).tolist()



                    new_pred += pred_list

                pred = np.array(new_pred[:no_test_days])

                for i in range(2, 6):

                    pred = np.add( pred, np.array(new_pred[(i-1)*no_test_days:i*no_test_days]))

                pred = pred/5

                pred = pred.tolist()



            except:

                model = l_regr(np.array([i for i in range(len(Z))]).reshape(-1, 1),Z)

                pred = [(model.coef_*(len(Z)+i) + model.intercept_).astype('int')[0] for i in range(no_test_days)]

        else:

            try:

                model = auto_arima(Z, seasonal=False, m=12)

                pred = model.predict(no_test_days)

                pred = pred.astype(int)

                pred = pred.tolist()

            except:

                model = l_regr(np.array([i for i in range(len(Z))]).reshape(-1, 1),Z)

                pred = [(model.coef_*(len(Z)+i) + model.intercept_).astype('int')[0][0] for i in range(1,32)]

    else:

        pred = [0] * no_test_days

    pred = z[-no_test_train_dates:].astype(int).tolist() + pred

    fatalities_pred+=pred

    pbar.update(1)

pbar.close()
if(len(fatalities_pred) == len(test)):

    print('the length of fatalities_pred and cases_pred is the same as the length of test')
submission = pd.DataFrame({'ForecastId': [i for i in range(1,len(cases_pred)+1)] ,'ConfirmedCases': cases_pred, 'Fatalities': fatalities_pred})

filename = 'submission.csv'

submission.to_csv(filename,index=False)