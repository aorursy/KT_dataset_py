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

import seaborn as sns



from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.models import Sequential

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import StandardScaler
raw_train_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

raw_test_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

sub_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')
# change dtypes, fillna

train_data = raw_train_data

train_data['Date'] = pd.to_datetime(train_data['Date'])

train_data['Province_State'] = train_data['Province_State'].fillna('None')



test_data = raw_test_data

test_data['Date'] = pd.to_datetime(test_data['Date'])
# unique identifiers for each province and country

ids = (

    train_data[['Province_State','Country_Region']]

    .drop_duplicates().reset_index(drop=True).to_dict('index')

)
# dict of dfs for each province

region_dfs_train = {i:train_data[(train_data['Province_State']==ids[i]['Province_State'])&

          (train_data['Country_Region']==ids[i]['Country_Region'])] for i in ids}
# function to convert series into supervised learning problem

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    """

    Frame a time series as a supervised learning dataset.

    Arguments:

        data: Sequence of observations as a list or NumPy array.

        n_in: Number of lag observations as input (X).

        n_out: Number of observations as output (y).

        dropnan: Boolean whether or not to drop rows with NaN values.

    Returns:

        Pandas DataFrame of series framed for supervised learning.

    """

    n_vars = 1 if type(data) is list else data.shape[1]

    df = pd.DataFrame(data)

    cols, names = list(), list()

    # input sequence (t-n, ... t-1)

    for i in range(n_in, 0, -1):

        cols.append(df.shift(i))

        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)

    for i in range(0, n_out):

        cols.append(df.shift(-i))

        if i == 0:

            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

        else:

            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

     # put it all together

    agg = pd.concat(cols, axis=1)

    agg.columns = names

    # drop rows with NaN values

    if dropnan:

        agg.dropna(inplace=True)

    return agg
# prepare train and val data into sequences with n_in = lag time steps and n_out = future time steps



n_in = 6

n_out = 1

n_feats = 1 # cases and fatalities



train_cases = [series_to_supervised(region_dfs_train[i][['ConfirmedCases']],

                                           n_in=n_in, n_out=n_out, dropnan=True) for i in region_dfs_train]



train_fatal = [series_to_supervised(region_dfs_train[i][['Fatalities']],

                                           n_in=n_in, n_out=n_out, dropnan=True) for i in region_dfs_train]



reshaped_cases = pd.concat(train_cases).reset_index(drop=True)

reshaped_fatal = pd.concat(train_fatal).reset_index(drop=True)
# # drop 7000 zero sequences

# zero_indices = reshaped_train_data[reshaped_train_data.sum(axis=1)==0].sample(8000).index

# model_data = reshaped_train_data.drop(zero_indices,axis=0)
# LSTM input is 3D (samples, timesteps, feats), output is 2D(samples,feats)

# cross val data

restack_cases = reshaped_cases.values

restack_fatal = reshaped_fatal.values

x_cases = restack_cases[:,:-1].reshape(len(restack_cases),n_in,1)

y_cases = restack_cases[:,-1:]

x_fatal = restack_fatal[:,:-1].reshape(len(restack_fatal),n_in,1)

y_fatal = restack_fatal[:,-1:]
print(x_cases.shape,y_cases.shape,x_fatal.shape,y_fatal.shape)
# train test split for cases

tr_x_c, val_x_c, tr_y_c, val_y_c = train_test_split(x_cases, 

                                                    y_cases, 

                                                    test_size=0.2, 

                                                    random_state=0)



# train test split for fatalities

tr_x_f, val_x_f, tr_y_f, val_y_f = train_test_split(x_fatal, 

                                                    y_fatal, 

                                                    test_size=0.2, 

                                                    random_state=0)
tr_x_c.shape,val_x_c.shape,tr_x_f.shape,val_x_f.shape
import keras.backend as K



def rmsle(pred,true):

    assert pred.shape[0]==true.shape[0]

    return K.sqrt(K.mean(K.square(K.log(pred+1) - K.log(true+1))))
def build_regressor(rmsle,lstm_nodes,d1,d2,dropout):

    

    # define model

    model = Sequential()

    model.add(LSTM(lstm_nodes, activation='relu', input_shape=(n_in,1)))

    model.add(Dense(d1, activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(d2, activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(1, activation='relu'))

    model.compile(optimizer='adam', loss=rmsle)

    

    return model
cases_model = build_regressor(lstm_nodes=10,

                       d1=64,

                       d2=32,

                       dropout=0.05,

                             rmsle=rmsle)



history_c = cases_model.fit(tr_x_c, tr_y_c, epochs=100,batch_size=64,

                            validation_data=(val_x_c,val_y_c))
fatal_model = build_regressor(lstm_nodes=10,

                       d1=64,

                       d2=32,

                       dropout=0.05,

                             rmsle=rmsle)



history_f = fatal_model.fit(tr_x_f, tr_y_f, epochs=100,batch_size=64,

                            validation_data=(val_x_f,val_y_f))
plt.figure(figsize=(8,6))

plt.plot(history_c.history['loss'], label='Train')

plt.plot(history_c.history['val_loss'], label='Test')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(loc='upper left')

plt.show()
plt.figure(figsize=(8,6))

plt.plot(history_f.history['loss'], label='Train')

plt.plot(history_f.history['val_loss'], label='Test')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(loc='upper left')

plt.show()
# number of days to predict

pred_days = test_data['Date'].max()-test_data['Date'].min()

pred_days
# first batch of predictions

first_predict_date = test_data['Date'].min()

pred_data = {key:region_dfs_train[key].loc[(region_dfs_train[key]['Date']>=(first_predict_date-pd.DateOffset(days=n_in)))&

                                    (region_dfs_train[key]['Date']<first_predict_date)] 

             for key in region_dfs_train}



test_cases = [pred_data[i]['ConfirmedCases'].values.reshape(1,n_in,1) for i in pred_data]

test_fatal = [pred_data[i]['Fatalities'].values.reshape(1,n_in,1) for i in pred_data]



first_c_input = np.vstack(test_cases)

first_f_input = np.vstack(test_fatal)



first_c_pred = cases_model.predict(first_c_input)

first_f_pred = fatal_model.predict(first_f_input)
# iterate prediction output back into model input for the next days

c_pin = [first_c_input]

c_pout = [first_c_pred]



# first prediction is done outside of loop, need to loop for following 41 days

for i in range(42):

    p = cases_model.predict(c_pin[i])

    c_pout.append(p)

    t= np.insert(c_pin[i],n_in,c_pout[i],axis=1)[:,1:,:]

    c_pin.append(t)
# iterate prediction output back into model input for the next days

f_pin = [first_f_input]

f_pout = [first_f_pred]



# first prediction is done outside of loop, need to loop for following 41 days

for i in range(42):

    p = fatal_model.predict(f_pin[i])

    f_pout.append(p)

    t= np.insert(f_pin[i],n_in,f_pout[i],axis=1)[:,1:,:]

    f_pin.append(t)
# create the prediction dataframe

pred_df = pd.DataFrame(np.concatenate(c_pout))

pred_df.columns = ['ConfirmedCases']

pred_df['Date'] = np.repeat(test_data['Date'].unique(),len(ids))

pred_df['Province_State'] = list(test_data.drop_duplicates(subset=['Province_State','Country_Region'])['Province_State'])*43

pred_df['Country_Region'] = list(test_data.drop_duplicates(subset=['Province_State','Country_Region'])['Country_Region'])*43



pred_df = pred_df.sort_values(by=['Country_Region','Date']).reset_index(drop=True)



pred_df['Fatalities'] = (np.concatenate(c_pout))
pred_df[pred_df['Date']=='2020-05-07']['ConfirmedCases'].sum(),pred_df[pred_df['Date']=='2020-05-07']['Fatalities'].sum()
def rmsle_check(pred,true):

    p = np.log(pred+1)

    a = np.log(true+1)

    s = np.sum((p-a)**2)

    return np.sqrt((1/len(pred))*s)
true = train_data[train_data['Date']>=test_data['Date'].min()]

pred = pred_df[pred_df['Date']<=train_data['Date'].max()]



rmsle_check(pred['ConfirmedCases'],true['ConfirmedCases']),rmsle_check(pred['Fatalities'],true['Fatalities'])
sub = pred_df[['ConfirmedCases','Fatalities']]

sub['ForecastId'] = test_data['ForecastId']

sub
sub.to_csv("submission.csv",index=False)