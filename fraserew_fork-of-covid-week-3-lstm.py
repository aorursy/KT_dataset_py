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

n_feats = 2 # cases and fatalities



train_supervised = [series_to_supervised(region_dfs_train[i][['ConfirmedCases','Fatalities']],

                                           n_in=n_in, n_out=n_out, dropnan=True) for i in region_dfs_train]



reshaped_train_data = pd.concat(train_supervised).reset_index(drop=True)
# drop 7000 zero sequences

zero_indices = reshaped_train_data[reshaped_train_data.sum(axis=1)==0].sample(8000).index

model_data = reshaped_train_data.drop(zero_indices,axis=0)
model_data.shape
# LSTM input is 3D (samples, timesteps, feats), output is 2D(samples,feats)

# cross val data

restack = reshaped_train_data.values

x = restack[:,:-2].reshape(len(restack),n_in,2)

y = restack[:,-2:]
print(x.shape,y.shape)
import keras.backend as K



def rmsle(pred,true):

    assert pred.shape[0]==true.shape[0]

    return K.sqrt(K.mean(K.square(K.log(pred+1) - K.log(true+1))))
def build_regressor(lstm_nodes,d1,d2):

    

    # define model

    model = Sequential()

    model.add(LSTM(lstm_nodes, activation='relu', input_shape=(n_in,2)))

    model.add(Dense(d1, activation='relu'))

    model.add(Dense(d2, activation='relu'))

    model.add(Dense(2, activation='relu'))

    model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')

    

    return model
# scikit wrapper function

regressor = KerasRegressor(build_fn = build_regressor,verbose=0)



# grid search parameters

parameters = {'lstm_nodes':[12,40],

             'nb_epoch':[100],

             'batch_size':[32,256],

             'd1':[32,64],

             'd2':[16,32]}





gridsearch = GridSearchCV(estimator = regressor,

                 param_grid = parameters,

                 scoring = 'neg_mean_squared_log_error',

                 cv = 10,

                 n_jobs = -1,

                 verbose =0)
# restack train and val 

gridsearch = gridsearch.fit(x,y)
gridsearch.cv_results_
gridsearch.cv_results_['mean_test_score']
gridsearch.cv_results_['std_test_score']
model = build_regressor(lstm_nodes=gridsearch.best_params_['lstm_nodes'],

                       d1=gridsearch.best_params_['d1'],

                       d2=gridsearch.best_params_['d2'])

model.save("model.h5")
history = model.fit(x, y, epochs=gridsearch.best_params_['nb_epoch'],

                    batch_size=gridsearch.best_params_['batch_size'])
# number of days to predict

pred_days = test_data['Date'].max()-test_data['Date'].min()

pred_days
# first batch of predictions

first_predict_date = test_data['Date'].min()

pred_data = {key:region_dfs_train[key].loc[(region_dfs_train[key]['Date']>=(first_predict_date-pd.DateOffset(days=n_in)))&

                                    (region_dfs_train[key]['Date']<first_predict_date)] 

             for key in region_dfs_train}

test_reshaped = [pred_data[i][['ConfirmedCases','Fatalities']].values.reshape(1,n_in,2) for i in pred_data]

first_input = np.vstack(test_reshaped)

first_pred = model.predict(first_input)
# iterate prediction output back into model input for the next days

pin = [first_input]

pout = [first_pred]



# first prediction is done outside of loop, need to loop for following 41 days

for i in range(42):

    p = model.predict(pin[i])

    pout.append(p)

    t= np.insert(pin[i],n_in,pout[i],axis=1)[:,1:,:]

    pin.append(t)
# create the prediction dataframe

pred_df = pd.DataFrame(np.concatenate(pout))

pred_df.columns = ['ConfirmedCases','Fatalities']

pred_df['Date'] = np.repeat(test_data['Date'].unique(),len(ids))

pred_df['Province_State'] = list(test_data.drop_duplicates(subset=['Province_State','Country_Region'])['Province_State'])*43

pred_df['Country_Region'] = list(test_data.drop_duplicates(subset=['Province_State','Country_Region'])['Country_Region'])*43



pred_df = pred_df.sort_values(by=['Country_Region','Date']).reset_index(drop=True)
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
sub.to_csv("submission.csv",index=False)