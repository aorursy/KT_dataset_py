import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from datetime import date, timedelta

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint



%matplotlib inline
data = pd.read_csv('../input/C.test_data201803.csv',parse_dates = ['time'],index_col = 'time')
data = data.drop('Unnamed: 0',axis = 1)

data_prep = data.copy()
data.head()
def hour_shift(data_prep,df,n):

    if n> 0:

        data_prep[f'{n}_hour_before'] = df['KWH'].shift(n)

    else:

        data_prep[f'{-n}_hour_after'] = df['KWH'].shift(n)
def days_shift(data_prep,df,n):

    if n> 0:

        data_prep[f'{n}_days_before'] = data.loc[data.index - timedelta(days=n)]['KWH'].values

    else:

        data_prep[f'{-n}_days_after'] = data.loc[data.index + timedelta(days=-n)]['KWH'].values
hours_shift_list = [-6,-5,-4,-3,-2,-1,1,3]

days_shift_list = [1,3,7]

for hours in hours_shift_list:

    hour_shift(data_prep,data,hours)

for days in days_shift_list:

    days_shift(data_prep,data,days)

    
def add_more_features(df):

    """

    Add 'month','day','hour' features

    """

    data_prep = df.copy()

    data_prep['month'] = df.index.month

    data_prep['day'] = df.index.day

    data_prep['hour'] = df.index.hour.astype(int)+1

    return data_prep
data_prep = add_more_features(data_prep)
data_prep = data_prep.dropna(axis = 0)
def mape(y_pred,y_label):

    """

    Define Mean Absolute Percentage Error

    """

    

    mape = np.sum(abs(y_pred - y_label)) / np.sum(y_label)

    return mape

    
# XGB

xgb_features = ['KWH','1_hour_before','3_hour_before','1_days_before','3_days_before','7_days_before','month','day','hour']

xgb_target = ['1_hour_after']



X_train,X_test,y_train,y_test = train_test_split(data_prep[xgb_features],

                                                data_prep[xgb_target],

                                                test_size = 0.2,

                                                random_state = 8896)
myxgb = XGBRegressor()



depth_range = np.arange(7,10,1)

num_estimators = np.arange(300,500,50)

grid_params = {

    'max_depth':depth_range,

    'n_estimators':num_estimators

}



gsearch = GridSearchCV(myxgb,

                       grid_params,

                       scoring = 'neg_mean_absolute_error',

                      verbose = 1,

                      n_jobs = -1,

                       cv = 5,

                      refit = True)

print('start training..')

gsearch.fit(X_train,y_train)

print('train finished!')
model = gsearch.best_estimator_
y_predict_validation = model.predict(X_test)

y_predict_train = model.predict(X_train)

xgb_score_validation = mape(y_predict_validation,y_test.values.reshape(1,-1))

xgb_score_train = mape(y_predict_train,y_train.values.reshape(1,-1))

print(f'The Mean_Percentage_Absolute_Error of the XGB model on validation set is {round(xgb_score_validation,3) * 100}%')

print(f'The Mean_Percentage_Absolute_Error of the XGB model on train set is {round(xgb_score_train,3) * 100}%')

#Keras



def buildTrain(train, past_hour, future_hour):

    """

    Create the dataset with time-windows

    """

    X_train, Y_train = [], []

    for i in range(train.shape[0] - future_hour - past_hour):

        X_train.append(np.array(train.iloc[i:i + past_hour]))

        Y_train.append(np.array(train.iloc[i + past_hour:i + past_hour + future_hour]["KWH"]))

    return np.array(X_train), np.array(Y_train)
def normalization(df):

    """

    Scaling the dataset

    """

    df = df.apply(lambda x :(x - np.min(x)) / (np.max(x) - np.min(x)))

    

    return df
# Many-To-Many Predict

# Predict 6 hours in the future using data from past 6 hours

lstm_features = ['KWH','1_days_before','3_days_before','7_days_before','month','day','hour']



print('Dataset Preparing...')

data_lstm = data_prep[lstm_features]

data_norm = normalization(data_lstm)

X_train,y_train = buildTrain(data_norm,past_hour = 6,future_hour = 6)

print('Dataset Preparing finished...')
X_train_lstm,X_test_lstm,y_train_lstm,y_test_lstm = train_test_split(X_train,

                                                y_train,

                                                test_size = 0.1,

                                                random_state = 8896)
def buildManyToManyModel(shape):

    """

    Create LSTM-RNN

    """

    model = Sequential()

    model.add(LSTM(16, input_shape = (shape[1], shape[2]),return_sequences=True))

    # output shape: (6, 1)

    model.add(TimeDistributed(Dense(1)))

    

    model.compile(loss="mse", optimizer="adam")

    model.summary()

    return model
# Transform Target from 2 dimmension to 3 dimension

y_train_lstm = y_train_lstm[:,:,np.newaxis]

y_test_lstm = y_test_lstm[:,:,np.newaxis]

model = buildManyToManyModel(X_train_lstm.shape)

callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")

print('Start fitting LSTM')

model.fit(X_train_lstm, y_train_lstm, epochs= 500, batch_size=128,verbose = 0,validation_data=(X_test_lstm, y_test_lstm), callbacks=[callback])

print('fitting LSTM finished!')
y_predict_test = model.predict(X_test_lstm)

y_predict_train = model.predict(X_train_lstm)

lstm_score_test = mape(y_predict_test,y_test_lstm)

lstm_score_train = mape(y_predict_train,y_train_lstm)

print(f'The Mean_Percentage_Absolute_Error of the LSTM model on validation set is {round(lstm_score_test,3) * 100}%')

print(f'The Mean_Percentage_Absolute_Error of the LSTM model on train set is {round(lstm_score_train,3) * 100}%')
# update: Set December Data as Test Set

# XGB  using MAE as score
test_dec = data_prep[data_prep.index.month == 12]

train_dec = data_prep[data_prep.index.month != 12]

xgb_features = ['KWH','1_hour_before','3_hour_before','1_days_before','3_days_before','7_days_before','month','day','hour']

xgb_target = ['1_hour_after']



X_train_dec = train_dec[xgb_features]

y_train_dec = train_dec[xgb_target]

X_test_dec = test_dec[xgb_features]

y_test_dec = test_dec['1_hour_after']
myxgb2 = XGBRegressor()



depth_range = np.arange(7,10,1)

num_estimators = np.arange(300,500,50)

grid_params = {

    'max_depth':depth_range,

    'n_estimators':num_estimators

}



gsearch = GridSearchCV(myxgb2,

                       grid_params,

                       scoring = 'neg_mean_absolute_error',

                      verbose = 1,

                      n_jobs = -1,

                       cv = 10,

                      refit = True)
print('start training XGB2..')

gsearch.fit(X_train_dec,y_train_dec)

print('XGB2 training finished!')
model = gsearch.best_estimator_

predict_test = model.predict(X_test_dec)

predict_train= model.predict(X_train_dec)

mae_test = mean_absolute_error(predict_test,y_test_dec)

mae_train = mean_absolute_error(predict_train,y_train_dec)
print(f'The Mean_Absolute_Error of the XGB model on Dec.testset is {mae_test}')

print(f'The Mean_Absolute_Error of the XGB model on train set is {mae_train}')
#LSTM on Dec Test.

#6 hours To 1 hour
def buildManyToOneModel(shape):

    """

    Create LSTM-RNN

    """

    model = Sequential()

    model.add(LSTM(16, input_shape = (shape[1], shape[2]),return_sequences=False))

    # output shape: (6, 1)

    model.add(Dense(1)) #不可使用timedistributed

    

    model.compile(loss="mse", optimizer="adam")

    model.summary()

    return model



# Many-To-One Predict

# Predict 1 hour in the future using data from past 6 hours

lstm_features = ['KWH','1_days_before','3_days_before','7_days_before','month','day','hour']



print('Dataset Preparing...')

data_lstm = data_prep[lstm_features]



y_min = np.min(data_lstm.KWH)

y_gap = np.max(data_lstm.KWH) - np.min(data_lstm.KWH)



data_norm_lstm = normalization(data_lstm)







data_test = data_norm_lstm[data_norm_lstm.index.month == 12]

data_train = data_norm_lstm [data_norm_lstm .index.month != 12]





X_train,y_train = buildTrain(data_train,past_hour = 6,future_hour = 1)

X_test,y_test = buildTrain(data_test,past_hour = 6,future_hour = 1)





print('Dataset Preparing finished...')
# NO NEED Transform Target from 2 dimmension to 3 dimension in 'many-to-one' prediction
print(X_train.shape)

print(y_train.shape)
model = buildManyToOneModel(X_train.shape)

callback = EarlyStopping(monitor="loss", patience=10, verbose=0, mode="auto")

print('Start fitting LSTM')

model.fit(X_train, y_train, epochs= 500, batch_size=128, verbose = 0,validation_data=(X_test, y_test), shuffle = True, callbacks=[callback])

print('Fitting Finished')
def reverse_scaling(y_norm,y_min = y_min,y_gap = y_gap):

    '''

    Reverse The Scale

    '''

    y_norm = (y_norm * y_gap) + y_min

    return y_norm
predict = model.predict(X_test)

mae_lstm_test = mean_absolute_error(reverse_scaling(predict),reverse_scaling(y_test))

mae_lstm_train = mean_absolute_error(reverse_scaling(model.predict(X_train)),reverse_scaling(y_train))
print(f'The Mean_Absolute_Error of the LSTM model on Dec_test set is {mae_lstm_test}')

print(f'The Mean_Absolute_Error of the LSTM model on train set is {mae_lstm_train}')
mape(reverse_scaling(predict),reverse_scaling(y_test))