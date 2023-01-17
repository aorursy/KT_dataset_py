# Standard libraries for data wrangling & viz

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

import pandas as pd



# LSTM modelling & Simple LR

import tensorflow as tf



# Automated forecasts

from fbprophet import Prophet

from fbprophet.plot import add_changepoints_to_plot



# Preprocessing & utlity tools

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, ParameterGrid

from sklearn.metrics import mean_squared_error



# curve smoothing

from scipy.signal import savgol_filter



# sklearn ML models

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ARDRegression, SGDRegressor, BayesianRidge

from sklearn.svm import SVR, LinearSVR

from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor



# XGBoosted Tree model

from xgboost import XGBRegressor



# Fancy progress bars

import tqdm



# General configurations

%matplotlib inline

style.use('ggplot')



# supress warning messages

import warnings

warnings.simplefilter(action='ignore')

tf.autograph.set_verbosity(1)
# load the CSV files and parse the date columns

train = pd.read_csv("../input/into-the-future/train.csv", parse_dates=['time'])

test = pd.read_csv("../input/into-the-future/test.csv", parse_dates=['time'])



train.shape, test.shape
train.head()
train.info()
# Missing values?

train.isna().sum().sum(), test.isna().sum().sum()
# dropping ID see some summary stats

train.iloc[:, 1:].describe()
# min and max for test and train data

print ("Train min time: {}\nTrain max time: {}".format(train.time.min(), train.time.max()))

print ("\nTest min time: {}\nTest max time: {}".format(test.time.min(), test.time.max()))
# Time delta for train and test data in seconds

((train.time.max() - train.time.min()).total_seconds(),

(test.time.max() - test.time.min()).total_seconds())
# lets check data freq

train.time.diff()[1:4]
# same time freq throughout?

print ("Train freq 10s throughout?\n>>", all(train.time.diff()[1:] == np.timedelta64(10, 's')))

print ("\nTest freq 10s throughout?\n>>", all(test.time.diff()[1:] == np.timedelta64(10, 's')))
# is the dF sorted according to the datatime? (Ascending Order)

train.time.is_monotonic_increasing, test.time.is_monotonic_increasing
# visualizing the dataset (Train + Test)

temp = pd.date_range(train.time.min(), test.time.max(), periods=20)

f, ax = plt.subplots(figsize=(20, 10), nrows=2)



ax[0].plot('time', 'feature_1', c='b', data=pd.concat([train, test]))

ax[0].vlines(test.time.min(), 350, 750, color='g', ls='dotted', label='Train End')

ax[0].set(title="Feature 1 vs Time", xlabel='Time', ylabel='Feature_1', xticks=temp)

ax[0].set_xticklabels(labels=temp.strftime('%H:%M:%S'))

ax[0].legend()



ax[1].plot('time', 'feature_2', c='r', data=pd.concat([train, test]))

ax[1].vlines(test.time.min(), 47e3, 55e3, color='g', ls='dotted', label='Train End')

ax[1].set(title="Feature 2 vs Time", xlabel='Time', ylabel='Feature_2', xticks=temp)

ax[1].set_xticklabels(labels=temp.strftime('%H:%M:%S'))

ax[1].legend();
test.feature_1.plot(figsize=(20, 5), title='Test: Feature 1 Vs Time')

plt.xticks(range(0, len(test), 20), test.time.iloc[::20].dt.strftime("%H:%M:%S"))

plt.xlabel("Time")

plt.ylabel("Feature 1");
stride = 150

f, ax = plt.subplots(figsize=(20, 10), nrows=2, ncols=2)

ax = ax.ravel()

for i in range(0, len(train), stride):

    train.iloc[i: i+stride, 3].plot(ax=ax[i//stride], c='r', title="Portion: "+str(i//stride+1))
smooth = 201

f, ax = plt.subplots(figsize=(20, 8), nrows=2)

train.feature_2.plot(alpha=0.5, ax=ax[0], color='b')

ax[0].plot(savgol_filter(train.feature_2[1:], smooth, 3), color='r', label='Smoothed Curve')

ax[0].legend()



# custom curve to fit the general trend (curve apprx)

temp = 1 - (1 / (1.008 ** train.id))

ax[1].plot(temp, label='Trend Curve')

ax[1].legend();
f, ax = plt.subplots(figsize=(20, 5), ncols=2)

ax[0].set_title('Autocorelation: Feature_1')

pd.plotting.autocorrelation_plot(train.feature_1, ax=ax[0], label='Train')

pd.plotting.autocorrelation_plot(test.feature_1, ax=ax[0], label='Test')

ax[0].set_xticks(range(0, 600, 50))

ax[0].grid(True)



ax[1].set_title('Autocorelation: Feature_2')

pd.plotting.autocorrelation_plot(train.feature_2, ax=ax[1])

ax[1].grid(True);
## creating simple features

train['Hour'] = train.time.dt.hour

train['Minute'] = train.time.dt.minute

train['Second'] = train.time.dt.second

# Seconds elapsed since start

train['S_since_start'] = (train.time - train.time.min()).apply(lambda x: x.total_seconds())

# Minutes elapsed since start

train['M_since_start'] = train['S_since_start'] // 60



# smooth out the feature_2

# train['feature_2'] = np.concatenate([[47730], savgol_filter(train.feature_2[1:], smooth, 3)])



## repeat same for test dataset as well

test['Hour'] = test.time.dt.hour

test['Minute'] = test.time.dt.minute

test['Second'] = test.time.dt.second

# Seconds elapsed since start

test['S_since_start'] = (test.time - train.time.min()).apply(lambda x: x.total_seconds())

# Minutes elapsed since start

test['M_since_start'] = test['S_since_start'] // 60
# creating diff features for feature_1, We combine test 

# and train so that test's initial rows would have the

# correct difference values instead of being NANs

data = pd.concat([train, test]).reset_index(drop=True)

    

# trend approx

best_ = (0, 0) # value, score



for i in tqdm.tqdm(np.linspace(1, 2, 10000)):

    train['trend_apprx'] = 1 - (1 / (i ** train.id))

    temp = train.corr()['feature_2'].loc['trend_apprx']

    if temp > best_[1]:

        best_ = (i, temp)

        

data['trend_apprx'] = 1 - (1 / (best_[0] ** data.id))

print ("Best value that captures trend: {}\nCorr value: {}".format(*best_))

data.head()
def batch_scaler(col, passthrough=['id', 'time', 'feature_2', 'trend_apprx']):

    '''A function to scale the values.

    - Performs Min-Max scaling on those values engineered from time

    - Performs Z-score scaling on the Numeric values

    '''

    if col.name in passthrough:

        'Ignore some columns which doesnot or should be scaled'

        return col

    

    if 'feature' in col.name: 

        'all numeric values are ensured to have name `feature_{}`'

        return (col - col.mean()) / col.std()

    

    else:

        return col / col.max()    
# Let's scale

data = data.apply(batch_scaler, axis=0)

data.head()
# split them back to Train/Test

train = data.iloc[:len(train)]

test = data.iloc[len(train):]



train.shape, test.shape
# 80 - 20 split

train_len = int(len(train) * .80)

train_len, len(train) - train_len
# split the train to Train, Val (Already scaled & remove unnecessary columns)

Train = train.iloc[1:train_len].drop(['id', 'time'], axis=1)

Val = train.iloc[train_len:].drop(['id', 'time'], axis=1)



Train.shape, Val.shape
model = LinearRegression()

model = model.fit(Train.trend_apprx.values.reshape(-1, 1), Train.feature_2)



print ("Error:", mean_squared_error(Val.feature_2, model.predict(Val.trend_apprx.values.reshape(-1, 1)), squared=False))



# plot model predictions

ax = train.feature_2.plot(figsize=(15, 5))

ax.plot(model.intercept_ + (train.trend_apprx * model.coef_));
test['feature_2'] = model.predict(test.trend_apprx.values.reshape(-1, 1))



# Visualize Predictions

ax = train['feature_2'].plot(figsize=(15, 5), title='Model 0: Predictions', color='g')

ax.vlines(train.id.max(), 48e3, 54e3, ls='dotted')

test.set_index('id')['feature_2'].plot(ax=ax, color='r');



# save model as our baseline

test[['id', 'feature_2']].to_csv("LR_solo_feature_sub.csv", index=False)
# Simple linear regression model

tf.keras.backend.clear_session()

model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Input(shape=(Train.shape[1]-1,)))

model.add(tf.keras.layers.Dense(1))



model.compile(loss='mse', 

              metrics=tf.keras.metrics.RootMeanSquaredError('rmse'), 

              # after having experimented with other optimizers, SGD was chosen to perform best

              optimizer=tf.keras.optimizers.SGD(lr=0.005, nesterov=True, momentum=0.8)

             )



model.summary()
hist = model.fit(

    Train.drop(["feature_2"], axis=1),

    Train['feature_2'],

    validation_data=(Val.drop(['feature_2'], axis=1), Val['feature_2']),

    batch_size=8, # small batch size also seems to give better results

    epochs=200,

    verbose=0,

    callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, monitor='val_rmse', restore_best_weights=True),

              tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.25, monitor='val_rmse')],

)



print ("Final loss acheived in {} epochs:\n>>> Train RMSE: {}\n>>> Test RMSE: {}"

       .format(len(hist.epoch), hist.history['rmse'][-1], hist.history['val_rmse'][-1]))



# plot the model performance

(pd.DataFrame(hist.history)[["rmse", "val_rmse"]]

 .plot(figsize=(15, 5), title='Model 1: Performance', ylim=[0, 5000])

 .set(ylabel='Loss', xlabel='No of Epochs'));
# Make predictions on the Test dataset

test['feature_2'] = model.predict(test.drop(['id', 'time', 'feature_2'], axis=1))



# Plot the predictions to verify

ax = train['feature_2'].plot(figsize=(15, 5), title='Model 1: Predictions', color='g')

ax.vlines(train.id.max(), 48e3, 56e3, ls='dotted')

test.set_index('id')['feature_2'].plot(ax=ax, color='r');
# Simple linear regression model with dropouts

tf.keras.backend.clear_session()

model = tf.keras.models.Sequential()



model.add(tf.keras.layers.Input(shape=(Train.shape[1]-1,)))

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(1))



model.compile(loss='mse', 

              metrics=tf.keras.metrics.RootMeanSquaredError('rmse'), 

              # decreased Learning rate but we run it a bit longer

              optimizer=tf.keras.optimizers.SGD(lr=0.01, nesterov=True, momentum=0.8))



hist = model.fit(

    Train.drop(["feature_2"], axis=1),

    Train['feature_2'],

    validation_data=(Val.drop(['feature_2'], axis=1), Val['feature_2']),

    batch_size=8,

    epochs=500,

    verbose=0,

    callbacks=[tf.keras.callbacks.EarlyStopping(patience=25, monitor='val_rmse', restore_best_weights=True),

              tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.75, monitor='val_rmse')],

)



print ("Final loss acheived in {} epochs:\n>>> Train RMSE: {}\n>>> Test RMSE: {}"

       .format(len(hist.epoch), hist.history['rmse'][-1], hist.history['val_rmse'][-1]))



# plot the model performance

(pd.DataFrame(hist.history)[["rmse", "val_rmse"]]

 .plot(figsize=(15, 5), title='Model 2: Performance', ylim=[0, 1500])

 .set(ylabel='Loss', xlabel='No of Epochs'));
# Predict on the test dataset

test['feature_2'] = model.predict(test.drop(['id', 'time', 'feature_2'], axis=1))



# Visualize Predictions

ax = train['feature_2'].plot(figsize=(15, 5), title='Model 2: Predictions', color='g')

ax.vlines(train.id.max(), 48e3, 54e3, ls='dotted')

test.set_index('id')['feature_2'].plot(ax=ax, color='r');
test[['id', 'feature_2']].to_csv("LR_basic_sub.csv", index=False)
# scale our target since we would also be feeding it as partial inputs

scaler = StandardScaler()

Train_scaled = scaler.fit_transform(Train.feature_2.values.reshape(-1, 1))

Val_scaled = scaler.transform(Val.feature_2.values.reshape(-1, 1))



# Std -> 1; Mean -> 0

Train_scaled.std(), Train_scaled.mean()
# How many past rows would we feed our LSTM model?

minutes = 10

inp_size = 6 * minutes



# High batch size seems to perform a bit better than 8

batch_size = 128



# window, flat_map, preprocess, Shuffle and Batch

Train_dataset = (

    tf.data.Dataset

    .from_tensor_slices(Train_scaled)

    .window(inp_size+1, shift=1, drop_remainder=True)

    .flat_map(lambda x: x.batch(inp_size+1))

    # split to X, y

    .map(lambda x: (x[:-1], x[-1]))

    # shuffle only for Train

    .shuffle(100)

    .batch(batch_size, drop_remainder=True)

    .repeat().prefetch(1)

)



# Same as above except we don't have to shuffle the dataset. Also 

# we set batchsize as 32 since we have very few Val instances

Val_dataset = (

    tf.data.Dataset

    .from_tensor_slices(Val_scaled)

    .window(inp_size+1, shift=1, drop_remainder=True)

    .flat_map(lambda x: x.batch(inp_size+1))

    .map(lambda x: (x[:-1], x[-1]))

    .batch(32, drop_remainder=True).cache()

)



Train_dataset, Val_dataset
# clear residual layer blocks from memory

tf.keras.backend.clear_session()



model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape=(inp_size,)))

# reshape Input for GRU

model.add(tf.keras.layers.Reshape(target_shape=(1, inp_size)))

model.add(tf.keras.layers.GRU(16, recurrent_dropout=0.1, dropout=0.1))

model.add(tf.keras.layers.Dense(1))



model.compile(

    loss='mse', 

    # since values are scaled & targets are scaled, Adam seems to work well

    optimizer=tf.keras.optimizers.Adam(0.001),

    metrics=tf.keras.metrics.RootMeanSquaredError('rmse'))



model.summary()
hist = model.fit(

    Train_dataset, epochs=500, 

    validation_data=Val_dataset, 

    steps_per_epoch=len(Train)//batch_size,

    verbose=0,

    callbacks=[tf.keras.callbacks.EarlyStopping(patience=25, monitor='val_rmse'),

              tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_rmse')])



print ("Final loss acheived in {} epochs:\n>>> Train RMSE: {}\n>>> Test RMSE: {}"

       .format(len(hist.epoch), hist.history['rmse'][-1], hist.history['val_rmse'][-1]))



# plot the model performance

(pd.DataFrame(hist.history)[["rmse", "val_rmse"]]

 .plot(figsize=(15, 5), title='Model 3: Performance', ylim=[0, 0.5])

 .set(ylabel='Loss', xlabel='No of Epochs'));
# feed in past data as input features for our model

preds = Val_scaled[-inp_size:].ravel()

for i in tqdm.trange(len(test)):

    

    # predict and concatenate

    temp = model.predict(preds[-inp_size:].reshape(1, inp_size)).ravel()        

    preds = np.concatenate([preds, temp], axis=0)



# drop the past data we added in the first step

preds = preds[inp_size:]

preds.shape
# predictions need to rescaled

test['feature_2'] = scaler.inverse_transform(preds)



# Plot the model predictions to verify

ax = train['feature_2'].plot(figsize=(15, 5), title='Model 3: Predictions', color='g')

ax.vlines(train.id.max(), 48e3, 54e3, ls='dotted')

test.set_index('id')['feature_2'].plot(ax=ax, color='r');
test[['id', 'feature_2']].to_csv("LSTM_op.csv", index=False)
# drop the feature_2 we had predicted from previous models

test.drop('feature_2', axis=1, inplace=True)
# let's concatenate test and train once again to create lag features

data = pd.concat([train, test])



for i in range(6, 31, 6):

    data[f'feature_2_lag_{i}'] = data.feature_2.shift(i).fillna(0.0)



data.shape
# lets obtain our Train/Val/test back from our data

Train = data.iloc[1:train_len].drop(['id', 'time'], axis=1)

Val = data.iloc[train_len:len(train)].drop(['id', 'time'], axis=1)

test = data.iloc[len(train):]



Train.shape, Val.shape, test.shape
# we set seed to be able to produce similar 

# op when performing Grid Searches

np.random.seed(1)



scores = []



# We try several different models with default configs

for name, model in [

    # ensemble models

    ("RFR", RandomForestRegressor()), 

    ("ETSR", ExtraTreesRegressor()),

    ("ADBR", AdaBoostRegressor()),

    

    # linear models

    ("LR", LinearRegression()), ("RR", Ridge()), ("Lasso", Lasso()), 

    ("ARDR", ARDRegression()), ("LSVR", LinearSVR()), ("BR", BayesianRidge()),

    ("SVR", SVR()), ("SGDR", SGDRegressor()),

    

    # tree models

    ("DTR", DecisionTreeRegressor()),

    ("ETR", ExtraTreeRegressor()),

    

    # boosting models

    ("XGBR", XGBRegressor())

]:

    

    model.fit(Train.drop('feature_2', 1), Train.feature_2)

    score = mean_squared_error(Val.feature_2, model.predict(Val.drop('feature_2', 1)), squared=False)

    scores.append((name, score))



for name, score in sorted(scores, key=lambda x : x[1]):

    print ("{:5} Model has scored: {:.2f}".format(name, score))
# some parameters to choose the `better` model from

grid_params = {

    

    "LR": {

        "fit_intercept": [True, False],

        "normalize": [True, False]

    },

    

    "Lasso": {

        "alpha": [10., 5., 1., 0.1, 0.001, 0.0001],

        "fit_intercept": [True, False],

        "positive": [True, False],

    },

    

    "RR": {

        "alpha": [10., 5., 1., 0.1, 0.001, 0.0001],

        "fit_intercept": [True, False],

        "normalize": [True, False],

    },

    

    "ARDR": {

        "alpha_1": [1e-6, 1e-5, 1e-4, 1e-7],

        "alpha_2": [1e-6, 1e-5, 1e-4, 1e-7],

        "lambda_1": [1e-6, 1e-5, 1e-4, 1e-7],

        "lambda_2": [1e-6, 1e-5, 1e-4, 1e-7],        

    },

    

    "LSVR": {

        "loss": ['epsilon_insensitive', 'squared_epsilon_insensitive'],

        "C": [1, 10, 0.1, 0.01, 0.001, 100],   

    }

}



# models initialised with default configs

better_models = [

    

    # linear models

    ("LR", LinearRegression()), 

    ("RR", Ridge()), 

    ("Lasso", Lasso()), 

    ("ARDR", ARDRegression()), 

    ("LSVR", LinearSVR()),

]



best_models = []



for name, model in better_models:

    

    # creating the grid object to fine tune

    grid = GridSearchCV(

        model, grid_params[name], n_jobs=-1, 

        scoring='neg_root_mean_squared_error', 

        cv=TimeSeriesSplit(n_splits=3))

    

    grid.fit(Train.drop("feature_2", axis=1), Train['feature_2'])

    

    print ("\nFor model {}, the best score: \nOn Train is: {:.2f} \n On Test is: {:.2f}"

           .format(name, -grid.best_score_, 

                   mean_squared_error(Val.feature_2, grid.predict(Val.drop('feature_2', axis=1)), squared=False)))

    

    # Save the fine tuned model to List

    best_models.append((name, grid.best_estimator_))
# Voting regressor for this purpose

model = VotingRegressor(best_models, n_jobs=-1)

model.fit(Train.drop("feature_2", axis=1), Train['feature_2'])



# how does it score?

mean_squared_error(Val.feature_2, model.predict(Val.drop('feature_2', axis=1)), squared=False)
for i in range(len(test)//6):

    temp = test.iloc[i*6:(i+1)*6].drop(['id', 'time', 'feature_2'], axis=1).fillna(0.0)

    test.iloc[i*6:(i+1)*6, 3] = model.predict(temp)

    

    for i in range(6, 31, 6):

        test[f'feature_2_lag_{i}'] = test.feature_2.shift(i).fillna(0.0)

    

test.iloc[(i+1)*6:, 3] = model.predict(test.iloc[(i+1)*6:].drop(['id', 'time', 'feature_2'], axis=1))



ax = train['feature_2'].plot(figsize=(15, 5), title='Model 4: Predictions', color='g')

ax.vlines(train.id.max(), 48e3, 55e3, ls='dotted')

test.set_index('id')['feature_2'].plot(ax=ax, color='r');
test[['id', 'feature_2']].to_csv("ML_op.csv", index=False)
# train dataset for prophet

fbp_train = train.iloc[1:, [1, 3]]

fbp_train = fbp_train.rename({"time":"ds", "feature_2": "y"}, axis=1)

fbp_train.head(5)
# basic prophet model with fine tuning

model = Prophet()

model.fit(fbp_train);
# Create dataset for forecast

forecast = model.make_future_dataframe(periods=375, freq='10S')



# Predict using our model

forecast = model.predict(forecast)

forecast.tail(3)
f = model.plot(forecast, figsize=(15, 5))

add_changepoints_to_plot(f.gca(), model, forecast)

f.gca().set_title("Model 5: Forecast with uncertainity")

f.gca().set(ylim=[45e3, 58e3]);
# let's save the output we had predicted

test['feature_2'] = forecast.iloc[len(train)-1:, -1].values

test[['id', 'feature_2']].to_csv("prophet_basic_pred.csv", index=False)
params_grid = {'changepoint_prior_scale':[0.05, 0.075, 0.001, 0.0001],

              'n_changepoints' : [1, 2, 3, 4, 10, 30]}



# sklearns function to create param grids

grid = ParameterGrid(params_grid)



"Total models possible:", len(list(grid))
# saving results (our imitation of grid.cv_results_ ;)

grid_results = pd.DataFrame(columns = ['RMSE','Parameters'])



for g in tqdm.tqdm(grid):

    model = Prophet(changepoint_prior_scale = g['changepoint_prior_scale'],

                         n_changepoints = g['n_changepoints'])

    

    model.fit(fbp_train[:train_len-1])

    

    result = mean_squared_error(fbp_train.iloc[train_len-1:, -1].values, 

                                model.predict(fbp_train.iloc[train_len-1:])['yhat'].values, 

                                squared=False)

    

    grid_results = grid_results.append({'RMSE': result, 'Parameters': g}, ignore_index=True)



# we keep only the top 5 parameters

grid_results = grid_results.sort_values('RMSE')[:5].reset_index(drop=True)

grid_results
# model with best parameter config

model = Prophet(**grid_results.Parameters[0])

model.fit(fbp_train)



# make dataframe for prediction & predict

forecast = model.make_future_dataframe(periods=375, freq='10S')

forecast = model.predict(forecast)



# plot the predictions with uncertainity estimates

f = model.plot(forecast, figsize=(15, 5))

add_changepoints_to_plot(f.gca(), model, forecast)

f.gca().set_title("Model 5 (fine-tuned): Forecast with uncertainity")

f.gca().set(ylim=[45e3, 56e3]);
# save the predictions to csv file

test['feature_2'] = forecast.iloc[len(train)-1:, -1].values

test[['id', 'feature_2']].to_csv("prophet_fine_tuned_pred.csv", index=False)
# Add in feature_1 as well in the predictions

fbp_train = train.iloc[1:, [1, -1, 3]]

fbp_train = fbp_train.rename({"time":"ds", "feature_2": "y"}, axis=1)

fbp_train.head(5)
# basic model with previous fine tuned param configs

model = Prophet(**grid_results.Parameters[0])

# add a new regressor 

model.add_regressor('trend_apprx')

# fit the model to our new dataset

model.fit(fbp_train);
# make test dataset

forecast = model.make_future_dataframe(periods=375, freq='10S')

# additionally we add in the feature_1 column to test dataset

forecast['trend_apprx'] = pd.concat([train['trend_apprx'], test['trend_apprx']]).values[1:]



# forecast on the test_dataset

forecast = model.predict(forecast)



# plot the predictions with uncertainity estimates

f = model.plot(forecast, figsize=(15, 5))

add_changepoints_to_plot(f.gca(), model, forecast)

f.gca().set_title("Model Forecast with uncertainity")

f.gca().set(ylim=[45e3, 58e3]);
# save the preditions to csv File

test['feature_2'] = forecast.iloc[len(train)-1:, -1].values

test[['id', 'feature_2']].to_csv("prophet_fine_r2_pred.csv", index=False)
final = pd.DataFrame({'id':test.id})



for loc in ("./LR_basic_sub.csv", "./LR_solo_feature_sub.csv", "./LSTM_op.csv", 

            "./ML_op.csv", "./prophet_basic_pred.csv", 'prophet_fine_r2_pred.csv', 

            './prophet_fine_tuned_pred.csv'):

    final = final.merge(pd.read_csv(loc).rename({'feature_2':loc.split("./")[-1].split(".")[0]}, axis=1), on='id')



final['Prophet_mean'] = final[['prophet_basic_pred', 'prophet_fine_r2_pred', 'prophet_fine_tuned_pred']].mean(axis = 1)

final['avg'] = final.drop(['id', 'LSTM_op', 'ML_op'], axis=1).mean(axis=1)



final.head()
ax = final.set_index('id').plot(figsize=(25, 10), ylim=[52e3, 57e3], xlim=[150, 950], title='Likely Model Predictions Together')

plt.plot(savgol_filter(train.feature_2, 101, 3));
final[['id', 'avg']].rename({"avg": "feature_2"}, axis=1).to_csv("Final_op.csv", index=False)