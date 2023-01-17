!pip install tqdm

import os, datetime, importlib



os.environ['KMP_DUPLICATE_LIB_OK']='True'



from typing import Union

import sklearn

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from datetime import timedelta

from tqdm import tqdm

tqdm.pandas()



import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, LSTM, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate

from tensorflow.keras.optimizers import Adam



timestring = lambda : datetime.datetime.now().strftime("%H_%M_%S")



import m5_helpers

import m5_models

import nbeats



%matplotlib inline
path = "../input/m5-forecasting-accuracy"



train_df = pd.read_csv(f'{path}/sales_train_validation.csv')

calendar = pd.read_csv(f'{path}/calendar.csv')

prices = pd.read_csv(f'{path}/sell_prices.csv')



train_fold_df = train_df.iloc[:, :-28]

valid_fold_df = train_df.iloc[:, -28:]



e = m5_helpers.WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, prices)
# Use this is you want to fit on the full 42840 sequences

df_train = e.train_series

df_valid = e.valid_series

df = pd.concat([df_train, df_valid], axis=1)

df = df.reset_index().rename(columns={'index':'id'})
length = len(df.columns)-1

start_day = datetime.datetime.strptime('2011-01-29', '%Y-%m-%d')

date_list = [(start_day + datetime.timedelta(days=x)).date() for x in range(length)]

df.columns = [df.columns[0]]+date_list

data_start_date = df.columns[1]

data_end_date = df.columns[-1]



date_to_index = pd.Series(index=pd.Index([pd.to_datetime(c) for c in df.columns[1:]]), data=[i for i in range(len(df.columns[1:]))])



print('Data ranges from %s to %s' % (data_start_date, data_end_date))

df.head()
def plot_random_series(df, n_series):

    

    sample = df.sample(n_series, random_state=np.random.randint(100))

    page_labels = sample['id'].tolist()

    series_samples = sample.loc[:,data_start_date:data_end_date]

    

    plt.figure(figsize=(10,6))

    

    for i in range(series_samples.shape[0]):

        pd.Series(series_samples.iloc[i]).astype(np.float64).plot(linewidth=1.5)

    

    plt.title('Randomly Selected Time Series')

    plt.legend(page_labels)

    

plot_random_series(df, 6)
pred_steps = 28

pred_length = timedelta(pred_steps)



first_day = pd.to_datetime(data_start_date) 

last_day = pd.to_datetime(data_end_date)



test_pred_start = last_day - pred_length + timedelta(1)

test_pred_end = last_day



val_pred_start = test_pred_start - pred_length

val_pred_end = test_pred_start - timedelta(days=1)



train_pred_start = val_pred_start - pred_length

train_pred_end = val_pred_start - timedelta(days=1)



enc_length = train_pred_start - first_day



train_enc_start = first_day

train_enc_end = train_enc_start + enc_length - timedelta(1)



val_enc_start = train_enc_start + pred_length

val_enc_end = val_enc_start + enc_length - timedelta(1)



test_enc_start = val_enc_start + pred_length

test_enc_end = test_enc_start + enc_length - timedelta(1)



print('Train encoding:', train_enc_start, '-', train_enc_end)

print('Train prediction:', train_pred_start, '-', train_pred_end, '\n')

print('Val encoding:', val_enc_start, '-', val_enc_end)

print('Val prediction:', val_pred_start, '-', val_pred_end, '\n')

print('Test encoding:', test_enc_start, '-', test_enc_end)

print('Test prediction:', test_pred_start, '-', test_pred_end)



print('\nEncoding interval:', enc_length.days)

print('Prediction interval:', pred_length.days)
n_samples = None #Set this to a finite number to train on a reduced number of samples

batch_size = 64

epochs = 5

learning_rate = 1e-3



series_array = df[df.columns[1:]].values



encoder_input_data, encode_series_mean, decoder_input_data, decoder_target_data = m5_helpers.get_all_data(date_to_index, series_array, train_enc_start, train_enc_end, train_pred_start, train_pred_end, n_samples=n_samples)



callbacks = m5_helpers.make_callbacks('lstm', 'lstm_run_1')



model, dec_model, encoder_inputs, encoder_states = m5_models.create_enc_dec(learning_rate=learning_rate, hidden_size = 32, dropout = 0)



history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,

                     batch_size=batch_size,

                     epochs=epochs,

                     validation_split=0.2,

                     callbacks=callbacks)
history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,

                     batch_size=batch_size,

                     epochs=epochs,

                     validation_split=0.2,

                     callbacks=callbacks)
m5_helpers.make_training_plot(history)
encoder_model = Model(encoder_inputs, encoder_states)



encoder_input_data, encode_series_mean, decoder_input_data, decoder_target_data = m5_helpers.get_all_data(date_to_index, series_array, val_enc_start, val_enc_end, val_pred_start, val_pred_end, shuffle=False)



m5_helpers.predict_and_plot(encoder_model, encoder_input_data, decoder_target_data, encode_series_mean, sample_ind=0, enc_tail_len=250, lstm=True, dec_model=dec_model)
n_samples = None #Set this to a finite number to train on a reduced number of samples

batch_size = 256

epochs = 5

learning_rate = 3e-4



series_array = df[df.columns[1:]].values

# weights = e.weights[:first_n_samples][0]



encoder_input_data, encode_series_mean, decoder_input_data, decoder_target_data = m5_helpers.get_all_data(date_to_index, series_array, train_enc_start, train_enc_end, train_pred_start, train_pred_end, n_samples=n_samples)



# we append a lagged history of the target series to the input data, so that we can train with teacher forcing

lagged_target_history = decoder_target_data[:,:-1,:1]

encoder_input_data = np.concatenate([encoder_input_data, lagged_target_history], axis=1)



callbacks = m5_helpers.make_callbacks('wavenet_model', 'wavenet_1')

score_cb = m5_helpers.score_callback(e=e, ids=[x+'_validation' for x in df.id], val_encoder_input=encoder_input_data, val_encode_series_mean=encode_series_mean)

callbacks.insert(0, score_cb)



# Create a simple wavenet model

model = m5_models.create_simple_wave(learning_rate=learning_rate)



# Create a larger more complex wavenet model

# model = Model_Functions.create_full_wave(learning_rate=learning_rate)



print(model.summary())

history = model.fit(encoder_input_data, decoder_target_data,

                    batch_size=batch_size,

                    epochs=epochs,

                    validation_split=0.2,

                    callbacks=callbacks)
m5_helpers.make_training_plot(history)
encoder_input_data, encode_series_mean, decoder_input_data, decoder_target_data = m5_helpers.get_all_data(date_to_index, series_array, val_enc_start, val_enc_end, val_pred_start, val_pred_end, shuffle=False)



m5_helpers.predict_and_plot(model, encoder_input_data, decoder_target_data, encode_series_mean, 35000, enc_tail_len=250, lstm=False, dec_model=None)
batch_size = 1024

input_steps = 28*4

pred_steps=28



val_frac=0.1

series_array = sklearn.utils.shuffle(df[df.columns[1:]].values, random_state=42) # shuffle once up front so don't have to waste time later

weights = sklearn.utils.shuffle(e.weights[0], random_state=42)

train_samples = int((1-val_frac)*series_array.shape[0])-1

val_samples = int(val_frac**series_array.shape[0])



first_valid_day = first_day + timedelta(days=input_steps)

valid_train_end_days = [first_valid_day + timedelta(days=x) for x in range((train_enc_end-first_valid_day).days + 1)]



def data_generator():

    random_start_id = np.random.choice(train_samples-batch_size-val_samples, 1)[0]

    random_train_enc_end_day = np.random.choice(valid_train_end_days, 1)[0]

    random_train_enc_start_day = random_train_enc_end_day - timedelta(days=input_steps-1)

    

    random_train_pred_start_day = random_train_enc_end_day + timedelta(days=1)

    random_train_pred_end_day = random_train_pred_start_day + timedelta(days=pred_steps) - timedelta(1)



    encoder_input_data, encode_series_mean, decoder_input_data, decoder_target_data = m5_helpers.get_all_data(date_to_index, series_array, 

                                                                                                                    random_train_enc_start_day, random_train_enc_end_day, 

                                                                                                                    random_train_pred_start_day, random_train_pred_end_day, 

                                                                                                                    shuffle=True, n_samples=train_samples)

    yield (encoder_input_data[random_start_id:random_start_id+batch_size,:,:], 

    decoder_target_data[random_start_id:random_start_id+batch_size,:,:],

    weights[random_start_id:random_start_id+batch_size])



series_array = df[df.columns[1:]].values

weights = e.weights[0]



encoder_input_data, encode_series_mean, decoder_input_data, decoder_target_data = m5_helpers.get_all_data(date_to_index, series_array, val_pred_start - timedelta(days=input_steps), val_pred_start - timedelta(days=1), val_pred_start, val_pred_end, shuffle=False)

validation_data = (encoder_input_data, decoder_target_data, weights)

validation_dataset = tf.data.Dataset.from_tensor_slices(validation_data).batch(batch_size)
epochs = 5

learning_rate=3e-4



callbacks = m5_helpers.make_callbacks('nbeats_model', 'nbeats_run_1')



# val_encoder_input = pd.concat([df.id, pd.DataFrame(encoder_input_data.reshape(-1,28))], axis=1, ignore_index=True)



score_cb = m5_helpers.score_callback(e=e, ids=[x+'_validation' for x in df.id], val_encoder_input=encoder_input_data, val_encode_series_mean=encode_series_mean)

callbacks.insert(0, score_cb)



shapes = ((batch_size, input_steps, 1),(batch_size, pred_steps, 1),(batch_size,))

dataset = tf.data.Dataset.from_generator(data_generator, (tf.float32, tf.float32, tf.float32), output_shapes=shapes).repeat()



model = nbeats.NBeatsNet(backcast_length=input_steps, forecast_length=pred_steps,

                  stack_types=(nbeats.NBeatsNet.GENERIC_BLOCK, nbeats.NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2,

                  thetas_dim=(4, 4), share_weights_in_stack=False, hidden_layer_units=8)



model.compile(Adam(learning_rate=learning_rate), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

history = model.fit(dataset, steps_per_epoch=200, validation_data=validation_dataset, callbacks=callbacks, epochs=epochs)
m5_helpers.make_training_plot(history)
encoder_input_data, encode_series_mean, decoder_input_data, decoder_target_data = m5_helpers.get_all_data(date_to_index, series_array, val_pred_start - timedelta(days=input_steps), val_pred_start - timedelta(days=1), val_pred_start, val_pred_end, shuffle=False)



m5_helpers.predict_and_plot(model, encoder_input_data, decoder_target_data, encode_series_mean, sample_ind=1000, enc_tail_len=input_steps, lstm=False, dec_model=None, nbeats=True)
predict_batch_size = 1024

series_array = pd.read_csv(f'{path}/sales_train_validation.csv').iloc[:,6:].values

batches = series_array.shape[0]//predict_batch_size+1

preds = []



encoder_input_data, encode_series_mean, decoder_input_data, decoder_target_data = m5_helpers.get_all_data(date_to_index, series_array, val_pred_start - timedelta(days=input_steps), val_pred_start - timedelta(days=1), val_pred_start, val_pred_end, shuffle=False)

predictions = model.predict(encoder_input_data)
predictions = m5_helpers.untransform_series_decode(predictions, encode_series_mean)

predictions = pd.DataFrame(predictions.reshape(-1,28))

predictions['id']=train_df.id

predictions = predictions[['id']+list(predictions.columns[:-1])]

predictions.columns = ['id'] + ['d_' + str(x+1886) for x in np.arange(28)]



# Score on val set

print('Score: ' + str(e.score(predictions.iloc[:,1:])))
# Look at which time series are contributing to this score

print(e.contributors)
# For submission

# submitte_predictions = predictions.iloc[:,1:]

# submitte_predictions.columns = ['F'+str(x+1) for x in np.arange(28)]



# Make submissions file

sample_submission = pd.DataFrame(pd.read_csv(f'{path}/sample_submission.csv').id)

sample_submission = sample_submission.merge(predictions, on='id', how='left')

sample_submission.fillna(0, inplace=True)

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()