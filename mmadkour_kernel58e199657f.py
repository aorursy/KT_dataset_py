!ls 
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set()
import pandas as pd

import zipfile

zf = zipfile.ZipFile('../input/web-traffic-time-series-forecasting/train_1.csv.zip') # having First.csv zipped file.

df = pd.read_csv(zf.open('train_1.csv'))

df.describe()
df.shape
df.head(10)
df.info()




data_start_date = df.columns[1]

data_end_date = df.columns[-1]

print('Data ranges from %s to %s' % (data_start_date, data_end_date))



def plot_random_series(df, n_series):

    

    sample = df.sample(n_series, random_state=8)

    page_labels = sample['Page'].tolist()

    series_samples = sample.loc[:,data_start_date:data_end_date]

    

    plt.figure(figsize=(10,6))

    

    for i in range(series_samples.shape[0]):

        np.log1p(pd.Series(series_samples.iloc[i]).astype(np.float64)).plot(linewidth=1.5)

        #pd.Series(series_samples.iloc[i]).plot(linewidth=1.5)

    

    plt.title('Randomly Selected Wikipedia Page Daily Views Over Time (Log(views) + 1)')

    plt.legend(page_labels)

    

plot_random_series(df, 6)
from datetime import timedelta



pred_steps = 14

pred_length=timedelta(pred_steps)



first_day = pd.to_datetime(data_start_date) 

last_day = pd.to_datetime(data_end_date)



val_pred_start = last_day - pred_length + timedelta(1)

val_pred_end = last_day



train_pred_start = val_pred_start - pred_length

train_pred_end = val_pred_start - timedelta(days=1)
enc_length = train_pred_start - first_day



train_enc_start = first_day

train_enc_end = train_enc_start + enc_length - timedelta(1)



val_enc_start = train_enc_start + pred_length

val_enc_end = val_enc_start + enc_length - timedelta(1)
print('Train encoding:', train_enc_start, '-', train_enc_end)

print('Train prediction:', train_pred_start, '-', train_pred_end, '\n')

print('Val encoding:', val_enc_start, '-', val_enc_end)

print('Val prediction:', val_pred_start, '-', val_pred_end)



print('\nEncoding interval:', enc_length.days)

print('Prediction interval:', pred_length.days)
date_to_index = pd.Series(index=pd.Index([pd.to_datetime(c) for c in df.columns[1:]]),

                          data=[i for i in range(len(df.columns[1:]))])



series_array = df[df.columns[1:]].values



def get_time_block_series(series_array, date_to_index, start_date, end_date):

    

    inds = date_to_index[start_date:end_date]

    return series_array[:,inds]



def transform_series_encode(series_array):

    

    series_array = np.log1p(np.nan_to_num(series_array)) # filling NaN with 0

    series_mean = series_array.mean(axis=1).reshape(-1,1) 

    series_array = series_array - series_mean

    series_array = series_array.reshape((series_array.shape[0],series_array.shape[1], 1))

    

    return series_array, series_mean



def transform_series_decode(series_array, encode_series_mean):

    

    series_array = np.log1p(np.nan_to_num(series_array)) # filling NaN with 0

    series_array = series_array - encode_series_mean

    series_array = series_array.reshape((series_array.shape[0],series_array.shape[1], 1))

    

    return series_array
from keras.models import Model

from keras.layers import Input, LSTM, Dense

from keras.optimizers import Adam



latent_dim = 50 # LSTM hidden units

dropout = .20 



# Define an input series and encode it with an LSTM. 

encoder_inputs = Input(shape=(None, 1)) 

encoder = LSTM(latent_dim, dropout=dropout, return_state=True)

encoder_outputs, state_h, state_c = encoder(encoder_inputs)



# We discard `encoder_outputs` and only keep the final states. These represent the "context"

# vector that we use as the basis for decoding.

encoder_states = [state_h, state_c]



# Set up the decoder, using `encoder_states` as initial state.

# This is where teacher forcing inputs are fed in.

decoder_inputs = Input(shape=(None, 1)) 



# We set up our decoder using `encoder_states` as initial state.  

# We return full output sequences and return internal states as well. 

# We don't use the return states in the training model, but we will use them in inference.

decoder_lstm = LSTM(latent_dim, dropout=dropout, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(decoder_inputs,

                                     initial_state=encoder_states)



decoder_dense = Dense(1) # 1 continuous output at each timestep

decoder_outputs = decoder_dense(decoder_outputs)



# Define the model that will turn

# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)




model.summary()



first_n_samples = 20000

batch_size = 2**11

epochs = 100



# sample of series from train_enc_start to train_enc_end  

encoder_input_data = get_time_block_series(series_array, date_to_index, 

                                           train_enc_start, train_enc_end)[:first_n_samples]

encoder_input_data, encode_series_mean = transform_series_encode(encoder_input_data)



# sample of series from train_pred_start to train_pred_end 

decoder_target_data = get_time_block_series(series_array, date_to_index, 

                                            train_pred_start, train_pred_end)[:first_n_samples]

decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean)



# lagged target series for teacher forcing

decoder_input_data = np.zeros(decoder_target_data.shape)

decoder_input_data[:,1:,0] = decoder_target_data[:,:-1,0]

decoder_input_data[:,0,0] = encoder_input_data[:,-1,0]



model.compile(Adam(), loss='mean_absolute_error')

history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,

                     batch_size=batch_size,

                     epochs=epochs,

                     validation_split=0.2)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Import Data

#df = pd.read_csv('https://github.com/selva86/datasets/raw/master/AirPassengers.csv')

# Draw Plot

ll=df.iloc[133].tolist()

ll.pop(0)

ll



fig, (ax1) = plt.subplots(1, 1,figsize=(16,6), dpi= 80)

plot_acf(ll, ax=ax1, lags=400)

#plot_pacf(ll, ax=ax2, lags=400)

# Decorate

# lighten the borders

ax1.spines["top"].set_alpha(.3); ax2.spines["top"].set_alpha(.3)

ax1.spines["bottom"].set_alpha(.3); ax2.spines["bottom"].set_alpha(.3)

ax1.spines["right"].set_alpha(.3); ax2.spines["right"].set_alpha(.3)

ax1.spines["left"].set_alpha(.3); ax2.spines["left"].set_alpha(.3)

# font size of tick labels

ax1.tick_params(axis='both', labelsize=12)

plt.show()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])



plt.xlabel('Epoch')

plt.ylabel('Mean Absolute Error Loss')

plt.title('Loss Over Time')

plt.legend(['Train','Valid'])
# from our previous model - mapping encoder sequence to state vectors

encoder_model = Model(encoder_inputs, encoder_states)



# A modified version of the decoding stage that takes in predicted target inputs

# and encoded state vectors, returning predicted target outputs and decoder state vectors.

# We need to hang onto these state vectors to run the next step of the inference loop.

decoder_state_input_h = Input(shape=(latent_dim,))

decoder_state_input_c = Input(shape=(latent_dim,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]



decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]



decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs,

                      [decoder_outputs] + decoder_states)



def decode_sequence(input_seq):

    

    # Encode the input as state vectors.

    states_value = encoder_model.predict(input_seq)



    # Generate empty target sequence of length 1.

    target_seq = np.zeros((1, 1, 1))

    

    # Populate the first target sequence with end of encoding series pageviews

    target_seq[0, 0, 0] = input_seq[0, -1, 0]



    # Sampling loop for a batch of sequences - we will fill decoded_seq with predictions

    # (to simplify, here we assume a batch of size 1).



    decoded_seq = np.zeros((1,pred_steps,1))

    

    for i in range(pred_steps):

        

        output, h, c = decoder_model.predict([target_seq] + states_value)

        

        decoded_seq[0,i,0] = output[0,0,0]



        # Update the target sequence (of length 1).

        target_seq = np.zeros((1, 1, 1))

        target_seq[0, 0, 0] = output[0,0,0]



        # Update states

        states_value = [h, c]



    return decoded_seq
encoder_input_data = get_time_block_series(series_array, date_to_index, val_enc_start, val_enc_end)

encoder_input_data, encode_series_mean = transform_series_encode(encoder_input_data)



decoder_target_data = get_time_block_series(series_array, date_to_index, val_pred_start, val_pred_end)

decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean)
def predict_and_plot(encoder_input_data, decoder_target_data, sample_ind, enc_tail_len=50):



    encode_series = encoder_input_data[sample_ind:sample_ind+1,:,:] 

    pred_series = decode_sequence(encode_series)

    

    encode_series = encode_series.reshape(-1,1)

    pred_series = pred_series.reshape(-1,1)   

    target_series = decoder_target_data[sample_ind,:,:1].reshape(-1,1) 

    

    encode_series_tail = np.concatenate([encode_series[-enc_tail_len:],target_series[:1]])

    x_encode = encode_series_tail.shape[0]

    

    plt.figure(figsize=(10,6))   

    

    plt.plot(range(1,x_encode+1),encode_series_tail)

    plt.plot(range(x_encode,x_encode+pred_steps),target_series,color='orange')

    plt.plot(range(x_encode,x_encode+pred_steps),pred_series,color='teal',linestyle='--')

    

    plt.title('Encoder Series Tail of Length %d, Target Series, and Predictions' % enc_tail_len)

    plt.legend(['Encoding Series','Target Series','Predictions'])
predict_and_plot(encoder_input_data, decoder_target_data, 300)