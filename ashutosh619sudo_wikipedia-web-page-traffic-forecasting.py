# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/web-traffic-time-series-forecasting/train_1.csv.zip')

df.head()
df.info()
start_date = df.columns[1]

last_date = df.columns[-1]

print(f"Date range: {start_date} to {last_date}")
def plot_graph(df,n_series):

    sample = df.sample(n_series,random_state=42)

    page = sample["Page"].to_list()

    series_sample = sample.loc[:,start_date:last_date]

    plt.figure(figsize=(15,10))

    

    

    for i in range(series_sample.shape[0]):

        np.log1p(pd.Series(series_sample.iloc[i]).astype(np.float64)).plot(linewidth=1.5)

    

    plt.title("Time vs Views on random website")

    plt.legend(page)

plot_graph(df,6)    
from datetime import timedelta



pred_steps = 14

pred_length=timedelta(pred_steps)



first_day = pd.to_datetime(start_date) 

last_day = pd.to_datetime(last_date)



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

from keras.layers import Input, Conv1D, Dense, Dropout, Lambda, concatenate

from keras.optimizers import Adam



# convolutional layer parameters

n_filters = 32 

filter_width = 2

dilation_rates = [2**i for i in range(8)] 



# define an input history series and pass it through a stack of dilated causal convolutions. 

history_seq = Input(shape=(None, 1))

x = history_seq



for dilation_rate in dilation_rates:

    x = Conv1D(filters=n_filters,

               kernel_size=filter_width, 

               padding='causal',

               dilation_rate=dilation_rate)(x)



x = Dense(128, activation='relu')(x)

x = Dropout(.2)(x)

x = Dense(1)(x)



# extract the last 14 time steps as the training target

def slice(x, seq_length):

    return x[:,-seq_length:,:]



pred_seq_train = Lambda(slice, arguments={'seq_length':14})(x)



model = Model(history_seq, pred_seq_train)
model.summary()
first_n_samples = 40000

batch_size = 2**11

epochs = 10



# sample of series from train_enc_start to train_enc_end  

encoder_input_data = get_time_block_series(series_array, date_to_index, 

                                           train_enc_start, train_enc_end)[:first_n_samples]

encoder_input_data, encode_series_mean = transform_series_encode(encoder_input_data)



# sample of series from train_pred_start to train_pred_end 

decoder_target_data = get_time_block_series(series_array, date_to_index, 

                                            train_pred_start, train_pred_end)[:first_n_samples]

decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean)



# we append a lagged history of the target series to the input data, 

# so that we can train with teacher forcing

lagged_target_history = decoder_target_data[:,:-1,:1]

encoder_input_data = np.concatenate([encoder_input_data, lagged_target_history], axis=1)



model.compile(Adam(), loss='mean_absolute_error')

history = model.fit(encoder_input_data, decoder_target_data,

                    batch_size=batch_size,

                    epochs=epochs,

                    validation_split=0.2)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])



plt.xlabel('Epoch')

plt.ylabel('Mean Absolute Error Loss')

plt.title('Loss Over Time')

plt.legend(['Train','Valid'])


def predict_sequence(input_sequence):



    history_sequence = input_sequence.copy()

    pred_sequence = np.zeros((1,pred_steps,1)) # initialize output (pred_steps time steps)  

    

    for i in range(pred_steps):

        

        # record next time step prediction (last time step of model output) 

        last_step_pred = model.predict(history_sequence)[0,-1,0]

        pred_sequence[0,i,0] = last_step_pred

        

        # add the next time step prediction to the history sequence

        history_sequence = np.concatenate([history_sequence, 

                                           last_step_pred.reshape(-1,1,1)], axis=1)



    return pred_sequence
encoder_input_data = get_time_block_series(series_array, date_to_index, val_enc_start, val_enc_end)

encoder_input_data, encode_series_mean = transform_series_encode(encoder_input_data)



decoder_target_data = get_time_block_series(series_array, date_to_index, val_pred_start, val_pred_end)

decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean)

def predict_and_plot(encoder_input_data, decoder_target_data, sample_ind, enc_tail_len=50):



    encode_series = encoder_input_data[sample_ind:sample_ind+1,:,:] 

    pred_series = predict_sequence(encode_series)

    

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
#better than LSTM

predict_and_plot(encoder_input_data, decoder_target_data, 100)
predict_and_plot(encoder_input_data, decoder_target_data, 6007)