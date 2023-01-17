import pandas as pd
import numpy as np
import keras
from google.cloud import bigquery
query = '''
#standardSQL
SELECT
  id,
  title,
  REGEXP_REPLACE(NET.HOST(url), 'www.', '') AS domain,
  FORMAT_TIMESTAMP("%Y-%m-%d %H:%M:%S", timestamp, "America/New_York") AS created_at,
  score,
  TIMESTAMP_DIFF(LEAD(timestamp, 30) OVER (ORDER BY timestamp), timestamp, SECOND) as time_on_new
FROM
  `bigquery-public-data.hacker_news.full`
WHERE
  DATETIME(timestamp, "America/New_York") BETWEEN '2017-01-01 00:00:00' AND '2018-08-01 00:00:00'
  AND type = "story"
  AND url != ''
  AND deleted IS NULL
  AND dead IS NULL
ORDER BY
  created_at DESC
'''

client = bigquery.Client()

query_job = client.query(query)

iterator = query_job.result(timeout=30)
rows = list(iterator)

df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

df = df.sample(frac=1, random_state=123).dropna().reset_index(drop=True)
df.head(10)
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence, Tokenizer

num_words = 20000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(df['title'].values)
maxlen = 15

titles = tokenizer.texts_to_sequences(df['title'].values)
titles = sequence.pad_sequences(titles, maxlen=maxlen)
print(titles[0:5,])
num_domains = 100

domain_counts = df['domain'].value_counts()[0:num_domains]

print(domain_counts)
from sklearn.preprocessing import LabelBinarizer

top_domains = np.array(domain_counts.index, dtype=object)

domain_encoder = LabelBinarizer()
domain_encoder.fit(top_domains)

domains = domain_encoder.transform(df['domain'].values.astype(str))
domains[0]
from keras.utils import to_categorical

dayofweeks = to_categorical(pd.to_datetime(df['created_at']).dt.dayofweek)
hours = to_categorical(pd.to_datetime(df['created_at']).dt.hour)

print(dayofweeks[0:5])
print(hours[0:5])
weights = np.where(df['score'].values == 1, 0.5, 1.0)
print(weights[0:5])
from sklearn.preprocessing import MinMaxScaler

trend_encoder = MinMaxScaler()
trends = trend_encoder.fit_transform(pd.to_datetime(df['created_at']).values.reshape(-1, 1))
trends[0:5]
newtime_encoder = MinMaxScaler()
newtimes = trend_encoder.fit_transform(df['time_on_new'].values.reshape(-1, 1))
newtimes[0:5]
from keras import backend as K

def r_2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
def hybrid_loss(y_true, y_pred):
    weight_mae = 0.1
    weight_msle = 1.
    weight_poisson = 0.1
    
    mae_loss = weight_mae * K.mean(K.abs(y_pred - y_true), axis=-1)
    
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    msle_loss = weight_msle * K.mean(K.square(first_log - second_log), axis=-1)
    
    poisson_loss = weight_poisson * K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)
    return mae_loss + msle_loss + poisson_loss
from keras.models import Input, Model
from keras.layers import Dense, Embedding, CuDNNGRU, CuDNNLSTM, LSTM, concatenate, Activation, BatchNormalization
from keras.layers.core import Masking, Dropout, Reshape, SpatialDropout1D
from keras.regularizers import l1, l2

input_titles = Input(shape=(maxlen,), name='input_titles')
input_domains = Input(shape=(num_domains,), name='input_domains')
input_dayofweeks = Input(shape=(7,), name='input_dayofweeks')
input_hours = Input(shape=(24,), name='input_hours')
# input_trend = Input(shape=(1,), name='input_trend')
# input_newtime = Input(shape=(1,), name='input_newtime')

embedding_titles = Embedding(num_words + 1, 50, name='embedding_titles', mask_zero=False)(input_titles)
spatial_dropout = SpatialDropout1D(0.2, name='spatial_dropout')(embedding_titles)
rnn_titles = CuDNNLSTM(128, name='rnn_titles')(spatial_dropout)

concat = concatenate([rnn_titles, input_domains, input_dayofweeks, input_hours], name='concat')

num_hidden_layers = 3

hidden = Dense(128, activation='relu', name='hidden_1', kernel_regularizer=l2(1e-2))(concat)
hidden = BatchNormalization(name="bn_1")(hidden)
hidden = Dropout(0.5, name="dropout_1")(hidden)

for i in range(num_hidden_layers-1):
    hidden = Dense(256, activation='relu', name='hidden_{}'.format(i+2), kernel_regularizer=l2(1e-2))(hidden)
    hidden = BatchNormalization(name="bn_{}".format(i+2))(hidden)
    hidden = Dropout(0.5, name="dropout_{}".format(i+2))(hidden)
    
output = Dense(1, activation='relu', name='output', kernel_regularizer=l2(1e-2))(hidden)

model = Model(inputs=[input_titles,
                      input_domains,
                      input_dayofweeks,
                      input_hours],
                      outputs=[output])

model.compile(loss=hybrid_loss,
              optimizer='adam',
              metrics=['mse', 'mae', r_2])

model.summary()
from keras.callbacks import LearningRateScheduler, Callback

base_lr = 1e-3
num_epochs = 25
split_prop = 0.2

def lr_linear_decay(epoch):
            return (base_lr * (1 - (epoch / num_epochs)))
    
model.fit([titles, domains, dayofweeks, hours], [df['score'].values],
          batch_size=1024,
          epochs=num_epochs,
          validation_split=split_prop,
          callbacks=[LearningRateScheduler(lr_linear_decay)],
          sample_weight=weights)
val_size = int(split_prop * df.shape[0])

predictions = model.predict([titles[-val_size:],
                             domains[-val_size:],
                             dayofweeks[-val_size:],
                             hours[-val_size:]])[:, 0]

predictions
df_preds = pd.concat([pd.Series(df['title'].values[-val_size:]),
                      pd.Series(df['score'].values[-val_size:]),
                      pd.Series(predictions)],
                     axis=1)
df_preds.columns = ['title', 'actual', 'predicted']
# df_preds.to_csv('hn_val.csv', index=False)
df_preds.head(50)
train_size = int((1-split_prop) * df.shape[0])

predictions = model.predict([titles[:train_size],
                             domains[:train_size],
                             dayofweeks[:train_size],
                             hours[:train_size]])[:, 0]

df_preds = pd.concat([pd.Series(df['title'].values[:train_size]),
                      pd.Series(df['score'].values[:train_size]),
                      pd.Series(predictions)],
                     axis=1)
df_preds.columns = ['title', 'actual', 'predicted']
# df_preds.to_csv('hn_train.csv', index=False)
df_preds.head(50)