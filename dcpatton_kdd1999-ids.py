import tensorflow as tf

import pandas as pd

import random



seed = 52

random.seed(seed)

tf.random.set_seed(seed)



tf.__version__
data_df = pd.read_csv('/kaggle/input/kdd-cup-1999-data/kddcup.data_10_percent_corrected', header=None)
data_df.columns = [

    'duration',

    'protocol_type',

    'service',

    'flag',

    'src_bytes',

    'dst_bytes',

    'land',

    'wrong_fragment',

    'urgent',

    'hot',

    'num_failed_logins',

    'logged_in',

    'num_compromised',

    'root_shell',

    'su_attempted',

    'num_root',

    'num_file_creations',

    'num_shells',

    'num_access_files',

    'num_outbound_cmds',

    'is_host_login',

    'is_guest_login',

    'count',

    'srv_count',

    'serror_rate',

    'srv_serror_rate',

    'rerror_rate',

    'srv_rerror_rate',

    'same_srv_rate',

    'diff_srv_rate',

    'srv_diff_host_rate',

    'dst_host_count',

    'dst_host_srv_count',

    'dst_host_same_srv_rate',

    'dst_host_diff_srv_rate',

    'dst_host_same_src_port_rate',

    'dst_host_srv_diff_host_rate',

    'dst_host_serror_rate',

    'dst_host_srv_serror_rate',

    'dst_host_rerror_rate',

    'dst_host_srv_rerror_rate',

    'outcome'

]
data_df.sample(10)
data_df.outcome.value_counts()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

data_df[['outcome']] = label_encoder.fit_transform(data_df[['outcome']])
data_df.info()
assert 1 == len(data_df.num_outbound_cmds.unique())  # only one unique value, so drop it

data_df.drop('num_outbound_cmds', axis='columns', inplace=True)
data_df.isna().sum()
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=seed, stratify=data_df['outcome'])

print(train_df.shape)

print(test_df.shape)
train_df.outcome.value_counts()
test_df.outcome.value_counts()
def df_to_dataset(dataframe, shuffle=True, batch_size=32):

  dataframe = dataframe.copy()

  labels = dataframe.pop('outcome')

  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))

  if shuffle:

    ds = ds.shuffle(buffer_size=1024)

  ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

  return ds
for column in data_df.columns:

  print(column + ': ' + str(data_df[column].nunique()))
from tensorflow import feature_column



feature_columns = []



# numeric cols

for column in ['duration','src_bytes','dst_bytes','wrong_fragment','urgent','hot',

               'num_failed_logins','num_compromised','num_root','num_file_creations',

               'num_shells','num_access_files','count','srv_count','serror_rate',

               'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',

               'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',

               'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',

               'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',

               'dst_host_rerror_rate','dst_host_srv_rerror_rate']:

  feature_columns.append(feature_column.numeric_column(column))



# indicator_columns

indicator_column_names = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 

                          'root_shell', 'su_attempted', 'is_host_login', 'is_guest_login']

for col_name in indicator_column_names:

  categorical_column = feature_column.categorical_column_with_vocabulary_list(

      col_name, data_df[col_name].unique())

  indicator_column = feature_column.indicator_column(categorical_column)

  feature_columns.append(indicator_column)
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
batch_size = 128

train_ds = df_to_dataset(train_df, batch_size=batch_size)

test_ds = df_to_dataset(test_df, shuffle=False, batch_size=batch_size)
from tensorflow.keras.layers import Dense



def create_model():

  tf.keras.backend.clear_session()

  model = tf.keras.Sequential([

    feature_layer,

    Dense(256, activation='relu'),

    Dense(128, activation='relu'),

    Dense(64, activation='relu'),

    Dense(23, activation='softmax')

  ])



  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

  return model



model = create_model()
filepath = 'model.h5'



mc = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, 

                                        save_weights_only=True, mode='auto')



es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')



history = model.fit(train_ds, epochs=200, validation_data=test_ds, callbacks=[mc, es])
model.load_weights('model.h5')

model.evaluate(test_ds)
import numpy as np

y_preds = model.predict(test_ds, verbose=1)

print(y_preds.shape)

y_preds = np.argmax(y_preds, axis=1)

from sklearn.metrics import classification_report

y_true = test_df['outcome']

print(classification_report(y_true, y_preds))
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced', np.unique(train_df['outcome']), train_df['outcome'])

class_weights
class_keys = np.unique(train_df['outcome'])

class_keys
class_weight_dict = dict(zip(class_keys,class_weights))
model = create_model()
filepath = 'model.h5'



mc = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, 

                                        save_weights_only=True, mode='auto')



es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')



history = model.fit(train_ds, epochs=200, validation_data=test_ds, callbacks=[mc, es], class_weight=class_weight_dict)
model.load_weights('model.h5')

model.evaluate(test_ds)
y_preds = model.predict(test_ds, verbose=1)

y_preds = np.argmax(y_preds, axis=1)

y_true = test_df['outcome']

print(classification_report(y_true, y_preds))
label_encoder.classes_[11]
train_df.loc[(train_df.outcome != 11),'outcome'] = 1

train_df.loc[(train_df.outcome == 11),'outcome'] = 0

test_df.loc[(test_df.outcome != 11),'outcome'] = 1

test_df.loc[(test_df.outcome == 11),'outcome'] = 0
train_df.outcome.value_counts()
train_ds = df_to_dataset(train_df, batch_size=batch_size)

test_ds = df_to_dataset(test_df, shuffle=False, batch_size=batch_size)
tf.keras.backend.clear_session()

model = tf.keras.Sequential([

  feature_layer,

  Dense(256, activation='relu'),

  Dense(128, activation='relu'),

  Dense(64, activation='relu'),

  Dense(1, activation='sigmoid')

])



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
filepath = 'model.h5'



mc = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, 

                                        save_weights_only=True, mode='auto')



es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')



history = model.fit(train_ds, epochs=200, validation_data=test_ds, callbacks=[mc, es])
model.load_weights('model.h5')

model.evaluate(test_ds)
y_preds = model.predict(test_ds, verbose=1)

y_preds

y_preds[y_preds > 0.5] = 1

y_preds[y_preds <= 0.5] = 0

y_true = test_df['outcome']

print(classification_report(y_true, y_preds))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_true, y_preds))