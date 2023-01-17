import numpy as np
import pandas as pd
from datetime import datetime
import datetime

from keras.layers import Input, Dense, Activation, Reshape
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.optimizers import Adam
from sklearn import preprocessing

import re
import matplotlib.pyplot as plt
%matplotlib inline

from keras.layers import BatchNormalization,Dropout
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from keras.callbacks import ModelCheckpoint

from fastai.imports import *
from fastai.column_data import *
from fastai.structured import *

from keras.callbacks import ReduceLROnPlateau
from pandas.api.types import is_string_dtype, is_numeric_dtype
import statsmodels.api as sm

from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
PATH='../input/'

# from pylab import rcParams
# rcParams['figure.figsize'] = 11, 9

# decomposition = sm.tsa.seasonal_decompose(y_time, model='additive')
# fig = decomposition.plot()
# plt.show()
#"Is_quarter_end", "Is_quarter_start"
#"Is_month_end", "Is_month_start", "Is_year_end", "Is_year_start"

cat_feature = ["store", "item"]
con_feature = ["Year", "Month", "Week", "Day", "Dayofweek", "Dayofyear"]
def prepare_df(data_df, isTrain=True, shuffle=True):
    add_datepart(data_df, "date")
    if shuffle:
        data_df = data_df.sample(frac=1)

    for cat_f in cat_feature:
        data_df[cat_f] = data_df[cat_f].astype("category").cat.as_ordered()

    mapper = DataFrameMapper([
         (con_feature, StandardScaler())
    ])
    data_df[con_feature] = mapper.fit_transform(data_df)

    label_encoders = []
    for f_name in cat_feature:
        le = LabelEncoder()
        le.fit(data_df[f_name])
        label_encoders.append(le)
        data_df[f_name] = le.transform(data_df[f_name])

    sales_scaler = None
    if isTrain:
        sales_scaler = StandardScaler()
        sales_values = data_df.sales.values.reshape(-1,1)
        scaled = sales_scaler.fit_transform(sales_values)
        data_df.sales = scaled
    
    return data_df, sales_scaler, label_encoders

train_df = pd.read_csv(f'{PATH}train.csv',parse_dates=['date'])

train_df, scaler, label_encoders = prepare_df(train_df)
train_df.head()
# time_df = train_df.copy()
# time_df["date"] = time_df["date"].values.astype('datetime64')
# time_idx = time_df.set_index("date")

# y_time = time_idx['sales'].resample('MS').mean()

# y_time.plot(figsize=(15, 6))
# plt.show()


def data_for_model(data_df):
    x_fit = []

    for cat in cat_feature:
        x_fit.append(data_df[cat].values)

    for con in con_feature:
        x_fit.append(data_df[con].values)
        
    return x_fit

train_validation_ratio = 0.9
train_size = int(train_validation_ratio * train_df.shape[0])

x_train_df = train_df[:train_size]
x_val_df = train_df[train_size:]
y_train, y_val = train_df.sales[:train_size].values, train_df.sales[train_size:].values

x_fit_train = data_for_model(x_train_df)
x_fit_val = data_for_model(x_val_df)
emb_space = [(len(le.classes_), min(25, len(le.classes_)) // 2 ) for idx, le in enumerate(label_encoders)]
emb_space
model_inputs = []
model_embeddings = []
    
for input_dim, output_dim in emb_space:
    i = Input(shape=(1,))
    emb = Embedding(input_dim=input_dim, output_dim=output_dim)(i)
    
    model_inputs.append(i)
    model_embeddings.append(emb)
    
    
con_outputs = []
for con in con_feature:
    elaps_input = Input(shape=(1,))
    elaps_output = Dense(10)(elaps_input) 
    #elaps_output = BatchNormalization()(elaps_output)
    elaps_output = Activation("relu")(elaps_output)
    
    elaps_output = Reshape(target_shape=(1,10))(elaps_output)

    model_inputs.append(elaps_input)
    con_outputs.append(elaps_output)

merge_embeddings = concatenate(model_embeddings, axis=-1)
if len(con_outputs) > 1:
    merge_con_output = concatenate(con_outputs)
else:
    merge_con_output = con_outputs[0]

merge_embedding_cont = concatenate([merge_embeddings, merge_con_output])
merge_embedding_cont

output_tensor = Dense(1000, name="dense1024")(merge_embedding_cont)
output_tensor = BatchNormalization()(output_tensor)
output_tensor = Activation('relu')(output_tensor)
#output_tensor = Dropout(0.5)(output_tensor)

output_tensor = Dense(500, name="dense512")(output_tensor)
output_tensor = BatchNormalization()(output_tensor)
output_tensor = Activation("relu")(output_tensor)
#output_tensor = Dropout(0.5)(output_tensor)

output_tensor = Dense(1, activation='linear', name="output")(output_tensor)

optimizer = Adam(lr=10e-3)

nn_model = Model(inputs=model_inputs, outputs=output_tensor)
nn_model.compile(loss="mean_absolute_error", optimizer=optimizer, metrics=['mape'])


reduceLr=ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, verbose=1)
checkpoint = ModelCheckpoint("nn_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')#val_mean_absolute_percentage_error
callbacks_list = [checkpoint, reduceLr]

history = nn_model.fit(x=x_fit_train, y=y_train.reshape(-1,1,1),
                       validation_data=(x_fit_val, y_val.reshape(-1,1,1)),
                       batch_size=1024, epochs=20, callbacks=callbacks_list)
from keras.models import load_model
tt_model = load_model('nn_model.hdf5')
test_df = pd.read_csv(f'{PATH}test.csv',parse_dates=['date']).drop("id", axis=1)
test_df, _, _ = prepare_df(test_df, isTrain=False, shuffle=False)
x_fit_test = data_for_model(test_df)

scaled_preds = tt_model.predict(x=x_fit_test)

scaled_predictions = tt_model.predict(x=x_fit_test)
y_predictions = scaler.inverse_transform(scaled_preds)
y_predictions = y_predictions.reshape(-1)
submission_df = pd.DataFrame()
submission_df["id"] = pd.read_csv(f'{PATH}test.csv',parse_dates=['date'])["id"]
submission_df["sales"] = y_predictions
submission_df.to_csv('submission.csv',index=False)

submission_df.head()
# from IPython.display import FileLink
# FileLink('submission.csv')


# figure = plt.figure(figsize=(12, 10))
# grid = plt.GridSpec(12, 12, wspace=4.5, hspace=0.1)

# loss_plot = figure.add_subplot(grid[:5, :6])
# mse_plot = figure.add_subplot(grid[:5, 6:])

# loss_plot.plot(history.history['loss'])
# loss_plot.plot(history.history['val_loss'])
# loss_plot.set_xlabel('epoch')
# loss_plot.set_ylabel('loss')
# loss_plot.legend(['Train Loss', 'Validation Loss'], loc='best')

# mse_plot.plot(history.history['mean_absolute_percentage_error'])
# mse_plot.plot(history.history['val_mean_absolute_percentage_error'])
# mse_plot.set_xlabel('epoch')
# mse_plot.set_ylabel('mse')
# mse_plot.legend(['Train mse', 'Validation mse'])