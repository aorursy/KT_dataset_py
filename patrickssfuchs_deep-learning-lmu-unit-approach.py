import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!git clone --recurse-submodules https://github.com/abr/neurips2019 lmu

!pip install nengolib

from lmu.lmu import LMUCell

from keras.layers.recurrent import RNN
!pip install tensorflow-addons

import tensorflow_addons as tfa

from tensorflow_addons.optimizers import RectifiedAdam
import numpy as np

np.random.seed(420)

import pandas as pd

from tqdm.notebook import tqdm

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt

import plotly.graph_objects as go
def create_dataset(seq_len, scaler = None):

    train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

    train = train.replace(np.nan, "")

    train["Id"] = 0

    train = train.rename(columns = {"Id": "ForecastId"})



    test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv").replace(np.nan, "")

    test = test.join(pd.Series(np.zeros((len(test),), dtype="float64"), name="ConfirmedCases"))

    test = test.join(pd.Series(np.zeros((len(test),), dtype="float64"), name="Fatalities"))

    date = train.loc[len(train) -1, "Date"]



    

    tmp_train = train[train["Date"] >= test.loc[0, "Date"]]

    tmp_test = test[test["Date"] <= train.loc[len(train) - 1, "Date"]]

    tmp_test[["ConfirmedCases", "Fatalities"]] = tmp_train[["ConfirmedCases", "Fatalities"]].values

    test = test[test["Date"] > date] 



    df = pd.concat([train[train["Date"] < "2020-03-26"],test, tmp_test], axis=0)

    df = df.join(pd.Series([[]]*len(df), name="x_window"))

    df = df.join(pd.Series([[]]*len(df), name="y_window"))

    paises_estado = df["Country_Region"] + " " + df["Province_State"] 

    paises_estado.name = "Paises_Estado"

    unique_place = paises_estado.copy()

    unique_place.name = "Unique_Places"

    df = pd.concat([df,paises_estado, unique_place], axis=1)

    df["Unique_Places"] = df["Unique_Places"].apply(lambda x: x[:-1] if x[-1] == " " else x)

    df["Paises_Estado"] = df["Paises_Estado"].apply(lambda x: x[:-1] if x[-1] == " " else x)

    df = df.sort_values(["Country_Region","Province_State","Date"])

    df = df.reset_index()

    



    if scaler != None:

        scaler.fit(df[["ConfirmedCases", "Fatalities"]])

        df[["ConfirmedCases", "Fatalities"]] = scaler.transform(df[["ConfirmedCases", "Fatalities"]])

    

    Onehot = OneHotEncoder(categories = "auto" )

    Onehot.fit(df["Paises_Estado"].values.reshape(-1,1))

    Onehot.sparse = False

    labels_encoded = Onehot.transform(df["Paises_Estado"].values.reshape(-1,1))

    for i in tqdm(range(len(df)), desc = "One Hot"):

        df.loc[i, "Paises_Estado"] = [labels_encoded[i:i+1]]

    

    

    for i in tqdm(range(len(df)), desc = "Sliding Window"):

        if df.loc[i,"Date"] == "2020-01-22": 

            flag = True 

            count=0

            queue_array = [np.zeros((1,seq_len,2), dtype="float32")]

        if flag:

            if i != 0:

                queue_array = [np.roll(queue_array[0], -1)]

                queue_array[0][0,-1] = np.expand_dims(df[["ConfirmedCases", "Fatalities"]].iloc[i-1:i],  axis=0)[0]

            df.loc[i,"x_window"] = queue_array

            df.loc[i,"y_window"] = [(np.expand_dims(df[["ConfirmedCases", "Fatalities"]].iloc[i],  axis=0)).astype("float32")] #[np.zeros((1,2), dtype="float64")]

            count += 1

        else:

            df.loc[i,"x_window"] = [(np.expand_dims(df[["ConfirmedCases", "Fatalities"]].iloc[i-seq_len:i],  axis=0)).astype("float32")]

            df.loc[i,"y_window"] = [(np.expand_dims(df[["ConfirmedCases", "Fatalities"]].iloc[i],  axis=0)).astype("float32")]

        if count == seq_len: flag = False

    

    if scaler != None: return df, scaler, date

    else: return df



    

def load_data(data):

    shape = list(data.values[0].shape)

    shape[0] = len(data)

    data_array = np.zeros(shape, dtype = "float64")

    for i in range(len(data)):

        data_array[i] = data.values[i]

    return data_array
seq_len = 5

df, scaler, date_train = create_dataset(seq_len, MinMaxScaler())


train_data = df[df["Date"] <= date_train]

test_data = df[df["Date"] >= "2020-03-26"]



days_to_val = 12

date = str(pd.to_datetime(pd.to_datetime(date_train).value - (60*60*24)*10**9*days_to_val))[:-9]



x_train_window = load_data(train_data[train_data["Date"] <= date]["x_window"])

x_train_place = load_data(train_data[train_data["Date"] <= date]["Paises_Estado"])

y_train = load_data(train_data[train_data["Date"] <= date]["y_window"])



x_val_window = load_data(train_data[train_data["Date"] > date]["x_window"])

x_val_place = load_data(train_data[train_data["Date"] > date]["Paises_Estado"])

y_val =  load_data(train_data[train_data["Date"] > date]["y_window"])
import tensorflow as tf

from keras.models import Sequential, Model

from keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, Conv2D, Activation, MaxPooling2D, Flatten, Reshape, Add

from keras.layers import Conv1D, MaxPooling1D, SimpleRNN, LeakyReLU, PReLU, ELU, ReLU, Concatenate, RepeatVector, AveragePooling1D

from keras.callbacks import ModelCheckpoint, TensorBoard

import keras.backend as K
batch_size = 1024*4



inp_window = Input(shape=x_train_window[0,:].shape)

inp_pais = Input(shape=x_train_place[0,:].shape)





#dense = Dense(20, activation = activation)(inp_pais)

#dense = Dense(20, activation = activation)(dense)

rep = RepeatVector(x_train_window.shape[1])(inp_pais)



concat = Concatenate()([inp_window, rep])

#concat_conv = Conv1D(128, 1, activation = activation, padding="same")(concat)



#conv = Conv1D(128, 1, padding="same")(concat)

#conv = ELU()(conv)

#conv = Conv1D(64, 1, padding="same")(concat)

#conv = ELU()(conv)

#conv = MaxPooling1D(2)(conv)

#conv = Conv1D(64, 3, activation = activation,padding="same")(conv)

#conv = MaxPooling1D(2)(conv)

#conv = Conv1D(64, 3, activation = activation,padding="same")(conv)

#conv = MaxPooling1D(2)(conv)



#conv = Conv1D(32, 1, activation = activation,padding="same")(conv)



#lstm = LSTM(lstm_out, return_sequences=False)(inp_window)

lstm = RNN(LMUCell(units=212, order = 256, theta = 212), return_sequences=False)(concat)

#lstm = Dense(lstm_out//2)(lstm)

#lstm = ELU()(lstm)

#lstm = Dense(lstm_out//2)(lstm)

#lstm = ELU()(lstm)



#inp_pais = Input(shape=x_train_place[0,:].shape)

#dense_pais = Dense(lstm_out//2, activation = activation)(inp_pais)

#dense_pais = Dense(lstm_out//2, activation = activation)(dense_pais)



#concat2 = Concatenate()([reshape, inp_pais])

#flat = Flatten()(inp_window)



x = Dense(64*2)(lstm)

x = ELU()(x)

#x = Add()([x, inp_pais])

x = Dense(20*2)(x)

x = ELU()(x)

#x = Dense(lstm_out//2, activation = activation)(x)

#x = Dense(lstm_out//2, activation = activation)(x)



y_hat = Dense(2)(x)

#ff = tf.where(pred[:,1] < inp_window[:,-1,1], inp_window[:,-1,1], pred[:,1])

#cc = tf.where(pred[:,0] < inp_window[:,-1,0], inp_window[:,-1,0], pred[:,0])



#y_hat = Concatenate()([tf.expand_dims(cc, axis=1),tf.expand_dims(ff, axis=1)])





model = Model([inp_window,inp_pais], y_hat)

#model = Model([inp_window], y_hat)

model.summary()
def root_mean_squared_log_error(y_true, y_pred):

    return K.sqrt(K.mean(K.square(K.log(y_pred + 1) - K.log(y_true + 1)))) 



model.compile(loss = root_mean_squared_log_error, optimizer= RectifiedAdam() )

cp = ModelCheckpoint("test_sub.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only= True)



history = model.fit([x_train_window, x_train_place], y_train, batch_size = batch_size, epochs = 500,  

                       verbose = 1, validation_data=([x_val_window, x_val_place], y_val), callbacks = [cp])



#history = model.fit([x_train_window], y_train, batch_size = batch_size, epochs = 150,  

#                       verbose = 1, callbacks = [cp])
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')
model.load_weights("test_sub.hdf5")



days_to_show = days_to_val

date = str(pd.to_datetime(pd.to_datetime(date_train).value - (60*60*24)*10**9*days_to_show))[:-9]

#date = "2020-03-25"

paises = dict()

for pais in tqdm(train_data['Country_Region'].unique()):

    

    pais_data = train_data[train_data["Country_Region"] == pais]

    x_window = load_data(pais_data[pais_data["Date"] > date]["x_window"])

    x_place = load_data(pais_data[pais_data["Date"] > date]["Paises_Estado"])

    gt_val = load_data(pais_data[pais_data["Date"] > date]["y_window"])

    

    paises[pais] = model.evaluate([x_window, x_place], gt_val)

    #paises[pais] = model.evaluate([x_window], gt_val, verbose=0)



plt.figure(figsize=(10,40))

plt.barh(list(paises.keys()), paises.values())



days_to_predict = days_to_val

date = str(pd.to_datetime(pd.to_datetime(date_train).value - (60*60*24)*10**9*days_to_predict))[:-9]



from plotly.subplots import make_subplots

fig = make_subplots(

    rows=2, cols=2, subplot_titles=("Italy", "Spain", "Germany", "Brazil")

)

paises = ["Italy", "Spain", "Germany", "Brazil"]

for i in range(len(paises)):

    

    pais = paises[i]

    pais_data = train_data[train_data["Country_Region"] == pais]

    x_window = load_data(pais_data[pais_data["Date"] > date]["x_window"])

    x_place = load_data(pais_data[pais_data["Date"] > date]["Paises_Estado"])



    pred_data = model.predict([x_window,x_place])

    #pred_data = model.predict([x_window])

    pred_data = scaler.inverse_transform(pred_data)

    x_new_dates = pais_data[pais_data["Date"] > date]["Date"]





    x_dates = pais_data["Date"]

    true_data = pd.concat([pais_data['ConfirmedCases'], pais_data['Fatalities']], axis=1)

    true_data = scaler.inverse_transform(true_data)

    true_data = pd.DataFrame(true_data, columns = ['ConfirmedCases', 'Fatalities'])



    if i < 2: row = 1

    else: row = 2

        

    fig.add_trace(go.Scatter(x=x_dates, y=true_data['ConfirmedCases'], name=f'Confirmed Cases ({pais})'), row = row , col = (i % 2)+1)

    fig.add_trace(go.Scatter(x=x_dates, y=true_data['Fatalities'], name=f'Fatalities ({pais})'), row = row, col = (i % 2)+1)

    fig.add_trace(go.Scatter(x=x_new_dates, y=pred_data[:, 0], name=f'Predicted Confirmed Cases ({pais})'), row = row, col = (i % 2)+1)

    fig.add_trace(go.Scatter(x=x_new_dates, y=pred_data[:, 1], name=f'Predicted Fatalities ({pais})'), row = row, col = (i % 2)+1)



fig.update_layout(title='Forecast for COVID-19', xaxis_title="Date", yaxis_title="New Cases/Fatalities", template = "plotly_dark", width=1400, height=900)

fig.write_html("file.html")

fig.show()

test_data["ForecastId"] = np.arange(1,13159)

df_predict = pd.concat([train_data[train_data["Date"] < "2020-03-26"], test_data], axis=0)

df_predict = df_predict.sort_values(["Country_Region","Province_State","Date"])
max_look_back = 76

for i in tqdm(range(len(df_predict)), desc = "predict"):

    

    if df_predict.loc[i, "Date"] < date_train:

        continue

    elif df_predict.loc[i, "Date"] == date_train:

        decay_c = 1.00

        decay_f = 1.00

        continue

    else:

        tmp = [np.zeros((1,seq_len,2), dtype="float64")]

        if seq_len < max_look_back:

            tmp[0][-seq_len:] = [np.expand_dims(df_predict[["ConfirmedCases", "Fatalities"]].iloc[i-seq_len:i],  axis=0)][0]

        else:

            tmp[0][0,-max_look_back:] = np.expand_dims(df_predict[["ConfirmedCases", "Fatalities"]].iloc[i-max_look_back:i],  axis=0)[0]

            if max_look_back != seq_len:

                max_look_back += 1

                

        df_predict.loc[i,"x_window"] = tmp

        pred = model.predict([df_predict.loc[i,"x_window"], df_predict.loc[i,"Paises_Estado"]])

        #pred = model.predict([df_predict.loc[i,"x_window"]])

        if pred[0,0] < df_predict.loc[i-1,"ConfirmedCases"]: 

            pred[0,0] = df_predict.loc[i-1,"ConfirmedCases"] + ((df_predict.loc[i-1,"ConfirmedCases"] - df_predict.loc[i-2,"ConfirmedCases"]) * decay_c)

            if not(decay_c < 0.0):

                decay_c -= 0.02

        if pred[0,1] < df_predict.loc[i-1,"Fatalities"]: 

            pred[0,1] = df_predict.loc[i-1,"Fatalities"] + ((df_predict.loc[i-1,"Fatalities"] - df_predict.loc[i-2,"Fatalities"]) * decay_f)

            if not(decay_f < 0.0):

                decay_f -= 0.02

        df_predict.loc[i,"ConfirmedCases"], df_predict.loc[i,"Fatalities"] = pred[0,0] , pred[0,1]
df_predict[df_predict["Country_Region"] == "Italy"].iloc[50:100]
days_to_predict = days_to_val + 46

date = str(pd.to_datetime(pd.to_datetime("2020-05-07").value - (60*60*24)*10**9*days_to_predict))[:-9]



fig = make_subplots(

    rows=2, cols=2, subplot_titles=("Italy", "Spain", "Germany", "Brazil")

)

paises = ["Italy", "Spain", "Germany", "Brazil"]



for i in range(len(paises)):



    pais = paises[i]

    pais_data = df_predict[df_predict["Country_Region"] == pais]

    x_window = load_data(pais_data[pais_data["Date"] > date]["x_window"])

    x_place = load_data(pais_data[pais_data["Date"] > date]["Paises_Estado"])



    pred_data = model.predict([x_window, x_place])

    #pred_data = model.predict([x_window])

    pred_data = scaler.inverse_transform(pred_data)

    x_new_dates = pais_data[pais_data["Date"] > date]["Date"]



    x_dates = pais_data["Date"]

    true_data = pd.concat([pais_data['ConfirmedCases'], pais_data['Fatalities']], axis=1)

    true_data = scaler.inverse_transform(true_data)

    true_data = pd.DataFrame(true_data, columns = ['ConfirmedCases', 'Fatalities'])



    if i < 2: row = 1

    else: row = 2

    fig.add_trace(go.Scatter(x=x_dates, y=true_data['ConfirmedCases'], name=f'Confirmed Cases ({pais})'), row = row , col = (i % 2)+1)

    fig.add_trace(go.Scatter(x=x_dates, y=true_data['Fatalities'], name=f'Fatalities ({pais})'), row = row, col = (i % 2)+1)

    fig.add_trace(go.Scatter(x=x_new_dates, y=pred_data[:, 0], name=f'Predicted Confirmed Cases ({pais})', mode='lines+markers'), row = row, col = (i % 2)+1)

    fig.add_trace(go.Scatter(x=x_new_dates, y=pred_data[:, 1], name=f'Predicted Fatalities ({pais})', mode='lines+markers'), row = row, col = (i % 2)+1)



fig.update_layout(title='Forecast for COVID-19', xaxis_title="Date", yaxis_title="New Cases/Fatalities", template = "plotly_dark", width=1400, height=900)

fig.write_html("file.html")

fig.show()
sub = df_predict[df_predict["ForecastId"] != 0]

sub = sub[["ForecastId", "ConfirmedCases", "Fatalities"]]

sub[["ConfirmedCases", "Fatalities"]] = scaler.inverse_transform(sub[["ConfirmedCases", "Fatalities"]])

sub.shape

sub.iloc[:50]
sub.to_csv("submission.csv", index=False)
#df_predict[df_predict["Date"] > "2020-03-25"]
#def root_mean_squared_log_error_np(y_true, y_pred):

#    return np.sqrt(np.mean(np.square(np.log(y_pred + 1) - np.log(y_true + 1)))) 
'''

y_true = df_predict[df_predict["Date"] >= "2020-03-26"]

y_true = y_true[y_true["Date"] <= "2020-04-08"  ]

y_true = y_true[y_true["Unique_Places"] == "Italy"]

y_pred = load_data(y_true["x_window"])

y_true.iloc[:50]

#sub[["ConfirmedCases", "Fatalities"]] = scaler.inverse_transform(sub[["ConfirmedCases", "Fatalities"]])

'''
'''

y_true = df_predict[df_predict["Date"] >= "2020-03-26"]

y_true = y_true[y_true["Date"] <= "2020-04-08"  ]

y_true = y_true[y_true["Unique_Places"] == pais]

y_true[["ConfirmedCases", "Fatalities"]] = scaler.inverse_transform(y_true[["ConfirmedCases", "Fatalities"]])

y_pred_tmp = y_true[["ConfirmedCases", "Fatalities"]].values[:-2]

y_pred = load_data(y_true["x_window"][-2:])

y_pred = model.predict(y_pred)

#y_true = scaler.inverse_transform(y_true)

y_pred = scaler.inverse_transform(y_pred)

#print(y_true["ConfirmedCases"].values[len(y_pred) -2])

if y_pred[0,0] < y_true["ConfirmedCases"].values[len(y_true) - 3]:

    y_pred[0,0] = y_true["ConfirmedCases"].values[len(y_true) - 3] * 1.07

if y_pred[1,0] < y_pred[0,0]:

    y_pred[1,0] = y_pred[0,0]  * 1.07



if y_pred[0,1] < y_true["Fatalities"].values[len(y_true) - 3]:

    #print(y_pred[0,1])

    y_pred[0,1] = y_true["Fatalities"].values[len(y_true) - 3] * 1.07

    #print(y_pred[0,1])

if y_pred[1,1] < y_pred[0,1]:

    #print(y_pred[1,1])

    y_pred[1,1] = y_pred[0,1] * 1.07

    #print(y_pred[1,1])

y_pred = np.concatenate([y_pred_tmp, y_pred], axis=0)

'''
'''

paises = []

paises_zeros = []

for pais in tqdm(train_data['Unique_Places'].unique()):

    #if pais not in "Italy":

    #    continue

    #else:

        y_true = df_predict[df_predict["Date"] >= "2020-03-26"]

        y_true = y_true[y_true["Date"] <= "2020-04-08"  ]

        y_true = y_true[y_true["Unique_Places"] == pais]

        y_true[["ConfirmedCases", "Fatalities"]] = scaler.inverse_transform(y_true[["ConfirmedCases", "Fatalities"]])

        y_pred_tmp = y_true[["ConfirmedCases", "Fatalities"]].values[:-2]

        y_pred = load_data(y_true["x_window"][-2:])

        y_pred = model.predict(y_pred)

        #y_true = scaler.inverse_transform(y_true)

        y_pred = scaler.inverse_transform(y_pred)

        #print(y_true["ConfirmedCases"].values[len(y_pred) -2])

        if y_pred[0,0] < y_true["ConfirmedCases"].values[len(y_true) - 3]:

            y_pred[0,0] = y_true["ConfirmedCases"].values[len(y_true) - 3] * 1.07

        if y_pred[1,0] < y_pred[0,0]:

            y_pred[1,0] = y_pred[0,0]  * 1.07



        if y_pred[0,1] < y_true["Fatalities"].values[len(y_true) - 3]:

            #print(y_pred[0,1])

            y_pred[0,1] = y_true["Fatalities"].values[len(y_true) - 3] * 1.07

            #print(y_pred[0,1])

        if y_pred[1,1] < y_pred[0,1]:

            #print(y_pred[1,1])

            y_pred[1,1] = y_pred[0,1] * 1.07

            #print(y_pred[1,1])

        y_pred = np.concatenate([y_pred_tmp, y_pred], axis=0)

        #print(y_pred)

        y_pred_with_zeros = y_pred.copy()

        y_pred_with_zeros[-2:] = 0.0, 0.0 

        paises_zeros.append( np.mean(root_mean_squared_log_error_np(y_true[["ConfirmedCases", "Fatalities"]], y_pred_with_zeros)))

        paises.append( np.mean(root_mean_squared_log_error_np(y_true[["ConfirmedCases", "Fatalities"]], y_pred)))

        #break



    #paises

'''
#y_pred
#np.sum(paises)
#np.sum(paises_zeros)
#y_pred
#y_true[["ConfirmedCases", "Fatalities"]].values
'''

def root_mean_squared_log_error_np(y_true, y_pred):

    tmp_log_pred = np.log(y_pred + 1)

    #print(tmp_log_pred)

    tmp_log_true = np.log(y_true + 1)

    #print(tmp_log_true)

    tmp_square = np.square(tmp_log_pred - tmp_log_true)

    #print(tmp_square)

    tmp_mean = np.mean(tmp_square)

    print(tmp_mean)

    tmp_sqrt = np.sqrt(tmp_mean)

    print(np.mean(tmp_sqrt))

    

    

    #return np.sqrt(np.mean(np.square(np.log(y_pred + 1) - np.log(y_true + 1)))) 

'''
#root_mean_squared_log_error_np(y_true[["ConfirmedCases", "Fatalities"]], y_pred)
'''

paises = []

paises_zeros = []

for pais in tqdm(train_data['Unique_Places'].unique()):

    #if pais not in "Italy":

    #    continue

    #else:

        y_pred = df_predict[df_predict["Date"] >= "2020-03-26"]

        y_pred = y_pred[y_pred["Date"] <= "2020-05-07"  ]

        y_pred = y_pred[y_pred["Unique_Places"] == pais]

        y_pred[["ConfirmedCases", "Fatalities"]] = scaler.inverse_transform(y_pred[["ConfirmedCases", "Fatalities"]])

        tmp = y_pred[y_pred["Date"] <= "2020-04-06"  ]

        tmp2 = y_pred[y_pred["Date"] > "2020-04-06"  ]

        tmp2[["ConfirmedCases", "Fatalities"]] = 0.0, 0.0

        y_true = pd.concat([tmp,tmp2], axis=0)

        

        

        

        

        

        y_pred_tmp = y_true[["ConfirmedCases", "Fatalities"]].values[:-2]

        y_pred = load_data(y_true["x_window"][-2:])

        y_pred = model.predict(y_pred)

        #y_true = scaler.inverse_transform(y_true)

        y_pred = scaler.inverse_transform(y_pred)

        #print(y_true["ConfirmedCases"].values[len(y_pred) -2])

        if y_pred[0,0] < y_true["ConfirmedCases"].values[len(y_true) - 3]:

            y_pred[0,0] = y_true["ConfirmedCases"].values[len(y_true) - 3] * 1.07

        if y_pred[1,0] < y_pred[0,0]:

            y_pred[1,0] = y_pred[0,0]  * 1.07



        if y_pred[0,1] < y_true["Fatalities"].values[len(y_true) - 3]:

            #print(y_pred[0,1])

            y_pred[0,1] = y_true["Fatalities"].values[len(y_true) - 3] * 1.07

            #print(y_pred[0,1])

        if y_pred[1,1] < y_pred[0,1]:

            #print(y_pred[1,1])

            y_pred[1,1] = y_pred[0,1] * 1.07

            #print(y_pred[1,1])

        y_pred = np.concatenate([y_pred_tmp, y_pred], axis=0)

        #print(y_pred)

        y_pred_with_zeros = y_pred.copy()

        y_pred_with_zeros[-2:] = 0.0, 0.0 

        paises_zeros.append( np.mean(root_mean_squared_log_error_np(y_true[["ConfirmedCases", "Fatalities"]], y_pred_with_zeros)))

        paises.append( np.mean(root_mean_squared_log_error_np(y_true[["ConfirmedCases", "Fatalities"]], y_pred)))

        #break



    #paises

'''
'''

y_pred = df_predict[df_predict["ForecastId"] != 0]

y_pred[["ConfirmedCases", "Fatalities"]] = scaler.inverse_transform(y_pred[["ConfirmedCases", "Fatalities"]])

y_true = y_pred.copy(deep = True)

#y_true = y_true[["ConfirmedCases", "Fatalities"]].values



#print(y_true)

for i in tqdm(range(y_true.shape[0])):

    #print(tmp)

    if y_true["Date"].iloc[i] > "2020-04-06":

        y_true["ConfirmedCases"].iloc[i] = 0.0

        y_true["Fatalities"].iloc[i] = 0.0

        

'''
#y_pred.iloc[:50]
#y_true.iloc[:50]
'''

def root_mean_squared_log_error_np(y_true, y_pred):

    return np.sqrt(np.mean(np.square(np.log(y_pred + 1) - np.log(y_true + 1)))) 



np.mean(root_mean_squared_log_error_np(y_true[["ConfirmedCases", "Fatalities"]], y_pred[["ConfirmedCases", "Fatalities"]]))

'''
#np.sum(root_mean_squared_log_error_np(y_true[["ConfirmedCases", "Fatalities"]], y_pred[["ConfirmedCases", "Fatalities"]]))#