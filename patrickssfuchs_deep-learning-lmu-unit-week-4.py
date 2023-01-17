import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!git clone --recurse-submodules https://github.com/Patrick-Schubert/tensorflow2_LMU.git lmu

!pip install nengolib

!pip install tensorflow-addons
#Install from Github

from lmu.lmu import LMUCell

from tensorflow_addons.optimizers import RectifiedAdam



#Deep Learning Frameworks

import torch

import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, Conv2D, Activation, MaxPooling2D, Flatten, Reshape, Add, RNN, BatchNormalization

from tensorflow.keras.layers import Conv1D, MaxPooling1D, SimpleRNN, LeakyReLU, PReLU, ELU, ReLU, Concatenate, RepeatVector, AveragePooling1D, TimeDistributed

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback







#Machine Learning Stuff

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder



#Data Visualization Stuff

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from plotly.subplots import make_subplots



#Data Stuff

import numpy as np

import pandas as pd



#Others

from tqdm.notebook import tqdm

import os

import random

import warnings

warnings.simplefilter(action='ignore')



def seed_keras(seed=420):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    # hack: Even though we don't use torch, this can be used for Keras. ",

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_keras()

#import f

#from f import create_dataset, build_train_data, build_nn, load_data, plot_train_statistics, plot_curve_for_countries, extrapolation, plot_extrapolation, submission, train

#from importlib import reload

#os.environ["CUDA_VISIBLE_DEVICES"] = "8"
dtype = "float64"



def create_dataset(seq_len, scaler = None, log = False, cut_zeros = False):

    train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

    train = train.replace(np.nan, "")

    train["Id"] = 0

    train = train.rename(columns = {"Id": "ForecastId"})



    test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv").replace(np.nan, "")

    test = test.join(pd.Series(np.zeros((len(test),), dtype = dtype), name="ConfirmedCases"))

    test = test.join(pd.Series(np.zeros((len(test),), dtype = dtype), name="Fatalities"))

    date = train.loc[len(train) -1, "Date"]

    date_test = test.loc[0, "Date"]

    

    tmp_train = train[train["Date"] >= test.loc[0, "Date"]]

    tmp_test = test[test["Date"] <= train.loc[len(train) - 1, "Date"]]

    tmp_test[["ConfirmedCases", "Fatalities"]] = tmp_train[["ConfirmedCases", "Fatalities"]].values

    test = test[test["Date"] > date] 



    df = pd.concat([train[train["Date"] < date_test],test, tmp_test], axis=0)

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

    

    

    if log: df[["ConfirmedCases", "Fatalities"]] = np.log1p(df[["ConfirmedCases", "Fatalities"]])

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

        if df.loc[i,"Date"] == train.loc[0, "Date"]: 

            flag = True 

            count=0

            queue_array = [np.zeros((1,seq_len,2), dtype=dtype)]

        if flag:

            if i != 0:

                queue_array = [np.roll(queue_array[0], -1)]

                queue_array[0][0,-1] = np.expand_dims(df[["ConfirmedCases", "Fatalities"]].iloc[i-1:i],  axis=0)[0]

            df.loc[i,"x_window"] = queue_array

            df.loc[i,"y_window"] = [(np.expand_dims(df[["ConfirmedCases", "Fatalities"]].iloc[i],  axis=0)).astype(dtype)] 

            count += 1

        else:

            df.loc[i,"x_window"] = [(np.expand_dims(df[["ConfirmedCases", "Fatalities"]].iloc[i-seq_len:i],  axis=0)).astype(dtype)]

            df.loc[i,"y_window"] = [(np.expand_dims(df[["ConfirmedCases", "Fatalities"]].iloc[i],  axis=0)).astype(dtype)]

        if count == seq_len: flag = False

    

    if cut_zeros:

        df = df[ (df["ConfirmedCases"] > 0.0) | (df["Date"] > date)]

        for pais in df["Unique_Places"].unique():

            pais_df = df[df["Unique_Places"] == pais]

            idx = pais_df.iloc[0].name

            df = df.drop(idx, axis=0)



    if scaler != None: return df, scaler, date, date_test

    else: return df, date, date_test

    

    

def load_data(data):

    shape = list(data.values[0].shape)

    shape[0] = len(data)

    data_array = np.zeros(shape, dtype = dtype)

    for i in range(len(data)):

        data_array[i] = data.values[i]

    return data_array



def build_train_data(train_data, date_train, days_to_val = 3):

    date = str(pd.to_datetime(pd.to_datetime(date_train).value - (60*60*24)*10**9*days_to_val))[:-9]

    

    x_train_window = load_data(train_data[train_data["Date"] <= date]["x_window"])

    x_train_place = load_data(train_data[train_data["Date"] <= date]["Paises_Estado"])

    y_train = load_data(train_data[train_data["Date"] <= date]["y_window"])



    x_val_window = load_data(train_data[train_data["Date"] > date]["x_window"])

    x_val_place = load_data(train_data[train_data["Date"] > date]["Paises_Estado"])

    y_val =  load_data(train_data[train_data["Date"] > date]["y_window"])

    

    return date, x_train_window, x_train_place , y_train ,x_val_window ,x_val_place, y_val



def build_nn(x_train_window, x_train_place):

    factor = 5

    inp_window = Input(shape=x_train_window[0,:].shape)

    inp_pais = Input(shape=x_train_place[0,:].shape)



    dense_pais = Dense(128, use_bias = False)(inp_pais)

    dense_pais = ELU()(dense_pais)

    dense_pais = Dense(128, use_bias = False)(dense_pais)

    dense_pais = ELU()(dense_pais)

    rep = RepeatVector(x_train_window.shape[1])(dense_pais)



    flat_window = Flatten()(inp_window)

    dense_window = Dense(30, use_bias = False)(flat_window)

    dense_window = ELU()(dense_window)

    dense_window = Dense(15, use_bias = False)(dense_window)

    dense_window = ELU()(dense_window)

    rep_window = RepeatVector(x_train_window.shape[1])(dense_window)





    concat = Concatenate(axis=-1)([inp_window, rep, rep_window])





    lstm = RNN(LMUCell(units=int(30*factor), order = 15*factor, theta = 30*factor), return_sequences=False, go_backwards = False)(concat)





    x = Dense(128, use_bias = False)(lstm)

    x = ELU()(x)

    x = Dense(64, use_bias = False)(x)

    x = ELU()(x)



    y_hat = Dense(2)(x)



    model = Model([inp_window, inp_pais], y_hat)

    return model

              

def train(x_train, y_train, x_val, y_val, model, model_name = "Sub", log = False):

    def root_mean_squared_log_error(y_true, y_pred):

        return K.mean(K.sqrt(K.mean(K.square(K.log(y_pred + 1) - K.log(y_true + 1)), axis = 0)))

    

    def root_mean_squared_error(y_true, y_pred):

        return K.mean(K.sqrt(K.mean(K.square(y_pred - y_true), axis = 0))) 

    

    if log:

        model.compile(loss = root_mean_squared_error, optimizer= RectifiedAdam())

    else:  

        model.compile(loss = root_mean_squared_log_error, optimizer= RectifiedAdam())



    cp = ModelCheckpoint(f"{model_name}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only= True)



    #batch_size = y_train.shape[0] // 33

    batch_size = 512

    from tqdm.keras import TqdmCallback

    



    class TerminateOnNaN(Callback):

        def on_batch_end(self, batch, logs=None):

            logs = logs or {}

            loss = logs.get('loss')

            if loss is not None:

                if np.isnan(loss) or np.isinf(loss):

                    print('Batch %d: Invalid loss, terminating training' % (batch))

                    self.model.stop_training = True

                    

    terminate = TerminateOnNaN()  

    history = model.fit(x_train, y_train, batch_size = batch_size, epochs = 1000,  

                       verbose = 0, validation_data=(x_val, y_val), callbacks = [cp, terminate,TqdmCallback(verbose=0)])

    

    if np.isinf(history.history["loss"][0]):

        print("Retrain")

        model = build_nn(x_train[0], x_train[1])

        model, history = train([x_train[0], x_train[1]], y_train, [x_val[0], x_val[1]], y_val, model, model_name, log = False)

    

    return model, history



    

    

def plot_train_statistics(history, model, model_name, train_df, date_train, days_to_show):

    fig = make_subplots(

        rows=2, cols=1, subplot_titles=("Model Loss", "Val Loss by Country")

    )

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    fig.add_trace(go.Scatter(x = np.arange(len(loss)), y = loss, name= "Train"), row = 1, col=1)

    fig.add_trace(go.Scatter(x = np.arange(len(val_loss)), y = loss, name= "Val"), row = 1, col=1)

    fig.update_xaxes(title_text="Epoch", row=1, col=1)

    fig.update_yaxes(title_text="Loss", row=1, col=1)



    model.load_weights(f"{model_name}.hdf5")



    date = str(pd.to_datetime(pd.to_datetime(date_train).value - (60*60*24)*10**9*days_to_show))[:-9]

    paises = dict()

    for pais in tqdm(train_df['Country_Region'].unique()):



        pais_data = train_df[train_df["Country_Region"] == pais]

        x_window = load_data(pais_data[pais_data["Date"] > date]["x_window"])

        x_place = load_data(pais_data[pais_data["Date"] > date]["Paises_Estado"])

        gt_val = load_data(pais_data[pais_data["Date"] > date]["y_window"])



        paises[pais] = model.evaluate([x_window, x_place], gt_val, verbose=0)

        #paises[pais] = model.evaluate([x_window], gt_val, verbose=0)



    fig.add_trace(go.Bar(x = list(paises.keys()), y=list(paises.values()), name="Countries", showlegend=False), row = 2, col = 1)

    fig.update_xaxes(title_text="Countries", tickangle=45,row=2, col=1)

    fig.update_yaxes(title_text="Val Loss", row=2, col=1)



    fig.update_layout(height=800, width=1300, template = "plotly_dark")

    fig.show()

    

    

def plot_curve_for_countries(paises, model, model_name, train_df, date_train, days_to_predict, scaler = None, log = False):



    model.load_weights(f"{model_name}.hdf5")

    date = str(pd.to_datetime(pd.to_datetime(date_train).value - (60*60*24)*10**9*days_to_predict))[:-9]



    fig = make_subplots( rows=2, cols=2, subplot_titles= paises )



    for i in range(len(paises)):



        pais = paises[i]

        pais_data = train_df[train_df["Country_Region"] == pais]

        x_window = load_data(pais_data[pais_data["Date"] > date]["x_window"])

        x_place = load_data(pais_data[pais_data["Date"] > date]["Paises_Estado"])



        pred_data = model.predict([x_window,x_place])

        #pred_data = model.predict([x_window])

        #pred_data = np.expm1(pred_data)

        if scaler != None:

            pred_data = scaler.inverse_transform(pred_data)

        #pred_data[:,0] = np.expm1(pred_data[:, 0]) + 0.1

        #pred_data[:,1] = np.expm1(pred_data[:, 1]) + 0.01

        #print(pred_data)

        #pred_data = np.expm1(pred_data)

        #print(pred_data)

        x_new_dates = pais_data[pais_data["Date"] > date]["Date"]





        x_dates = pais_data["Date"]

        true_data = pd.concat([pais_data['ConfirmedCases'], pais_data['Fatalities']], axis=1)

        #true_data = np.expm1(true_data)

        if scaler != None:

            true_data = scaler.inverse_transform(true_data)

        #true_data = np.expm1(true_data)

        true_data = pd.DataFrame(true_data, columns = ['ConfirmedCases', 'Fatalities'])

        #print((pred_data, true_data))

        

        if log:

            pred_data = np.expm1(pred_data)

            true_data = np.expm1(true_data)

        if i < 2: row = 1

        else: row = 2



        fig.add_trace(go.Scatter(x=x_dates, y=true_data['ConfirmedCases'], name=f'Confirmed Cases ({pais})'), row = row , col = (i % 2)+1)

        fig.add_trace(go.Scatter(x=x_dates, y=true_data['Fatalities'], name=f'Fatalities ({pais})'), row = row, col = (i % 2)+1)

        fig.add_trace(go.Scatter(x=x_new_dates, y=pred_data[:, 0], name=f'Predicted Confirmed Cases ({pais})'), row = row, col = (i % 2)+1)

        fig.add_trace(go.Scatter(x=x_new_dates, y=pred_data[:, 1], name=f'Predicted Fatalities ({pais})'), row = row, col = (i % 2)+1)



    fig.update_layout(title='Val Points Predictions', xaxis_title="Date", yaxis_title="New Cases/Fatalities", template = "plotly_dark", width=1400, height=900)

    #fig.write_html("file.html")

    fig.show()



    

def extrapolation(train_df, test_df, date_train, date_test, seq_len, model, model_name, trend = False):



    model.load_weights(f"{model_name}.hdf5")



    test_df["ForecastId"] = np.arange(1,13459+1)

    df_predict = pd.concat([train_df[train_df["Date"] < date_test], test_df], axis=0)

    df_predict = df_predict.sort_values(["Country_Region","Province_State","Date"])



    max_look_back = 75 + 1

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



            if trend:

                if pred[0,0] < df_predict.loc[i-1,"ConfirmedCases"]: 

                    pred[0,0] = df_predict.loc[i-1,"ConfirmedCases"] + ((df_predict.loc[i-1,"ConfirmedCases"] - df_predict.loc[i-2,"ConfirmedCases"]) * decay_c)

                    if not(decay_c < 0.0):

                        decay_c -= 0.02

                if pred[0,1] < df_predict.loc[i-1,"Fatalities"]: 

                    pred[0,1] = df_predict.loc[i-1,"Fatalities"] + ((df_predict.loc[i-1,"Fatalities"] - df_predict.loc[i-2,"Fatalities"]) * decay_f)

                    if not(decay_f < 0.0):

                        decay_f -= 0.02



            df_predict.loc[i,"ConfirmedCases"], df_predict.loc[i,"Fatalities"] = pred[0,0] , pred[0,1]

    return df_predict

    

def plot_extrapolation(paises, df_predict, date_train, scaler):



    fig = make_subplots(rows=2, cols=2, subplot_titles=paises)



    for i in range(len(paises)):



        pais = paises[i]

        pais_data = df_predict[df_predict["Country_Region"] == pais]

        pred_data = pais_data[pais_data["Date"] > date_train][["ConfirmedCases", "Fatalities"]]

        pred_data = scaler.inverse_transform(pred_data)

        x_new_dates = pais_data[pais_data["Date"] > date_train]["Date"]

        pred_data = pd.DataFrame(pred_data, columns = ['ConfirmedCases', 'Fatalities'])



        x_dates = pais_data[pais_data["Date"] <= date_train]["Date"]

        true_data = pais_data[pais_data["Date"] <= date_train][["ConfirmedCases", "Fatalities"]]

        true_data = scaler.inverse_transform(true_data)

        true_data = pd.DataFrame(true_data, columns = ['ConfirmedCases', 'Fatalities'])



        if i < 2: row = 1

        else: row = 2

        fig.add_trace(go.Scatter(x=x_dates, y=true_data['ConfirmedCases'], name=f'Confirmed Cases ({pais})'), row = row , col = (i % 2)+1)

        fig.add_trace(go.Scatter(x=x_dates, y=true_data['Fatalities'], name=f'Fatalities ({pais})'), row = row, col = (i % 2)+1)

        fig.add_trace(go.Scatter(x=x_new_dates, y=pred_data['ConfirmedCases'], name=f'Predicted Confirmed Cases ({pais})', mode='lines+markers'), row = row, col = (i % 2)+1)

        fig.add_trace(go.Scatter(x=x_new_dates, y=pred_data['Fatalities'], name=f'Predicted Fatalities ({pais})', mode='lines+markers'), row = row, col = (i % 2)+1)



    fig.update_layout(title='Extrapolation for COVID-19', xaxis_title="Date", yaxis_title="New Cases/Fatalities", template = "plotly_dark", width=1400, height=900)

    fig.show()    

    

    

def submission(df_predict, scaler, sanity_check = False):

    sub = df_predict[df_predict["ForecastId"] != 0]

    sub = sub[["ForecastId", "ConfirmedCases", "Fatalities"]]

    sub[["ConfirmedCases", "Fatalities"]] = scaler.inverse_transform(sub[["ConfirmedCases", "Fatalities"]])

    if sanity_check:

        print(sub.shape)

        print(sub.iloc[:50])

    sub.to_csv("submission.csv", index=False)

    return sub



seq_len = 15

df, scaler,date_train, date_test = create_dataset(seq_len, MinMaxScaler(), log=False, cut_zeros= False)
train_df = df[df["Date"] <= date_train]

test_df = df[df["Date"] >= date_test]



days_to_val = 21

date_training_set, x_train_window, x_train_place , y_train ,x_val_window ,x_val_place, y_val = build_train_data(train_df, date_train, days_to_val)
model_name = "Submission"

model = build_nn(x_train_window, x_train_place)

model, history = train([x_train_window, x_train_place], y_train, [x_val_window, x_val_place], y_val, model, model_name, log = False)



loss = min(history.history["val_loss"])

print(f"Min Val Loss: {loss}" )
plot_train_statistics(history, model, model_name, train_df, date_train, days_to_val)

plot_curve_for_countries(["Italy", "Spain", "Germany", "Brazil"], model, model_name, train_df, date_train, days_to_val, scaler, log = False)
df_predict = extrapolation(train_df, test_df, date_train, date_test, seq_len, model, model_name, trend = True)
plot_extrapolation(["Italy", "Spain", "Germany", "Brazil"], df_predict, date_train, scaler)
sub = submission(df_predict, scaler, sanity_check = True)