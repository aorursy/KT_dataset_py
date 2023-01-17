#Cargo librerias generales

import gc

import numpy as np 

import pandas as pd

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt



#Modulos de Keras/TF

from keras.models import Sequential, Model

from keras.layers import (Conv1D, AveragePooling1D, Reshape, Dense, Dropout, Embedding, Input,  Flatten, Concatenate, Multiply, RepeatVector, Permute)

from keras.preprocessing.sequence import pad_sequences

from keras.callbacks import EarlyStopping

from keras.optimizers import Adam

from sklearn.metrics import roc_auc_score

import tensorflow as tf

from keras import backend as k

import os

print(os.listdir("../input/banco-galicia-dataton-2019"))

print(os.listdir("../input/galicia-nn-precalculado"))
recalculate_datasets = False

#Define la cantidad de eventos maximos a considerar

history = 1500





# Genero algunos features temporales

def preproc(pv):    

    pv["DIA_SEM"]=pv["FEC_EVENT"].dt.dayofweek

    pv["DIA_ANIO"]=pv["FEC_EVENT"].dt.dayofyear

    pv["HORA"]=pv["FEC_EVENT"].dt.hour

    pv["mes"]=pv["FEC_EVENT"].dt.month

    pv["OFFSET"] = (pv["FEC_EVENT"] - pd.to_datetime('01-01-2019')).dt.days

    pv["OFFSET_MES"] = [int(a) for a in -pv["OFFSET"]/24]

    return(pv)





if recalculate_datasets:

    #Leo eventos de navegacion en sitio del banco

    data = preproc(pd.concat([

           pd.read_csv("./datasets/pageviews.zip" ,parse_dates=["FEC_EVENT"]),

           pd.read_csv("./datasets/pageviews_complemento.zip", parse_dates=["FEC_EVENT"])

    ]))





    #Leo las conversiones

    y_prev = pd.read_csv("./datasets/conversiones.zip")



    #Defino una secuencia del largo maximo nula. Es para que pad_sequences no genere un tensor de menor dimension

    #Si en el intervalo tratado no hay ningun caso que llegue al tamaño maximo

    dummyseq = pd.Series([[0 for i in range(history)],[]])[0:1]



    #Genero todos los intervalos desde [1] para 2 hasta [1,2...11] para 12 mas set de predicción [1...12] 

    for mes_corte in range(2,14):



        print("Genero dataset 1 a ", mes_corte - 1, " con objetivo ", mes_corte)



        #determino el offset en días para el intervalo que se esta generando. Este offset se usa para ajustar 

        #la cantidad de dias previas al mes objetivo para los eventos tratados

        offset_dias_corte = (13 - mes_corte) * 30



        #Se genera un dataset vacío para todos los clientes de los que hay historia

        y_train = pd.Series(0, index=sorted(data[data['mes']<mes_corte].USER_ID.unique()))



        #De los que tengo info de movimientos determino los que se convirtieron

        convertidos = np.intersect1d(y_prev[y_prev.mes == mes_corte].USER_ID.unique(),data[data['mes']<mes_corte].USER_ID.unique())        



        #LEs pongo objetivo positivo a los convertidos

        y_train.loc[convertidos] = 1



        #inicializo para la primera feature

        train = None



        #Genero un dataset con todos los features en secuencia. Cada uno se trata en un embedding separado

        for field in ["PAGE","CONTENT_CATEGORY", "CONTENT_CATEGORY_TOP", "CONTENT_CATEGORY_BOTTOM", "SITE_ID", "ON_SITE_SEARCH_TERM", "DIA_SEM", "HORA"]:



            trainfd = data[data['mes']<mes_corte].groupby("USER_ID").apply(lambda x: x[(x.FEC_EVENT.dt.month < mes_corte)]\

                                              .sort_values("FEC_EVENT")[field][-history:].values)



            #Agrego una secuencia dummy del largo maximo para que no quede una matriz menor

            trainfd = pd.concat([dummyseq,trainfd])

            trainfd = pad_sequences(trainfd)[:,:,np.newaxis]

            trainfd = trainfd[1:]



            if train is None:

                train = np.copy(trainfd)    

            else:

                train =  np.concatenate((train,trainfd),axis=2)





        #Determino adicionalmente la cantidad de dias hasta el corte del intervalo

        train_day_event  = data[data['mes']<mes_corte].groupby("USER_ID").apply(lambda x: x[(x.FEC_EVENT.dt.month < mes_corte)]\

                                              .sort_values("FEC_EVENT")["OFFSET"][-history:].values)



        train_day_event = (train_day_event + offset_dias_corte)



        train_day_event = pad_sequences(train_day_event,maxlen =history)[:,:,np.newaxis]



        train =  np.concatenate((train,train_day_event),axis=2)



        #Primer intervalo inicializo las matrices de train

        if mes_corte==2:

            prev_X = np.copy(train)

            prev_y = np.copy(y_train)



        #El ultimo intervalo es para predecir solamente    

        elif mes_corte == 13:

            y_test = np.copy(y_train)

            test = np.copy(train)

            print("test" ,len(test), len(y_test))               



        #Siguientes intervalos agrego a la matriz existente 

        else:

            prev_X = np.concatenate((train,prev_X ), axis = 0)

            prev_y = np.concatenate((y_train,prev_y ), axis = 0)        



    train = prev_X

    y_train = prev_y



    np.savez_compressed("../input/galicia-nn-precalculado/NNv11precomputed1500.npz", train=train, y_train = y_train, test = test, y_test=y_test)  
data = np.load("../input/galicia-nn-precalculado/NNv11precomputed1500.npz")



crop_hist = 500



train = data["train"][:,-crop_hist:,:]

y_train = data["y_train"]

test = data["test"][:,-crop_hist:,:]

y_test = data["y_test"]



del data

gc.collect()

def failsafe_auc(y_true, y_pred):

    if y_true.sum() == 0:

        return(0.5)

    else:

        return(roc_auc_score(y_true,y_pred))

    



def auroc(y_true, y_pred):

    return tf.py_func(failsafe_auc, (y_true, y_pred), tf.double)



def getModel():

    emb_depth = [64,16,16,16,8,32,32,8]

    input_layers = []

    embedded_lay = []

    for emb_inp in range(train.shape[2]-1):  ##### UNA NO EMBEDDING

        input_layers.append(Input(shape=(train.shape[1],), dtype='int32'))

        embedded_lay.append(Embedding(input_dim=(train[:,:,emb_inp].max() + 1),output_dim=emb_depth[emb_inp], 

                                  input_length=train.shape[1])(input_layers[-1]))

        



    

    #embedded_lay.append(Reshape((train.shape[1],1))(input_layers[-1]))

    conc_emb = Concatenate()(embedded_lay)

    

    #input_layer = Input(shape=(train.shape[1],), dtype='int32')

    input_layers.append(Input(shape=(train.shape[1],), dtype='float32'))

    

    h_factors = RepeatVector(np.sum(emb_depth))(input_layers[-1])

    #print(h_factors.shape)

    h_factors = Permute((2, 1))(h_factors)

    #print(h_factors.shape)    

    #h_factors = k.repeat_elements(h_factors,np.sum(emb_depth),2)

    #print(h_factors.shape,conc_emb.shape )

    

    conc_emb = Multiply()([h_factors,conc_emb])

    

    #print(conc_emb.shape)

    x = Conv1D(64, 1, activation="relu")(conc_emb)

    x = AveragePooling1D(5)(x)

    x = Dropout(0.3)(x)

    x = Conv1D(32, 1, activation="relu")(x)

    x = AveragePooling1D(5)(x)

    x = Dropout(0.1)(x)    

    x = Conv1D(4, 1, activation="relu")(x)

    x = AveragePooling1D(5)(x)

    x = Dropout(0.05)(x)    

    x = Flatten()(x)

    x = Dense(128, activation="relu")(x)

    x = Dropout(0.05)(x)

    x = Dense(64, activation="relu")(x)

    x = Dropout(0.05)(x)    

    x = Dense(16, activation="relu")(x)

    x = Dropout(0.05)(x)    

    x = Dense(4, activation="relu")(x)

    out_layer = Dense(1, activation="sigmoid")(x)



    #print(input_layers)

    

    model = Model(inputs=input_layers, outputs=out_layer)



    return(model)

    




def input_list(tensor, p_f_factor):

    lst = []

    for emb_in in range(tensor.shape[2]-1):

        lst.append(tensor[:,:,emb_in])

        

    lst.append(np.exp(tensor[:,:,-1]/p_f_factor).astype(float))

    return(lst)



lr, ffactor = 0.0017, 157







test_probs = []

i = 0



loss_log = []



for train_idx, valid_idx in KFold(n_splits=5, shuffle=True, random_state=42 ).split(train):



    Xt = train[train_idx]

    yt = y_train[train_idx]



    Xv = train[valid_idx]

    yv = y_train[valid_idx]



    loss_log_best = 0



    for iter in range(3):

        model = getModel()

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=[auroc])

        #model.summary()



        model.fit(input_list(Xt,ffactor), yt, validation_data= (input_list(Xv,ffactor),yv), batch_size=265, epochs=1, verbose=1,

                    shuffle=True, class_weight={0:1, 1:86})



        if roc_auc_score(yv,model.predict(input_list(Xv,ffactor))[:, -1]) > 0.6:



            model.fit(input_list(Xt,ffactor), yt, validation_data= (input_list(Xv,ffactor),yv), batch_size=265, epochs=150, verbose=1,

                        shuffle=True, class_weight={0:1, 1:86},

                        callbacks=[EarlyStopping(monitor='val_auroc', patience=5, verbose=1,

                                                 mode="max", restore_best_weights=True)])



        loss_log_current = roc_auc_score(yv,model.predict(input_list(Xv,ffactor))[:, -1])



        if loss_log_current > loss_log_best:

            loss_log_best = loss_log_current

            test_probs_best = pd.Series(model.predict(input_list(test,ffactor))[:, -1], name="fold_" + str(i))





    loss_log.append(loss_log_best)

    test_probs.append(test_probs_best)



    i+=1



test_probs = pd.concat(test_probs, axis=1).mean(axis=1)

test_probs.index.name="USER_ID"

test_probs.name="SCORE"



test_probs.to_csv("./Final-NN-v11.csv", header=True)



print("Loss: ", np.mean(loss_log) )        


