
import os 
import sys
import json
import math
import random
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, KFold,  StratifiedKFold

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L

import warnings
warnings.filterwarnings("ignore")

seed = 42
dropout_model = 0.36
hidden_dim_first = 128
hidden_dim_second = 248
hidden_dim_third = 208

commits_df = pd.DataFrame(columns = ['commit_num', 'dropout_model', 'hidden_dim_first', 'hidden_dim_second', 'hidden_dim_third', 'LB_score'])

n=0
commits_df.loc[n,'commit_num'] = 0
commits_df.loc[n,'dropout_model'] = 0.4
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 128
commits_df.loc[n,'hidden_dim_third'] = 128
commits_df.loc[n,'LB_score'] = 0.25883


n=1
commits_df.loc[n,'commit_num'] = 3
commits_df.loc[n,'dropout_model'] = 0.36
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 256
commits_df.loc[n,'hidden_dim_third'] = 128
commits_df.loc[n,'LB_score'] = 0.25855


n=2
commits_df.loc[n,'commit_num'] = 4
commits_df.loc[n,'dropout_model'] = 0.36
commits_df.loc[n,'hidden_dim_first'] = 256
commits_df.loc[n,'hidden_dim_second'] = 256
commits_df.loc[n,'hidden_dim_third'] = 128
commits_df.loc[n,'LB_score'] = 0.25887



n=3
commits_df.loc[n,'commit_num'] = 5
commits_df.loc[n,'dropout_model'] = 0.36
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 384
commits_df.loc[n,'hidden_dim_third'] = 128
commits_df.loc[n,'LB_score'] = 0.25880


n=4
commits_df.loc[n,'commit_num'] = 6
commits_df.loc[n,'dropout_model'] = 0.36
commits_df.loc[n,'hidden_dim_first'] = 384
commits_df.loc[n,'hidden_dim_second'] = 128
commits_df.loc[n,'hidden_dim_third'] = 128
commits_df.loc[n,'LB_score'] = 0.25989


n=5
commits_df.loc[n,'commit_num'] = 7
commits_df.loc[n,'dropout_model'] = 0.36
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 192
commits_df.loc[n,'hidden_dim_third'] = 128
commits_df.loc[n,'LB_score'] = 0.25877


n=6
commits_df.loc[n,'commit_num'] = 8
commits_df.loc[n,'dropout_model'] = 0.36
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 224
commits_df.loc[n,'hidden_dim_third'] = 128
commits_df.loc[n,'LB_score'] = 0.25868

n=7
commits_df.loc[n,'commit_num'] = 9
commits_df.loc[n,'dropout_model'] = 0.36
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 288
commits_df.loc[n,'hidden_dim_third'] = 128
commits_df.loc[n,'LB_score'] = 0.25881


n=8
commits_df.loc[n,'commit_num'] = 10
commits_df.loc[n,'dropout_model'] = 0.36
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 248
commits_df.loc[n,'hidden_dim_third'] = 128
commits_df.loc[n,'LB_score'] = 0.25850

n=9
commits_df.loc[n,'commit_num'] = 11
commits_df.loc[n,'dropout_model'] = 0.36
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 240
commits_df.loc[n,'hidden_dim_third'] = 128
commits_df.loc[n,'LB_score'] = 0.25868


n=10
commits_df.loc[n,'commit_num'] = 12
commits_df.loc[n,'dropout_model'] = 0.35
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 248
commits_df.loc[n,'hidden_dim_third'] = 128
commits_df.loc[n,'LB_score'] = 0.25884

n=11
commits_df.loc[n,'commit_num'] = 13
commits_df.loc[n,'dropout_model'] = 0.3
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 248
commits_df.loc[n,'hidden_dim_third'] = 128
commits_df.loc[n,'LB_score'] = 0.25879

n=12
commits_df.loc[n,'commit_num'] = 15
commits_df.loc[n,'dropout_model'] = 0.45
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 248
commits_df.loc[n,'hidden_dim_third'] = 128
commits_df.loc[n,'LB_score'] = 0.25920

n=13
commits_df.loc[n,'commit_num'] = 17
commits_df.loc[n,'dropout_model'] = 0.37
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 248
commits_df.loc[n,'hidden_dim_third'] = 128
commits_df.loc[n,'LB_score'] = 0.25885

n=14
commits_df.loc[n,'commit_num'] = 18
commits_df.loc[n,'dropout_model'] = 0.36
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 248
commits_df.loc[n,'hidden_dim_third'] = 248
commits_df.loc[n,'LB_score'] = 0.25841


n=15
commits_df.loc[n,'commit_num'] = 19
commits_df.loc[n,'dropout_model'] = 0.36
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 248
commits_df.loc[n,'hidden_dim_third'] = 384
commits_df.loc[n,'LB_score'] = 0.25956

n=16
commits_df.loc[n,'commit_num'] = 20
commits_df.loc[n,'dropout_model'] = 0.36
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 128
commits_df.loc[n,'hidden_dim_third'] = 248
commits_df.loc[n,'LB_score'] = 0.25891

n=17
commits_df.loc[n,'commit_num'] = 21
commits_df.loc[n,'dropout_model'] = 0.36
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 248
commits_df.loc[n,'hidden_dim_third'] = 240
commits_df.loc[n,'LB_score'] = 0.25831

n=18
commits_df.loc[n,'commit_num'] = 22
commits_df.loc[n,'dropout_model'] = 0.36
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 248
commits_df.loc[n,'hidden_dim_third'] = 216
commits_df.loc[n,'LB_score'] = 0.25827

n=19
commits_df.loc[n,'commit_num'] = 23
commits_df.loc[n,'dropout_model'] = 0.36
commits_df.loc[n,'hidden_dim_first'] = 128
commits_df.loc[n,'hidden_dim_second'] = 248
commits_df.loc[n,'hidden_dim_third'] = 184
commits_df.loc[n,'LB_score'] = 0.25898


commits_df['LB_score'] = pd.to_numeric(commits_df['LB_score'])
commits_df['best'] = 0
commits_df.loc[commits_df['LB_score'].idxmin(), 'best'] = 1

commits_df.sort_values(by=['LB_score'])


fig = px.scatter_3d(commits_df, x='hidden_dim_first', y='hidden_dim_second', z='LB_score', color = 'best', 
                    symbol = 'dropout_model',
                    title='hidden_dim_1st & 2nd and LB score visualization of COVID-19 mRNA VDP solutions')
fig.update(layout=dict(title=dict(x=0.07)))


# Interactive plot with results of parameters tuning
fig = px.scatter_3d(commits_df, x='hidden_dim_second', y='dropout_model', z='LB_score', color = 'best', 
                    symbol = 'hidden_dim_first',
                    title='hidden_dim_2nd & dropout and LB score visualization of COVID-19 mRNA VDP solutions')
fig.update(layout=dict(title=dict(x=0.07)))



fig = px.scatter_3d(commits_df, x='hidden_dim_second', y='hidden_dim_third', z='LB_score', color = 'best', 
                    symbol = 'hidden_dim_first',
                    title='hidden_dim_2nd & 3nd and LB score visualization of COVID-19 mRNA VDP solutions')
fig.update(layout=dict(title=dict(x=0.07)))
def Funcs(train,test,sample_sub):

    trainNew = train.iloc[:2000,:]
    testNew = train.loc[2000:,test.columns]
    
    test2= testNew.reset_index().iloc[:,1:]
    test = pd.concat([test2,test.iloc[3631:3633,:]])
    
    test= test.reset_index().iloc[:,1:]
    test.iloc[:,0] = list(range(0,402))
    
    
    

    v=[]

    for i in range(0,400):
        a = test.id.iloc[i] + '_'
        for j in range(0,107):
            #v = a + str(j)
            v.append(a + str(j))
            
    for i in range(400,402):
        a = test.id.iloc[i] + '_'
        for j in range(0,130):
            #v = a + str(j)
            v.append(a + str(j))
        
        


    sample_subNew = sample_sub[:len(v)]
    sample_subNew.iloc[:,0] = v
    
    sample_sub = sample_subNew
    train = trainNew

    
    return(train,test,sample_sub)
for ddd in range(0,1):

    train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)
    test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)
    sample_sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")    
    
    
   

    if ddd == 0:
        S = pd.concat([train.iloc[2000:],train.iloc[:2000,:]])
        S.iloc[:,0] = list(range(0,2400))
        train = S.reset_index().iloc[:,1:]

    
    if ddd == 1:
        S = pd.concat([train.iloc[1600:],train.iloc[:1600,:]])
        S.iloc[:,0] = list(range(0,2400))
        train = S.reset_index().iloc[:,1:]
        
    if ddd == 2:
        S = pd.concat([train.iloc[1200:],train.iloc[:1200,:]])
        S.iloc[:,0] = list(range(0,2400))
        train = S.reset_index().iloc[:,1:]
        
    if ddd == 3:
        S = pd.concat([train.iloc[800:],train.iloc[:800,:]])
        S.iloc[:,0] = list(range(0,2400))
        train = S.reset_index().iloc[:,1:]
    
    if ddd == 4:
        S = pd.concat([train.iloc[400:],train.iloc[:400,:]])
        S.iloc[:,0] = list(range(0,2400))
        train = S.reset_index().iloc[:,1:]
    
        
        
        
        
    train , test ,sample_sub = Funcs(train,test,sample_sub)
    
    target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
    
    token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

    def get_pair_index_structure(structure):
        structure = np.array([struc for struc in structure], dtype="<U4")

        open_index = np.where(structure == "(")[0]
        closed_index = np.where(structure == ")")[0]

        structure[open_index] = range(0, len(open_index))
        structure[closed_index] = range(len(open_index)-1, -1, -1)
        structure[structure == "."] = -1
        structure = structure.astype(int)

        pair_structure = np.array([-1]*len(structure))
        for i in range(len(open_index)):
            start, end = np.where(structure == i)[0]
            pair_structure[start] = end
            pair_structure[end] = start    

        return pair_structure
    
    
    def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
        return np.transpose(
            np.array(
                df[cols]
                .applymap(lambda seq: [token2int[x] for x in seq])
                .values
                .tolist()
            ),
            (0, 2, 1)
        )

    train_inputs_all = preprocess_inputs(train)
    train_labels_all = np.array(train[target_cols].values.tolist()).transpose((0, 2, 1))
    
    
    
    
    def MCRMSE(y_true, y_pred):
        colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
        return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)

    def gru_layer(hidden_dim, dropout):
        return tf.keras.layers.Bidirectional(
                                    tf.keras.layers.GRU(hidden_dim,
                                    dropout=dropout,
                                    return_sequences=True,
                                    kernel_initializer = 'orthogonal'))

    def lstm_layer(hidden_dim, dropout):
        return tf.keras.layers.Bidirectional(
                                    tf.keras.layers.LSTM(hidden_dim,
                                    dropout=dropout,
                                    return_sequences=True,
                                    kernel_initializer = 'orthogonal'))

    def build_model(model_type=1, seq_len=107, pred_len=68, embed_dim=100, 
                    dropout=dropout_model, hidden_dim_first = hidden_dim_first, 
                    hidden_dim_second = hidden_dim_second, hidden_dim_third = hidden_dim_third):

        inputs = tf.keras.layers.Input(shape=(seq_len, 3))

        embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)
        reshaped = tf.reshape(
            embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))

        reshaped = tf.keras.layers.SpatialDropout1D(.2)(reshaped)

        if model_type == 0:
            hidden = gru_layer(hidden_dim_first, dropout)(reshaped)
            hidden = gru_layer(hidden_dim_second, dropout)(hidden)
            hidden = gru_layer(hidden_dim_third, dropout)(hidden)

        elif model_type == 1:
            hidden = lstm_layer(hidden_dim_first, dropout)(reshaped)
            hidden = lstm_layer(hidden_dim_second, dropout)(hidden)
            hidden = lstm_layer(hidden_dim_third, dropout)(hidden)

        elif model_type == 2:
            hidden = gru_layer(hidden_dim_first, dropout)(reshaped)
            hidden = lstm_layer(hidden_dim_second, dropout)(hidden)
            hidden = lstm_layer(hidden_dim_third, dropout)(hidden)

        elif model_type == 3:
            hidden = lstm_layer(hidden_dim_first, dropout)(reshaped)
            hidden = gru_layer(hidden_dim_second, dropout)(hidden)
            hidden = gru_layer(hidden_dim_third, dropout)(hidden)

        elif model_type == 4:
            hidden = lstm_layer(hidden_dim_first, dropout)(reshaped)
            hidden = gru_layer(hidden_dim_second, dropout)(hidden)
            hidden = lstm_layer(hidden_dim_third, dropout)(hidden)

        truncated = hidden[:, :pred_len]

        out = tf.keras.layers.Dense(5, activation='linear')(truncated)

        model = tf.keras.Model(inputs=inputs, outputs=out)

        adam = tf.optimizers.Adam()
        model.compile(optimizer=adam, loss=MCRMSE)

        return model
    
    
    def train_and_predict(n_folds=5, model_name="model", model_type=0, epochs=90, debug=False,
                          dropout_model=dropout_model, hidden_dim_first = hidden_dim_first, 
                          hidden_dim_second = hidden_dim_second, hidden_dim_third = hidden_dim_third,
                          seed=seed):

        print("Model:", model_name)

        ensemble_preds = pd.DataFrame(index=sample_sub.index, columns=target_cols).fillna(0) # test dataframe with 0 values
        kf = KFold(n_folds, shuffle=True, random_state=seed)
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        val_losses = []
        historys = []

        for i, (train_index, val_index) in enumerate(skf.split(train_inputs_all, train['SN_filter'])):
            print("Fold:", str(i+1))

            model_train = build_model(model_type=model_type, 
                                      dropout=dropout_model, 
                                      hidden_dim_first = hidden_dim_first, 
                                      hidden_dim_second = hidden_dim_second, 
                                      hidden_dim_third = hidden_dim_third)
            model_short = build_model(model_type=model_type, seq_len=107, pred_len=107,
                                      dropout=dropout_model, 
                                      hidden_dim_first = hidden_dim_first, 
                                      hidden_dim_second = hidden_dim_second, 
                                      hidden_dim_third = hidden_dim_third)
            model_long = build_model(model_type=model_type, seq_len=130, pred_len=130,
                                     dropout=dropout_model, 
                                     hidden_dim_first = hidden_dim_first, 
                                     hidden_dim_second = hidden_dim_second, 
                                     hidden_dim_third = hidden_dim_third)

            train_inputs, train_labels = train_inputs_all[train_index], train_labels_all[train_index]
            val_inputs, val_labels = train_inputs_all[val_index], train_labels_all[val_index]

            checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{model_name}.h5')

            history = model_train.fit(
                train_inputs , train_labels, 
                validation_data=(val_inputs,val_labels),
                batch_size=64,
                epochs=epochs, # changed 70
                callbacks=[tf.keras.callbacks.ReduceLROnPlateau(), checkpoint],
                verbose=2 if debug else 0
            )

            print(f"{model_name} Min training loss={min(history.history['loss'])}, min validation loss={min(history.history['val_loss'])}")

            val_losses.append(min(history.history['val_loss']))
            historys.append(history)

            model_short.load_weights(f'{model_name}.h5')
            model_long.load_weights(f'{model_name}.h5')

            public_preds = model_short.predict(public_inputs)
            private_preds = model_long.predict(private_inputs)

            preds_model = []
            for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
                for i, uid in enumerate(df.id):
                    single_pred = preds[i]

                    single_df = pd.DataFrame(single_pred, columns=target_cols)
                    single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

                    preds_model.append(single_df)

            preds_model_df = pd.concat(preds_model)
            ensemble_preds[target_cols] += preds_model_df[target_cols].values / n_folds

            if debug:
                print("Intermediate ensemble result")
                print(ensemble_preds[target_cols].head())

        ensemble_preds["id_seqpos"] = preds_model_df["id_seqpos"].values
        ensemble_preds = pd.merge(sample_sub["id_seqpos"], ensemble_preds, on="id_seqpos", how="left")

        print("Mean Validation loss:", str(np.mean(val_losses)))

        if debug:
            fig, ax = plt.subplots(1, 1, figsize = (10, 5))
            for i, history in enumerate(historys):
                ax.plot(history.history['loss'])
                ax.plot(history.history['val_loss'])
                ax.set_title('model_'+str(i+1))
                ax.set_ylabel('Loss')
                ax.set_xlabel('Epoch')
            plt.show()

        return ensemble_preds


    public_df = test.query("seq_length == 107").copy()
    private_df = test.query("seq_length == 130").copy()
    public_inputs = preprocess_inputs(public_df)
    private_inputs = preprocess_inputs(private_df)

    ensembles = []

    for i in range(5):
        model_name = "model_"+str(i+1)

        ensemble = train_and_predict(n_folds=5, model_name=model_name, model_type=i, epochs=100,
                                     dropout_model=dropout_model, hidden_dim_first = hidden_dim_first, 
                                     hidden_dim_second = hidden_dim_second, hidden_dim_third = hidden_dim_third,
                                     seed=seed)
        ensembles.append(ensemble)
    
    ensemble_final = ensembles[0].copy()
    ensemble_final[target_cols] = 0

    for ensemble in ensembles:
        ensemble_final[target_cols] += ensemble[target_cols].values / len(ensembles)

    
    if ddd == 0:
        ensemble_final.to_csv('Inputs2_2.csv', index=False)
    if ddd == 1:
        ensemble_final.to_csv('Inputs2_3.csv', index=False)
    if ddd == 2:
        ensemble_final.to_csv('Inputs2_4.csv', index=False)
    if ddd == 3:
        ensemble_final.to_csv('Inputs2_5.csv', index=False)
    if ddd == 4:
        ensemble_final.to_csv('Inputs2_6.csv', index=False)