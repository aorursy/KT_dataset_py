import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
fires_df = pd.read_csv('../input/_visagio-hackathon_/database_fires.csv')



perguntas = pd.read_csv('../input/_visagio-hackathon_/respostas.csv')
len(perguntas) + len(fires_df)
perguntas.head()

perguntas['fires'] = -1
fires_df.head()
fires_df = pd.concat([fires_df, perguntas], ignore_index = True)
fires_df.sample(4)
len(fires_df)
len(fires_df.id.unique())
fires_df.sort_values('data', inplace = True)

fires_df.set_index('id', inplace = True)
fires_df.columns
def clean_dataset(fires_df):

    features = ['precipitacao', 'temp_max', 'temp_min',

       'insolacao', 'evaporacao_piche', 'temp_comp_med', 'umidade_rel_med',

       'vel_vento_med', 'altitude']



    #Fill na with the value on previous day

    fires_df[['estado', 'estacao'] + features] = fires_df[['estado', 'estacao'] + features].groupby(['estado', 'estacao'], as_index = False).apply(lambda x: x.fillna(method = 'ffill'))



    #Fill with mean on 'estado' and 'data'

    fires_df[['estado', 'data'] + features] = fires_df[['estado', 'data'] + features].groupby(['estado', 'data'], as_index = False, group_keys=False).apply(lambda x: x.fillna(x.mean()))



    #Fill with mean on 'estado'

    fires_df[['estado'] + features] = fires_df[['estado'] + features].groupby('estado', as_index = False, group_keys=False).apply(lambda x: x.fillna(x.mean()))



    fires_df[features] = fires_df[['estado'] + features].groupby('estado').transform(lambda x: (x - x.mean())/x.std())

    

    return fires_df[['estado', 'estacao'] + features + ['fires']]
fires_df = clean_dataset(fires_df)
#fires_df_bkp = fires_df.copy()
#fires_df = fires_df_bkp.copy()
features = [feat for feat in fires_df.columns.to_list() if feat not in ['estado', 'estacao', 'fires']]
features
max_lag = 12
list_of_features = features

for lag in range(max_lag):

    new_features = [feat + '_lag_' + str(lag + 1) for feat in features]

    fires_df[new_features] = fires_df.groupby(['estado', 'estacao'])[features].transform(lambda x: x.shift(lag + 1)).fillna(0)

    list_of_features = list_of_features + new_features

features = list_of_features



#fires_df[['umidade_rel_med_next', 'evaporacao_piche_next']] = fires_df.groupby(['estado', 'estacao'])[['umidade_rel_med', 'evaporacao_piche']].transform(lambda x: x.shift(-1)).fillna(0)
fires_df.columns[:28]
def structure_dataset(fires_df):

    train_df = fires_df[fires_df.fires != -1].copy()

    test_df = fires_df[fires_df.fires == -1].copy()

    

    features = fires_df.columns.to_list()

    features = [feat for feat in features if feat not in ['estado', 'fires', 'estacao']]

    X_num_train = train_df[features].values

    X_cat_train = train_df['estado'].values

    cat_dict = {estado: i for i, estado in enumerate(np.unique(X_cat_train))}

    X_cat_train = np.array([cat_dict[estado] for estado in X_cat_train.reshape(-1, )])

    y_train = train_df.fires.values

    

    id_test = np.array(test_df.index)

    X_num_test = test_df[features].values

    X_cat_test = test_df['estado'].values

    X_cat_test = np.array([cat_dict[estado] for estado in X_cat_test.reshape(-1, )])

    

    

    return {'X_train': {'numerical': X_num_train, 'estado': X_cat_train},

            'X_test': {'numerical': X_num_test, 'estado': X_cat_test},

            'train_target': y_train,

            'id_test': id_test}
dataset = structure_dataset(fires_df)
dataset['X_train']['numerical'] = dataset['X_train']['numerical'].reshape((-1, max_lag + 1, 9))

dataset['X_test']['numerical'] = dataset['X_test']['numerical'].reshape((-1, max_lag + 1, 9))
from keras.models import Model, load_model

from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, Dropout, LSTM

from keras.losses import binary_crossentropy

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.initializers import RandomNormal

from keras.optimizers import Adam


hists = []

for i, lr in enumerate([10**(-3)]):

    #Categorical

    categorical_inputs = []

    categorical_inputs.append(Input(shape = [1], name = 'estado'))



    categorical_embeddings = []

    embed_size = len(np.unique(dataset['X_train']['estado']))

    categorical_embeddings.append(Embedding(embed_size, 10)(categorical_inputs[0]))



    categorical_logits = Flatten()(categorical_embeddings[0])



    #Numerical

    lstm_init = RandomNormal(0, 0.1)

    numerical_inputs = Input(shape = (max_lag + 1, 9), name = 'numerical')

    numerical_logits = LSTM(24, input_shape = (max_lag + 1, 9), kernel_initializer = lstm_init)(numerical_inputs)



    #Compile and fit



    logits = Concatenate()([numerical_logits, categorical_logits])

    logits = Dense(128, activation = 'relu')(logits)

    logits = Dropout(0.5)(logits)

    logits = Dense(32, activation = 'relu')(logits)

    logits = Dropout(0.2)(logits)

    



    out = Dense(1, activation = 'sigmoid')(logits)



    model = Model(inputs = categorical_inputs + [numerical_inputs], outputs = out)



    optimizer = Adam(lr = lr)

    model.compile(optimizer = optimizer, loss =  'binary_crossentropy', metrics = ['accuracy'])



    es = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 1, restore_best_weights = True, min_delta = 0.0005)

    mc = ModelCheckpoint(filepath = 'best_model12_att2_lr' + str(lr) + '.h5', save_best_only = True, monitor = 'val_acc')

    

    history = model.fit(dataset['X_train'], (dataset['train_target']>0), epochs = 30, batch_size = 70,

                        validation_split = 0.2, callbacks = [es, mc], verbose = 1)

    hists.append(history)

    

    print(str(i) + ' - ésima iteração com lr = ' + str(round(lr, 6)))
model = load_model('best_model12_lr0.001.h5')
yhat = model.predict(dataset['X_test'], verbose = 1)
np.quantile(yhat, 0.333)
np.quantile(yhat, 0.667)
np.max(hists[0].history['val_acc'])
[np.max(hist.history['val_acc']) for hist in hists]
hist
(yhat>0.333).mean()
yhat = (yhat.reshape(-1, ) > 0.44435).astype(int)
submission_df = pd.DataFrame({'id': dataset['id_test'], 'fires': yhat})
#submission_df.to_csv('../Submissions/LSTM12deep_333_submission.csv', index = False)