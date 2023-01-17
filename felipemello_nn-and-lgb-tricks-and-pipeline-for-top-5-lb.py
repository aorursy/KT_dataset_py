import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from keras.layers import Dense, Input, Activation, Concatenate

from keras.layers import BatchNormalization,Add,Dropout

from keras.optimizers import Adam

from keras.models import Model, load_model

from keras import callbacks

from keras import backend as K

import warnings

import os

import lightgbm as lgb

import copy

import gc

import random

warnings.filterwarnings("ignore")

warnings.filterwarnings(action="ignore",category=DeprecationWarning)

warnings.filterwarnings(action="ignore",category=FutureWarning)
def calc_logmae(y_val, y_pred):

    

    return np.log(np.sum(np.abs(y_val - y_pred)/len(y_pred)))





def permutation_importance(model, X_val, y_val, calc_logmae, threshold=0.01,

                           minimize=True, verbose=True):

    results = {}

    

    y_pred = model.predict(X_val)

    

    results['base_score'] = calc_logmae(y_val, y_pred)

    if verbose:

        print(f'Base score {results["base_score"]:.5}')



    

    for col in tqdm(X_val.columns):

        freezed_col = X_val[col].copy()



        X_val[col] = np.random.permutation(X_val[col])

        preds = model.predict(X_val)

        results[col] = calc_logmae(y_val, preds)



        X_val[col] = freezed_col

        

        if verbose:

            print(f'column: {col} - {results[col]:.5}')

 

            

    if minimize:

        bad_features = [k for k in results if results[k] < results['base_score'] + threshold]

    else:

        bad_features = [k for k in results if results[k] > results['base_score'] + threshold]

    bad_features.remove('base_score')

    

    return results, bad_features



def load_lgb_params(mol_type, n_estimators = 2000):

    

    seed = 2319

    param_1J = {

        'num_leaves': int(0.7*(25**2)), #int(0.7*(25**2))

        'learning_rate': 0.1,

        'feature_fraction': 1,

        'save_binary': True, 

        'seed': seed,

        'feature_fraction_seed': seed,

        'bagging_seed': seed,

        'drop_seed': seed,

        'data_random_seed': seed,

        'objective': 'regression_l2',

        'boosting_type': 'gbdt',

        'verbosity': -1,

        'metric': 'mae',

        'is_unbalance': True,

        'boost_from_average': 'false',   

    #    'device': 'gpu',

    #    'gpu_platform_id': 0,

    #    'gpu_device_id': 0,

        'bagging_fraction': 1,

         'bagging_freq': 0,

         'lambda_l1': 0.5,

         'lambda_l2': 1.7244553699717466,

         'max_bin': 238, #238

         'max_depth': 25, #int(25)

    #     'min_data_in_leaf': int(203.49294923362797),

    #     'min_gain_to_split': 0.0665822332705641,

    #     'min_sum_hessian_in_leaf': 11.250160554801903,

         'n_estimators': n_estimators,

         'sparse_threshold': 1.0,

          'n_jobs': 6}  

    

    param_2J = {

        'num_leaves': int(1*(20**2)),

        'learning_rate': 0.1,

        'feature_fraction': 1,

        'save_binary': True, 

        'seed': seed,

        'feature_fraction_seed': seed,

        'bagging_seed': seed,

        'drop_seed': seed,

        'data_random_seed': seed,

        'objective': 'regression_l2',

        'boosting_type': 'gbdt',

        'verbosity': -1,

        'metric': 'mae',

        'is_unbalance': True,

        'boost_from_average': 'false',   

    #    'device': 'gpu',

    #    'gpu_platform_id': 0,

    #    'gpu_device_id': 0,

        'bagging_fraction': 1,

         'bagging_freq': int(0),

         'lambda_l1': 1,

         'lambda_l2': 1.89,

         'max_bin': 255, #255

         'max_depth': 20, #int(20)

         'min_data_in_leaf': int(10),

         'min_gain_to_split': 0,

         'min_sum_hessian_in_leaf': 1/869,

         'n_estimators': n_estimators,

         'sparse_threshold': 1.0,

         'n_jobs':6}  

    

    

    param_3J = {

        'num_leaves': int(1*(30**2)),

        'learning_rate': 0.1,

        'feature_fraction': 1,

        'save_binary': True, 

        'seed': seed,

        'feature_fraction_seed': seed,

        'bagging_seed': seed,

        'drop_seed': seed,

        'data_random_seed': seed,

        'objective': 'regression_l2',

        'boosting_type': 'gbdt',

        'verbosity': -1,

        'metric': 'mae',

        'is_unbalance': True,

        'boost_from_average': 'false',   

    #    'device': 'gpu',

    #    'gpu_platform_id': 0,

    #    'gpu_device_id': 0,

        'bagging_fraction': 1,

         'bagging_freq': int(0),

         'lambda_l1': 0.5,

         'lambda_l2': 1,

         'max_bin': 50, #50

         'max_depth': 20, #int(30)

         'min_data_in_leaf': int(10),

         'min_gain_to_split': 0,

         'min_sum_hessian_in_leaf': 1/202,

         'n_estimators': n_estimators,

         'sparse_threshold': 1.0,

         'n_jobs':6}  



    if mol_type[0] == '1':

        params = param_1J

    if mol_type[0] == '2':

        params = param_2J

    if mol_type[0] == '3':

        params = param_3J

        

    return params

        

def create_nn_model(input_shape):

    inp = Input(shape=(input_shape,))

    

    x = Dense(2048, activation="relu")(inp)

    x = BatchNormalization()(x)

    x = Dense(1024, activation="relu")(x)

    x = BatchNormalization()(x)

    x = Dense(512, activation="relu")(x)

    x = BatchNormalization()(x)

    x = Dense(32, activation="relu")(x)

    x = BatchNormalization()(x)

 

    out = Dense(5, activation="linear")(x)  



    model = Model(inputs=inp, outputs=out)

    return model



def plot_history(history, label):

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Loss for %s' % label)

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    _= plt.legend(['Train','Validation'], loc='upper left')

    plt.show()

    

def change_dists_to_yukawa(df, features):

    

    df[features] = np.exp(df[features]) / df[features]

    df.replace(np.inf, 0, inplace=True)



    return df



def get_folds(train, k_folds = 5, val_data_ratio_if_no_kfolds = 0.2, verbose = True):

        

    #get all molecules names

    molecules_names = train['molecule_name'].unique()

    random.shuffle(molecules_names)

    n_molecules = len(molecules_names)



    molecules_folds = []

    index_folds = []

    

    for k in range(1, k_folds+1):

        if k_folds > 1:

            ratio = 1/k_folds

        elif k_folds == 1:

            ratio = val_data_ratio_if_no_kfolds

        start_index = int(np.round(n_molecules*ratio*(k-1)))

        end_index = int(np.round(n_molecules*ratio*(k)))

        

        train_molecules = list(molecules_names[:start_index]) + list(molecules_names[end_index:])

        val_molecules = list(molecules_names[start_index:end_index])

        molecules_folds.append([train_molecules, val_molecules])

        

        index_folds.append([train[train['molecule_name'].isin(train_molecules)].index, train[train['molecule_name'].isin(val_molecules)].index])

       

        if verbose:

            print('-------------')

            print(f'fold {k}')

            print('validation molecules indices go from', start_index, 'to', end_index)

            print(f'{len(train_molecules)} train molecules and {len(val_molecules)} validation molecules')

            print(f'{len(index_folds[-1][0])} train samples and {len(index_folds[-1][1])} validation samples')



    return index_folds

    

def preprocess_train_data(mol_type, train, scalar_coupling_contributions, k_folds=5, val_data_ratio_if_no_kfolds = 0.2, verbose = True):

            

    seed = 2319

    random.seed(seed)  

    

    #randomize data

    train = train.sample(frac=1, random_state = seed).reset_index(drop=True)    

    

    #create folds based on molecules

    index_folds = get_folds(train, k_folds, val_data_ratio_if_no_kfolds, verbose)

    

        

    return train, index_folds



def split_train_and_val_data(train, trn_idx, val_idx, mol_features):

            

    X_train = train.loc[trn_idx, mol_features]

    X_val = train.loc[val_idx, mol_features]

    y_train = train.loc[trn_idx, ['scalar_coupling_constant', 'fc', 'sd','pso', 'dso']]

    y_val = train.loc[val_idx, ['scalar_coupling_constant', 'fc', 'sd','pso', 'dso']]

    

    std_scaler = StandardScaler().fit(train[mol_features])

    

    X_t = std_scaler.transform(X_train.values)

    X_v = std_scaler.transform(X_val.values)

    

    return X_t, X_v, y_train, y_val

     

def load_or_create_nn_model(X_t, k_fold, k_folds, file_folder, load_existing_model = True):

            

    if load_existing_model:

        try:

            nn_model = load_model(f'nn_model_{mol_type}_{str(k_fold)}')

        except:

            nn_model = create_nn_model(X_t.shape[1])

    else:

        nn_model = create_nn_model(X_t.shape[1])



    return nn_model



def select_best_features(file_folder, train, scalar_coupling_contributions):

        

    train, index_folds = preprocess_train_data(mol_type, train, scalar_coupling_contributions, 

                                  k_folds=1, val_data_ratio_if_no_kfolds = 0.05, verbose = True)

    

    trn_idx, val_idx = index_folds[0]

    

    mol_features = [c for c in train.columns if (c not in ['scalar_coupling_constant', 'id', 'molecule_name', 'type', 'aromaticity_vec_1', 'aromaticity_vec_0', 'fc', 'sd','pso', 'dso'])]

    

    X_t, X_v, y_t, y_v = split_train_and_val_data(train, trn_idx, val_idx, mol_features)

    

    lgb_params = load_lgb_params(mol_type, n_estimators = 1000)

    

    lgb_model = lgb.LGBMRegressor(**lgb_params)

    lgb_model.fit(X_t, y_t['scalar_coupling_constant'],

            eval_set=[(X_t, y_t['scalar_coupling_constant']),(X_v, y_v['scalar_coupling_constant'])],

              eval_metric='mae', verbose=100, early_stopping_rounds= 250)

    

    lgb_pred = lgb_model.predict(X_v, num_iteration=lgb_model.best_iteration_)

    lgb_score = calc_logmae(y_v['scalar_coupling_constant'], lgb_pred)

    

    print(f'Inital LGB_score --> logmae for {mol_type} is {lgb_score}')

    

    results, bad_features = permutation_importance(model = lgb_model, X_val = pd.DataFrame(X_v, columns = mol_features), y_val = y_v['scalar_coupling_constant'],

                                                   calc_logmae = calc_logmae, threshold=0.01, minimize=True, verbose=True)



    print(f'{len(bad_features)} were removed from {len(mol_features)} initial features')

    

    mol_features = [feat for feat in mol_features if (feat not in bad_features)]

    

    return mol_features

         

def get_patience_dict(max_number_epochs = 30, min_number_epochs = 3):

    

    """

    Patience is used to determine how many epochs can an NN run on PLateau.

    1JHN is the smallest dataset with 43363 datasamples.

    The number os epochs is defined for 1JHN, and every other patience is proportional to that

    """

    

    patience_dict = {'1JHN':max(min_number_epochs, int(max_number_epochs*43363/43363)), '1JHC':max(min_number_epochs, int(max_number_epochs*43363/709416)),

                     '2JHN':max(min_number_epochs, int(max_number_epochs*43363/119253)), '2JHC':max(min_number_epochs, int(max_number_epochs*43363/1140674)),

                     '2JHH':max(min_number_epochs, int(max_number_epochs*43363/378036)), '3JHN':max(min_number_epochs, int(max_number_epochs*43363/166415)),

                     '3JHC':max(min_number_epochs, int(max_number_epochs*43363/1510379)), '3JHH':max(min_number_epochs, int(max_number_epochs*43363/590611))}

    

    return patience_dict           



def run_nn(load_existing_model, k_fold, trn_idx, val_idx, train, nn_mol_features, k_folds, file_folder, nn_features_for_lgb_test, nn_features_for_lgb_train,

               scores_nn, test_selected, mol_type, oof_nn, pred_nn, run_number):

    

    print(f'fold {k_fold} of the {mol_type}')

    

    #model_name_wrt = ('nn_model_{mol_type}_{str(k_fold)}')



    X_t, X_v, y_t, y_v = split_train_and_val_data(train, trn_idx, val_idx, nn_mol_features)

        

    nn_model = load_or_create_nn_model(X_t, k_fold, k_folds, file_folder, load_existing_model = load_existing_model)

    nn_model.compile(loss='mse', optimizer=Adam())

    

    #-----CALL BACKS-----

    # Callback for Early Stopping... May want to raise the min_delta for small numbers of epochs

    # Callback for Reducing the Learning Rate... when the monitor levels out for 'patience' epochs, then the LR is reduced

    # Save the best value of the model for future use

    patience_dict = get_patience_dict(max_number_epochs = 10, min_number_epochs = 3) #change to a higher number to get a better result

    

    patience = patience_dict[mol_type]

    

    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=int(patience*1.7),verbose=1, mode='auto', restore_best_weights=True)

    rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=patience, min_lr=1.001e-6, mode='auto', verbose=1)

    #sv_mod = callbacks.ModelCheckpoint(model_name_wrt, monitor='val_loss', save_best_only=True, period=1)



    try:    

        history = nn_model.fit(X_t,y_t,

                validation_data=(X_v, y_v), 

                callbacks=[es, rlr], epochs=epoch_n, batch_size=batch_size, verbose=verbose)

    except:

        nn_model = create_nn_model(X_t.shape[1])

        nn_model.compile(loss='mse', optimizer=Adam())



        history = nn_model.fit(X_t,y_t,

                validation_data=(X_v, y_v), 

                callbacks=[es, rlr], epochs=epoch_n, batch_size=batch_size, verbose=verbose)

     

    #oof pred

    nn_oof_pred = nn_model.predict(X_v)

    

    nn_features_for_lgb_train.loc[val_idx, ['sc', 'fc', 'sd','pso', 'dso']] = nn_oof_pred

    

    nn_oof_pred = 0.5*nn_oof_pred[:,0] + 0.5*nn_oof_pred[:,1:].sum(axis=-1) #blend between 'sc' and ['fc', 'sd','pso', 'dso']

    oof_nn.iloc[val_idx, 0] = nn_oof_pred

    

    #test pred

    nn_test_pred = nn_model.predict(test_selected)

    

    nn_features_for_lgb_test.loc[:, ['sc', 'fc', 'sd','pso', 'dso']] += nn_test_pred

    

    nn_test_pred = 0.5*nn_test_pred[:,0] + 0.5*nn_test_pred[:,1:].sum(axis=-1) #blend between 'sc' and ['fc', 'sd','pso', 'dso']

    pred_nn.loc[:, k_fold] = nn_test_pred

    

    #save the generated NN features for the LGB model. They are the 5 predictions + a 32 neurons layer

    intermediate_layer_model = Model(inputs=nn_model.input, outputs=nn_model.layers[7].output)

    nn_features_for_lgb_train.loc[val_idx, cols_nn_features_for_lgb[5:(n_features + 5)]] = intermediate_layer_model.predict(X_v)

    nn_features_for_lgb_test.loc[:, cols_nn_features_for_lgb[5 + n_features*k_fold: 5 + n_features*(k_fold+1)]] = intermediate_layer_model.predict(test_selected)

    

    nn_score = calc_logmae(y_v['scalar_coupling_constant'].values, nn_oof_pred)

    scores_nn[mol_type][k_fold] = nn_score

    

    print(f'NN_score --> logmae for {mol_type} fold {k_fold} is {nn_score}')



    return nn_features_for_lgb_test, nn_features_for_lgb_train, pred_nn, oof_nn, scores_nn



def get_oofs_and_preds(df_type, mol_type):

    

    #this function is used for stacking models

    if df_type == 'oof':

        df = pd.read_csv(f'{original_data_folder}/train.csv', usecols = ['id', 'type'])

        file_names = ['OOF_FELIPE_LGB_1944.csv',

                      'OOF_FELIPE_NN_1917.csv',

                         'harsh_oof_1.688.csv',

                         'oof_lolstart_lgb_1720.csv',

                         'oof_yassine_lgb_5_folds_-1.295.csv',

                         'harsh_10fold_oof_1.670.csv']

    

    if df_type == 'pred':

        df = pd.read_csv(f'{original_data_folder}/test.csv', usecols = ['id', 'type'])

        file_names = ['PRED_FELIPE_LGB_1944.csv',

                      'PRED_FELIPE_NN_1917.csv',

                         'harsh_pred_1.688.csv',

                         'pred_lolstart_lgb_1720.csv',

                         'pred_yassine_lgb_5_folds-1.295.csv',

                         'harsh_10fold_pred_1.670.csv']

    

    sc_columns = []

    for i, file_name in enumerate(file_names):

        data = pd.read_csv(f'{preds_and_oofs_folder}/{file_name}')

        

        try:

            data.rename(columns={'oof':'scalar_coupling_constant'}, inplace=True)

        except:

            pass

        

        try:

            data.rename(columns={'pred':'scalar_coupling_constant'}, inplace=True)

        except:

            pass

        

        try:

            data.rename(columns={'prediction':'scalar_coupling_constant'}, inplace=True)

        except:

            pass

        

        try:

            data.rename(columns={'ind':'id'}, inplace=True)

        except:

            pass

        

        data.sort_values('id', inplace=True)

        df[f'sc_{i}'] = data['scalar_coupling_constant'].values

        sc_columns.append(f'sc_{i}')

    

    df = df[df['type'] == mol_type]

    del df['type']

    

    df.loc[:, sc_columns] -= df.loc[:, sc_columns].mean().mean()

    df.loc[:, sc_columns] /= df.loc[:, sc_columns].stack().std() 

              

    return df, sc_columns

              

def create_or_load_selected_features(preds_and_oofs_folder, train, scalar_coupling_contributions):

    

    try:

        selected_features = np.load(preds_and_oofs_folder + f'/nn_mol_features_{mol_type}.npy', allow_pickle=True)

        print(f'-------There are {len(selected_features)} features in your model-------')

    except:

        selected_features = select_best_features(preds_and_oofs_folder, train, scalar_coupling_contributions)

        np.save(preds_and_oofs_folder + f'/nn_mol_features_{mol_type}', selected_features)

    

    return list(selected_features)
original_data_folder = '../input/champs-scalar-coupling'

preds_and_oofs_folder = '../input/preds-on-oof-and-test'

train_and_test_with_feats_folder = '../input/features-for-top-5-lb-with-nn-or-lgb'

scalar_coupling_contributions = pd.read_csv(original_data_folder + '/scalar_coupling_contributions.csv')



config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 

config.gpu_options.allow_growth = True

config.gpu_options.per_process_gpu_memory_fraction = 1

sess = tf.Session(config=config) 

K.set_session(sess)



epoch_n = 500

verbose = 1

batch_size = 2048



#mol_types = ['1JHN', '1JHC', '2JHN', '2JHC', '2JHH', '3JHN', '3JHC', '3JHH']*20

mol_types = ['1JHN']



run_number = 0



k_folds = 1

scores_nn = dict()



for mol_type_index, mol_type in enumerate(mol_types):

    

    print(mol_type, f'- run number {run_number}')

    

    ###############

    #LOAD DATA

    ###############

    

    #scores_nn is a dict used to save the score reached on each mol_type

    try:

        scores_nn = np.load(f'run_{run_number}_scores_nn.npy', allow_pickle=True).item()

    except:

        scores_nn = dict()

    

    train = pd.read_csv(train_and_test_with_feats_folder + '/train_' + mol_type + '.csv').fillna(0)

    test_full = pd.read_csv(train_and_test_with_feats_folder + '/test_' + mol_type + '.csv').fillna(0)

    

    print(f' Working with data of type {mol_type} and shape {train.shape}')

    

    #predictions from other models used for stacking

    df_oofs, sc_columns = get_oofs_and_preds('oof', mol_type)

    df_pred, sc_columns = get_oofs_and_preds('pred', mol_type)



    #Add scalar contributions to our train df

    #This will be used by the NN as target values

    molecules_names = train['molecule_name'].unique()

    type_scalar_contributions = scalar_coupling_contributions[scalar_coupling_contributions['molecule_name'].isin(molecules_names)]

    type_scalar_contributions = type_scalar_contributions[type_scalar_contributions['type'] == mol_type]

    

    for col in ['fc', 'sd','pso', 'dso']:

        train[col] = type_scalar_contributions[col].values        

    

    del type_scalar_contributions, molecules_names

    

    ###############

    #SELECT BEST FEATURES

    #Use an LGBM model to selected the best features for this coupling type

    #The selection method is define as: If by randomizing the values of a feature, the LGB score doesnt change

    #by a considerable delta, then this feature is not relevant

    ###############



    nn_mol_features = create_or_load_selected_features(preds_and_oofs_folder, train, scalar_coupling_contributions)

        

    for col in sc_columns:

        nn_mol_features.append(col)

        



    ###############

    #SPLIT THE DATA EM PREPROCESS IT

    #Process the data, randomizing it and selecting a train and validations datasets.

    #The split is my molecule, so in the train_df we wont see any molecules that are in the val_df

    ###############



    train, index_folds = preprocess_train_data(mol_type, train, scalar_coupling_contributions, 

                                  k_folds=k_folds, val_data_ratio_if_no_kfolds = 0.2, verbose = True)

    

    #add target and oof predictions to our features

    train = pd.merge(train, df_oofs, on=['id'])

    test_full = pd.merge(test_full, df_pred, on=['id'])

    

    del df_oofs, df_pred    

    

    #scale test dataset

    std_scaler = StandardScaler().fit(train[nn_mol_features])

    test_selected = std_scaler.transform(test_full[nn_mol_features].values)

    del test_full

    

    ###############

    #CREATE THE ARRAYS THAT WE WILL FILL WITH OUR RESULTS

    ###############

    

    #create empty arrays for our predictions

    oof_nn = pd.DataFrame(np.zeros((len(train), 1)))

    pred_nn = pd.DataFrame(np.zeros((len(test_selected), k_folds)), columns = list(range(k_folds)))

    

    n_features = 32

    cols_nn_features_for_lgb = np.array(['sc', 'fc', 'sd','pso', 'dso'] + [f'nn_feat_{k_fold}_{i}'  for k_fold in range(k_folds) for i in range(1, n_features+1)]).flatten()

    

    nn_features_for_lgb_train = pd.DataFrame(np.zeros((train.shape[0], n_features + 5)), columns = cols_nn_features_for_lgb[:n_features + 5])

    nn_features_for_lgb_test= pd.DataFrame(np.zeros((test_selected.shape[0], n_features*k_folds + 5)), columns = cols_nn_features_for_lgb)

    

    #create the dict to save our scores

    if mol_type not in scores_nn:

        scores_nn[mol_type] = dict()

    

    gc.collect()

    

    ###############

    #START THE TRAINING

    ###############

    for k_fold, (trn_idx, val_idx) in enumerate(index_folds):

        

        load_existing_model = True

        nn_features_for_lgb_test, nn_features_for_lgb_train, pred_nn, oof_nn, scores_nn = run_nn(load_existing_model, k_fold, trn_idx, val_idx, train, nn_mol_features, k_folds,

                                                                                                 preds_and_oofs_folder, nn_features_for_lgb_test, nn_features_for_lgb_train,

                                                                                                 scores_nn, test_selected, mol_type, oof_nn, pred_nn, run_number)

        gc.collect()

        

    nn_features_for_lgb_test.loc[:, ['sc', 'fc', 'sd','pso', 'dso']] /= k_folds

    

    ###############

    #SAVE OUR RESULTS FOR THIS COUPLING TYPE

    ###############

    

    np.save(f'run_{run_number}_scores_nn', scores_nn)

    np.save(f'run_{run_number}_{mol_type}_oof_nn', oof_nn)

    pred_nn.to_pickle(f'run_{run_number}_{mol_type}_pred_nn')

    nn_features_for_lgb_train.to_pickle(f'run_{run_number}_{mol_type}_nn_features_for_lgb_train')

    nn_features_for_lgb_test.to_pickle(f'run_{run_number}_{mol_type}_nn_features_for_lgb_test')

    

    #del nn_features_for_lgb_test, nn_features_for_lgb_train, pred_nn, oof_nn, train, test_selected

    

    gc.collect()

for i in range(32):

    nn_feat = nn_features_for_lgb_train.iloc[val_idx, i+5]

    target = train.loc[val_idx, 'scalar_coupling_constant']

    plt.scatter(nn_feat, target, s = 0.2)

    plt.xlabel('scalar coupling constant')

    plt.ylabel(f'nn_feat_{i}')

    plt.show()
sub = pd.read_csv(f'{preds_and_oofs_folder}/final_model_submission.csv')

sub.to_csv('final_sub.csv', index=False)