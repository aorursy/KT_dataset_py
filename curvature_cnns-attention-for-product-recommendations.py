import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
#import necessary dependecies
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier,VotingClassifier
 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import os
import warnings
import numpy as np  
import seaborn as sns
import pandas as pd, os, gc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, log_loss
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, RobustScaler
%matplotlib inline
warnings.filterwarnings('ignore')
from typing import List
import tensorflow as tf
import random
from tqdm import tqdm 
import copy
 
tf.random.set_seed(111)
np.random.seed(111)
random.seed(111)
def attention_3d_block(inputs, name):
    # inputs.shape = (batch_size, time_steps, input_dim)
    TIME_STEPS = inputs.shape[1]
    SINGLE_ATTENTION_VECTOR = False

    input_dim = inputs.shape[2]
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:  
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name=name)(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul
import keras
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model
from keras.initializers import glorot_normal
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from keras.regularizers import l2

def get_model():
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)

    input_tensor = Input(shape=(1,df_train.shape[1]))

    x = Conv1D(64, 5,strides=5,padding='same',activation='relu')(input_tensor)
    x = attention_3d_block(x, 'attention_vec_1')
    x = Dropout(0.15)(x)

    x = Conv1D(128, 4,strides=3,padding='same',activation='relu')(x)
    x = attention_3d_block(x, 'attention_vec_2')
    x = Dropout(0.15)(x)

    x = Conv1D(256, 3,strides=3,padding='same',activation='relu')(x)
    x = attention_3d_block(x, 'attention_vec_3')
    x = Dropout(0.15)(x)

    x = GlobalMaxPooling1D()(x)

    x = Dropout(0.15)(x)

    out = Dense(21,kernel_initializer=glorot_normal(seed=1),
              bias_initializer=glorot_normal(seed=1),
              activation="softmax")(x)

    model = Model(inputs=input_tensor,outputs =out)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=["accuracy"])
    return model
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,StandardScaler
train_ = pd.read_csv('../input/zimnat-insurance-recommendation-challenge/Train.csv')
test_ = pd.read_csv('../input/zimnat-insurance-recommendation-challenge/Test.csv')
submission_ = pd.read_csv('../input/zimnat-insurance-recommendation-challenge/SampleSubmission.csv')
from sklearn.model_selection import KFold,GroupKFold

def get_train_test_names(train_, test_, submission_):
    kf = KFold(n_splits=5, shuffle=False)
    for r, (train_index, test_index) in enumerate(kf.split(train_)):
        test = train_.iloc[test_index]

        X_test = []
        X_test_columns = test.columns
        for v in test.values:
            info = v[:8]
            binary = v[8:]
            index = [k for k, i in enumerate(binary) if i == 1]
            for i in index:
                for k in range(len(binary)):
                    if k == i:
                        binary_transformed = list(copy.copy(binary))
                        binary_transformed[i] = 0
                        X_test.append(list(info) + binary_transformed)

        X_test = pd.DataFrame(X_test)
        X_test.columns = ['ID', 'join_date', 'sex', 'marital_status', 'birth_year', 'branch_code',
              'occupation_code', 'occupation_category_code', 'P5DA', 'RIBP', '8NN1',
              '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',
              'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3']
        X_test['ID'] = [str(r)+'_'+str(i) for i in range(X_test.shape[0])]

        yield train_.iloc[train_index], X_test, submission_, '0_fold' + str(r) + '.csv'
    yield train_, test_, submission_, '0_main.csv'
for train, test, submission, name in get_train_test_names(train_, test_, submission_):
    X_train = []
    X_train_columns = train.columns
    c = 0
    for v in train.values:
        info = v[:8]
        binary = v[8:]
        index = [k for k, i in enumerate(binary) if i == 1]
        for i in index:
            c+=1
            for k in range(len(binary)):
                if k == i:
                    binary_transformed = list(copy.copy(binary))
                    binary_transformed[i] = 0
                    X_train.append(list(info) + binary_transformed + [X_train_columns[8+k]] + [c])

    X_train = pd.DataFrame(X_train)
    X_train.columns = ['ID', 'join_date', 'sex', 'marital_status', 'birth_year', 'branch_code',
        'occupation_code', 'occupation_category_code', 'P5DA', 'RIBP', '8NN1',
        '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',
        'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3', 'product_pred', 'ID2']


    X_test = []
    true_values = []
    c = 0
    for v in test.values:
        c += 1
        info = v[:8]
        binary = v[8:]
        index = [k for k, i in enumerate(binary) if i == 1]
        X_test.append(list(info) + list(binary) + [c])
        for k in test.columns[8:][index]:
            true_values.append(v[0] + ' X ' + k)

    X_test = pd.DataFrame(X_test)
    X_test.columns = ['ID', 'join_date', 'sex', 'marital_status', 'birth_year', 'branch_code',
        'occupation_code', 'occupation_category_code', 'P5DA', 'RIBP', '8NN1',
        '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',
        'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3', 'ID2']


    features_train = []
    features_test = []
    columns = []

    append_features = ['P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 
    'N2MW', 'AHXO','BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 
    'ECY3', 'ID', 'ID2', 'join_date', 'sex', 'marital_status', 'branch_code', 'occupation_code', 'occupation_category_code',
    'birth_year']
    for v in append_features:
        features_train.append(X_train[v].values.reshape(-1, 1))
        features_test.append(X_test[v].values.reshape(-1, 1))
        columns.append(np.array([v]))

    y_train = X_train[['product_pred']]


    features_train = np.concatenate(features_train, axis=1)
    features_test = np.concatenate(features_test, axis=1)
    columns = np.concatenate(np.array(columns))

    X_train = pd.DataFrame(features_train)
    X_train.columns = columns
    X_test = pd.DataFrame(features_test)
    X_test.columns = columns
    ############################## fix code ##############################
    X_train.join_date = pd.to_datetime(X_train.join_date,)
    X_test.join_date = pd.to_datetime(X_test.join_date,)

    X_train.join_date = pd.to_datetime(X_train.join_date, format="%Y-%m-%d")
    X_test.join_date = pd.to_datetime(X_test.join_date, format="%Y-%m-%d")

    # new features
    X_train['num_products_subscribed'] = X_train.apply(lambda x : sum(x[X_train.columns[:21]]), axis = 1)
    X_train['join_month'] = X_train['join_date'].dt.month
    X_train['day_of_week'] = X_train['join_date'].dt.dayofweek
    X_train['day_of_week_name'] = X_train['join_date'].dt.day_name()
    X_train['age'] = np.abs(X_train['join_date'].dt.year - X_train['birth_year'])
    X_train['join_time_elapsed'] = np.abs(2020 - X_train['join_date'].dt.year)
    X_train['current_age'] = np.abs(2020 - X_train['birth_year'])

    X_test['num_products_subscribed'] = X_test.apply(lambda x : sum(x[X_test.columns[:21]]), axis = 1)
    X_test['join_month'] = X_test['join_date'].dt.month
    X_test['day_of_week'] = X_test['join_date'].dt.dayofweek
    X_test['day_of_week_name'] = X_test['join_date'].dt.day_name()
    X_test['join_time_elapsed'] = np.abs(2020 - X_test['join_date'].dt.year)
    X_test['age'] = np.abs( X_test['join_date'].dt.year - X_test['birth_year'])
    X_test['current_age'] = np.abs(2020 - X_test['birth_year'])



        
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    y_train = y_train.fillna(0)


    # LABEL ENCODE
    enc = LabelEncoder()
    def encode_LE(train,test,cols,verbose=True):
        for col in cols:

            df_comb = pd.concat([train[col].astype('str'),test[col].astype('str')],axis=0)
            df_comb = enc.fit_transform(df_comb)
            nm = col
            if df_comb.max()>32000: 
                train[nm] = df_comb[:len(train)].astype('int32')
                test[nm] = df_comb[len(train):].astype('int32')
            else:
                train[nm] = df_comb[:len(train)].astype('int16')
                test[nm] = df_comb[len(train):].astype('int16')
            del df_comb; x=gc.collect()
            if verbose: print(nm,', ',end='')

    X_train.day_of_week_name = X_train.day_of_week_name.astype('str')
    X_test.day_of_week_name = X_test.day_of_week_name.astype('str')
    X_train.join_date = X_train.join_date.astype('str')
    X_test.join_date = X_test.join_date.astype('str')

    data = X_train.append(X_test)
    for v in ['sex', 'marital_status', 'branch_code', 'occupation_code', 'occupation_category_code','day_of_week_name','join_date']:
        data.loc[:,v] = enc.fit_transform(data.loc[:,v])
    X_train = data[:X_train.shape[0]]
    X_test = data[-X_test.shape[0]:]

    enc.fit(y_train.iloc[:,0])
    y_train = pd.DataFrame(enc.transform(y_train.iloc[:,0]))
    y_train.columns = ['target']

    X = X_train.drop(['ID','ID2'], axis=1)
    test = X_test.drop(['ID','ID2'], axis=1)

    scaler = StandardScaler()

  
    X[['age']] = scaler.fit_transform(X[['age']])
    test[['age']] = scaler.fit_transform(test[['age']])


    X[['birth_year']] = scaler.fit_transform(X[['birth_year']])
    test[['birth_year']] = scaler.fit_transform(test[['birth_year']])


    df_train =X.values
    df_test = test.values
    y = y_train.target

    x = df_train.reshape(df_train.shape[0], 1,df_train.shape[1])
    xtest = df_test.reshape(df_test.shape[0],1,df_test.shape[1])
  
    ############################### MODELING CODE : PAY attention :) ######################################

    sk = StratifiedKFold(n_splits= 5,random_state=111,shuffle=True)

    es = EarlyStopping(monitor ="val_loss", mode ="min", verbose =1, patience = 100)

    nn_predictions = list()

    N_STARTS = 5
    tf.random.set_seed(111)


    for seed in range(N_STARTS):
        for n, (tr, te) in enumerate(sk.split(x, y)):
            print(f'Fold {n+1}')

            model = get_model()

            checkpoint_path = f'repeat:{seed+1}_Fold:{n+1}.hdf5'
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta= 1e-4, mode='min')
            #cb_checkpt = ModelCheckpoint(checkpoint_path, monitor = 'val_loss', verbose = 0, save_best_only = True,
            #                          save_weights_only = True, mode = 'min')
            model.fit(x[tr],
                    y[tr],
                    validation_data=(x[te], y[te]),
                    epochs=500, batch_size=1024,
                    callbacks=[reduce_lr_loss,es], verbose=2
                  )

            #model.load_weights(checkpoint_path)
            nn_predictions.append(model.predict(xtest,batch_size = 1024))
            print('')

    # get preds :D
    proba = np.average(nn_predictions, axis=0)
    y_test = pd.DataFrame(proba)
    y_test.columns = enc.inverse_transform(y_test.columns)

    answer_mass = []
    for i in range(X_test.shape[0]):
        id = X_test['ID'].iloc[i]
        for c in y_test.columns:
            answer_mass.append([id + ' X ' + c, y_test[c].iloc[i]])

    df_answer = pd.DataFrame(answer_mass)
    df_answer.columns = ['ID X PCODE', 'Label']
    for i in range(df_answer.shape[0]):
        if df_answer['ID X PCODE'].iloc[i] in true_values:
            df_answer['Label'].iloc[i] = 1.0

    df_answer.reset_index(drop=True, inplace=True)
    df_answer.to_csv(name, index=False)