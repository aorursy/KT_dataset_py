# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



## Install XGB

!pip install xgboost
import pandas as pd

from sklearn.model_selection import train_test_split

##

## Function to get the train validation split

##

def get_train_dataset(file_name):

    df = pd.read_csv(file_name)

    train,val = train_test_split(df, test_size = 0.2, stratify = df.target, random_state = True)

    #print(list(df))

    train_y = train.target

    train.drop(['target','id'], axis = 1, inplace = True)

    val_y = val.target

    val.drop(['target','id'],axis = 1, inplace = True) ##   

    return train, train_y, val, val_y

##

## Get full dataset without validation split

##

def get_full_data(file_name):

    train = pd.read_csv(file_name)

    ## Change the datatype to int.

    ## Fill the na value to -999

    for col in list(train):

        train[col] = train[col].map( lambda x: int(x) if str(x).isnumeric() else -999 )

        train[col] = train[col].astype('int')

    

    train_y = train.target

    train.drop(['target','id'], axis = 1, inplace = True)

    return train, train_y



train_file_name = 'data/equip_failures_training_set.csv'

#train_x, train_y, val_x,val_y = get_train_dataset(train_file_name)

train_x, train_y = get_full_data(train_file_name)
train_x.head()
## Helper function to calculate all na values per column

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="whitegrid")

from matplotlib.widgets import Slider

def get_na_counts(df):

    f, ax = plt.subplots(figsize=( 25,6))

    cols = list(df)[2:]

    nan_count = []

    for col in cols:

        nan_count.append(df[df[col]==-999].shape[0])

    

    sns.barplot(x=cols,y=nan_count)

    

get_na_counts(train_x)

        


def get_total_pure(df):

    

    valid = df.apply(lambda x: -999 in list(x)).sum()

    n_valid = df.shape[0] - valid

    fig1, ax1 = plt.subplots(figsize=(10,10))

    ax1.pie([valid,n_valid], explode=(0,0.2),shadow=True, colors=['red','#8194ff'],autopct='%10.1f%%', labels=['No Na', 'Contain Atleast One NA'], startangle=180)

get_total_pure(train_x)
train_x.replace(-999,0).describe()
## Plot the dataset in the each class

def get_class_balance(df):

    valid = df.sum()

    n_valid = df.shape[0] - valid

    fig1, ax1 = plt.subplots(figsize=(10,10))

    ax1.pie([valid,n_valid], explode=(0,0.2),shadow=True, colors=['red','#8194ff'],autopct='%10.1f%%', labels=['True', 'False'], startangle=180)

    

get_class_balance(train_y)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from xgboost import XGBClassifier



## Cross Validation steps

## you can change the parameter dictionary to add the parameters



def cross_validate(train_x, train_y):

    

    ## Feel free to edit the parameters.

    parameters = {'nthread':[5], #when use hyperthread, xgboost may become slower

#               'objective':['binary:logistic'],

              'learning_rate': [0.1], #so called `eta` value

              'max_depth': [15,10],

              'min_child_weight': [3,2,5],

              'gamma':[0.2],

                'subsample':[1.0],

                'colsample_bytree':[0.8],

              'silent': [1],

              'reg_alpha':[1e-5],

              #'subsample': [0.7],

              #'colsample_bytree': [0.7],

              'n_estimators': [250,500], #number of trees, change it to 1000 for better results

            'missing':[-999],

            'scale_pos':[1],

            'gpu_id':['6']

#               'seed': [1337]}

                 }



    

    xgb_model = XGBClassifier()

    

    ## Perform grid search with cv 5 to get best parameters

    ## Roc AUC is used as the scoring function.

    clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 

                   cv = 5, 

                   scoring='roc_auc',

                   verbose=2, refit=True)

    clf.fit(train_x, train_y)

    return clf
print(clf.best_params_)

print(clf.best_score_)
##

## Helper function to get test data.

##

def get_test_data(file_name):

    train = pd.read_csv(file_name)

    

    for col in list(train):

        train[col] = train[col].map( lambda x: int(x) if str(x).isnumeric() else -999 )

        train[col] = train[col].astype('int')

    

    return train



##

## Helper function to predict and save the result in submission format

##

def predict(model):

    test_df = get_test_data('equip_failures_test_set.csv')

    result_ret = pd.DataFrame()

    result_ret['id'] = test_df.id

    test_df.drop(['id'],axis=1, inplace=True)

    result = model.predict(test_df)

    result_ret['target'] = result

    result_ret.to_csv('result_100_red.csv', index = False)

    

predict(clf.best_estimator_)

    
#clf.best_estimator.get_score(importance_type='gain')



## Helper function to plot importance

def plot_imp( n  = 10,asc = False):

    over_all_imp = clf.best_estimator_.get_booster().get_score(importance_type='gain')

    #print(over_all_imp)

    imp_list = [(k,v) for k,v in over_all_imp.items()]

    #print(imp_list)

    fig1, ax1 = plt.subplots(figsize=(20,10))

    imp_list = sorted(imp_list, key = lambda x: x[1])

    imp_list = imp_list[:n] if asc else imp_list[len(imp_list)-n:]

    if not asc:

        imp_list.reverse()



    g = sns.boxplot(x = [ i[0] for i in imp_list], y = [i[1] for i in imp_list])

    plt.xticks(rotation=30)

    

plot_imp(n = 5, asc=True)
plot_imp(n = 10)
from sklearn import metrics

##

## simplify model with less Features

##

def remove_least_important(clf,train_x,train_y,n):

    over_all_imp = clf.best_estimator_.get_booster().get_score(importance_type='gain')

    imp_list = [(k,v) for k,v in over_all_imp.items()]

    imp_list = sorted(imp_list, key = lambda x: x[0])

    

    to_remove = [ imp_list[i][0] for i in range(n) ]

    train_x_cp = train_x.copy()

    train_x_cp.drop(to_remove,axis = 1,inplace=True)

    clf_cp = cross_validate(train_x_cp, train_y)

    print(clf_cp.best_params_)

    print(clf_cp.best_score)

#     xgb_model = XGBClassifier(**clf.best_params_)

#     X_train, X_test, y_train, y_test = train_test_split(train_x_cp, train_y, test_size=0.2, random_state=42, stratify=train_y)

#     xgb_model.fit(X_train,y_train)

    

#     fpr, tpr, thresholds = metrics.roc_curve(xgb_model.predict(X_test), y_test, pos_label=2)

   # print(metrics.auc(fpr, tpr))

    return clf_cp



    

clf_cp = remove_least_important(clf, train_x, train_y,20)

    

predict(clf_cp.best_estimator_)




df_result =  pd.read_csv('result_sim.csv')
df_result.target.value_counts()
list(df_result)
clf.best_params_
##

## Ensemble models

##



def ensemble_model(clf,train_x,train_y):

    test_df = get_test_data('equip_failures_test_set.csv')

    scores = clf.grid_scores_

    scores = sorted(scores, key = lambda x: -x[1])

    limits = 5

    model_params = scores[:5]

    models = []

    for params in model_params:

        x = XGBClassifier(**params[0])

        x.fit(train_x, train_y)

        model.append(x)

    

    r_df = pd.DataFrame()

    #r_df['id'] = test_df.id

    for i in range(len(models)):

        r_df[str(i)] = model[i].predict(test_df)

    r_df['mean'] = r_df.mean(axis = 1)

    final_result = pd.DataFrame()

    final_result['id'] = test_df.id

    final_result['target'] = r_df['mean'].map( lambda x: 0 if x<0.5 else 1)

    final_result.to_csv('xgb_ensemble.csv')
ensemble_model(clf,train_x,train_y)

clf.best_score_
import numpy as np

import pandas as pd

import re

import matplotlib.pyplot as plt



import keras



from keras import layers

from keras import backend as K

from keras.models import Model

from keras.models import load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers import Dense, Dropout, Conv1D, Flatten

from keras.layers import CuDNNLSTM, GRU, LSTM, ConvLSTM2D, CuDNNGRU

from keras.layers.normalization import BatchNormalization



import numpy as np



import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MaxAbsScaler

from sklearn.metrics import roc_curve, auc

from sklearn import preprocessing

from sklearn.metrics import f1_score



from sklearn.externals import joblib

import pickle

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

pd.options.display.max_seq_items = 2000



from keras.backend.tensorflow_backend import set_session

from keras.backend.tensorflow_backend import clear_session

from keras.backend.tensorflow_backend import get_session

import tensorflow

import gc



!pip install xgboost

import xgboost

import pickle



# Reset Keras Session

def reset_keras():

    sess = get_session()

    clear_session()

    sess.close()

    sess = get_session()



    try:

        del classifier # this is from global space - change this as you need

    except:

        pass



    print(gc.collect()) # if it's done something you should see a number being outputted



    # use the same config as you used to create the session

    config = tensorflow.ConfigProto()

    config.gpu_options.per_process_gpu_memory_fraction = 1

    config.gpu_options.visible_device_list = "0"

    set_session(tensorflow.Session(config=config))

    

reset_keras()



MODEL_PATH = 'model'

DATA_PATH = 'data'



TRAIN_NAME = '/equip_failures_training_set.csv'

TEST_NAME = '/equip_failures_test_set.csv'



!ls {MODEL_PATH}

!ls {DATA_PATH}



train_data = pd.read_csv(DATA_PATH+TRAIN_NAME)

test_data = pd.read_csv(DATA_PATH+TEST_NAME)



def check_na(train_data, na=2000):

    dict_list = []

    all_list = train_data.columns

    for x in all_list:

        if x!='id' and x!='target' and x!='sensor1_measure' and int(len(train_data[train_data[str(x)] == 'na'])) > na:

            dict_list.append(x)

    return dict_list



def remove_cols(data, remove_list):

    data = data.drop(remove_list, axis=1)

    return data



dict_list = check_na(train_data, 70000)

train_data = remove_cols(train_data, dict_list)

test_data = remove_cols(test_data, dict_list)



def preprocess(train_data, test_data):

    

    train_data = train_data.convert_objects(convert_numeric=True)

    test_data = test_data.convert_objects(convert_numeric=True)

    train_data = train_data.fillna(0)

    test_data = test_data.fillna(0)

    

    train_y = train_data['target']

    train_data = train_data.copy().drop(['id','target'],axis=1)

    test_data = test_data.copy().drop(['id'],axis=1)

    

    train_np = train_data.values

    test_np = test_data.values

    

    scaler = preprocessing.MinMaxScaler()

    train_x = scaler.fit_transform(train_np)

    test = scaler.transform(test_np)



    return train_x, train_y, test



train_x, train_y, test = preprocess(train_data, test_data)

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.20, stratify = train_y)



def cnn_gru_model(data):

    #input layer

    input_x = keras.Input(shape=(np.shape(data['x_train'])[1], 1), name="x_input")

    

    conv = (Conv1D(64, 1, activation='relu'))(input_x)

    conv = BatchNormalization()(conv)

    conv = (Conv1D(64, 2, activation='relu'))(conv)

    conv = BatchNormalization()(conv)

    conv = (Conv1D(64, 3, activation='relu'))(conv)

    conv = BatchNormalization()(conv)

    

    gru = (CuDNNGRU(64, return_sequences=True))(conv)

    gru = (CuDNNGRU(32, return_sequences=True))(gru)

    gru = (CuDNNGRU(16, return_sequences=False))(gru)

    

    gru = layers.Dense(64, activation='sigmoid')(gru)

    gru = layers.Dense(32, activation='sigmoid')(gru)

    y = layers.Dense(1, activation='sigmoid')(gru)

    

    #model

    #classifier = Model([input_x, input_xd], y)



    classifier = Model([input_x], y)

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier





def cnn_lstm_model(data):

    #input layer

    input_x = keras.Input(shape=(np.shape(data['x_train'])[1], 1), name="x_input")

    

    conv = (Conv1D(64, 1, activation='relu'))(input_x)

    conv = BatchNormalization()(conv)

    conv = (Conv1D(64, 2, activation='relu'))(conv)

    conv = BatchNormalization()(conv)

    conv = (Conv1D(64, 3, activation='relu'))(conv)

    conv = BatchNormalization()(conv)

            

    lstm = (CuDNNLSTM(64, return_sequences=True))(conv)

    lstm = (CuDNNLSTM(64, return_sequences=True))(lstm)

    lstm = (CuDNNLSTM(64, return_sequences=False))(lstm)



    #output layer

    lstm = layers.Dense(64, activation='relu')(lstm)

    lstm = layers.Dense(32, activation='relu')(lstm)

    y = layers.Dense(1, activation='sigmoid')(lstm)

    

    classifier = Model([input_x], y)

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier



def cnn_dnn_model(data):

    #input layer

    input_x = keras.Input(shape=(np.shape(data['x_train'])[1], 1), name="x_input")

    

    # hidden layer

    conv = (Conv1D(64, 1, activation='relu'))(input_x)

    conv = BatchNormalization()(conv)

    conv = (Conv1D(64, 2, activation='relu'))(conv)

    conv = BatchNormalization()(conv)

            

    conv = Flatten()(conv)

    conv = layers.Dense(2048, activation='sigmoid')(conv)

    conv = layers.Dense(512, activation='sigmoid')(conv)

    conv = layers.Dense(64, activation='sigmoid')(conv)

    y = layers.Dense(1, activation='sigmoid')(conv)



    classifier = Model([input_x], y)

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier



BATCH_SIZE = 64

EPOCHS = 50

PATIENCE = 15



def train_model(model, X_train, y_train, X_val, y_val, MODEL_NAME, class_weight):

    callbacks = [EarlyStopping(monitor='val_loss', patience=PATIENCE),

             ModelCheckpoint(filepath=MODEL_PATH+"/"+MODEL_NAME+".h5", monitor='val_loss', verbose=1, save_best_only=True)]

    model.summary()

    # X_train, X_val, y_train, y_val

    model.fit(x={"x_input": X_train}, y=y_train, shuffle=True, epochs=EPOCHS, callbacks=callbacks, batch_size=BATCH_SIZE,

              verbose=1, validation_data=([X_val], y_val), class_weight=class_weight)

    return model



# train on remaining dataset

def train_model_on_val(X_val, y_val, MODEL_NAME):

    model = load_model(MODEL_PATH+"/"+MODEL_NAME+".h5")

    model.summary()

    model.fit(x={"x_input": X_val}, y=y_val, shuffle=True, epochs=5, batch_size=BATCH_SIZE,

              verbose=1)

    model.save(MODEL_PATH+"/"+MODEL_NAME+"_full.h5")

    

def print_test_scores(train_x, train_y, test, MODEL_NAME):

    model_name = MODEL_PATH+"/"+MODEL_NAME+"_full.h5"

    model = load_model(model_name)



    pred_y = model.predict([train_x])

    fpr, tpr, thresholds = roc_curve(train_y, np.around(pred_y))

    train_auc = auc(fpr, tpr)



    print("train auc", train_auc)

    print("train f1 score", f1_score(train_y, np.around(pred_y), average='macro'))



    test = test.reshape(np.shape(test)[0], np.shape(test)[1], 1)

    pred_test = np.around(model.predict([test]))

    print("positives:", pred_test.sum())

    return pred_test



def generate_csv(pred_test, test):

    test_results = pd.DataFrame()

    test_results['id'] = test_data.id

    test_results['target'] = pred_test.astype('int')

    test_results.to_csv(MODEL_PATH+"/"+MODEL_NAME+'.csv', index = False)

    print("generated", MODEL_PATH+"/"+MODEL_NAME+'.csv')



X_train = X_train.reshape(np.shape(X_train)[0],np.shape(X_train)[1],1)

X_val = X_val.reshape(np.shape(X_val)[0],np.shape(X_val)[1],1)



MODEL_NAME = 'cnn_lstm'

network = cnn_lstm_model({'x_train': X_train})

class_weight = {0: 1, 1: 1}

model = train_model(network, X_train, y_train, X_val, y_val, MODEL_NAME, class_weight)

model = train_model_on_val(X_val, y_val, MODEL_NAME)

del model



train_x = train_x.reshape(np.shape(train_x)[0], np.shape(train_x)[1],1)

pred_test = print_test_scores(train_x, train_y, test, MODEL_NAME)

generate_csv(pred_test, test)



MODEL_NAME = 'cnn_gru'

network = cnn_gru_model({'x_train': X_train})

class_weight = {0: 1, 1: 1.2}

model = train_model(network, X_train, y_train, X_val, y_val, MODEL_NAME, class_weight)

model = train_model_on_val(X_val, y_val, MODEL_NAME)

del model



train_x = train_x.reshape(np.shape(train_x)[0], np.shape(train_x)[1],1)

pred_test = print_test_scores(train_x, train_y, test, MODEL_NAME)

generate_csv(pred_test, test)



MODEL_NAME = 'cnn_dnn'

network = cnn_dnn_model({'x_train': X_train})

class_weight = {0: 1, 1: 1.2}

model = train_model(network, X_train, y_train, X_val, y_val, MODEL_NAME, class_weight)

model = train_model_on_val(X_val, y_val, MODEL_NAME)

del model
MODEL_NAME = 'multi_ensemble'



def legendary_ensemble_train(models, train_x):

    n = len(models)

    y = np.zeros((len(train_x), n), dtype = int)

    i = 0

    

    # make predictions from each model

    for model in models:

        y_i = np.around(model.predict(train_x))

        y_i = y_i.reshape(np.shape(y_i)[0])

        y[:,i] = y_i

        i = i + 1

        print("model"+str(i))

        

    y = (np.sum(y, axis=1)).astype(int)

    return y



def load_models(file_name_list):

    models = []

    for filename in file_name_list:

        model = load_model(filename)

        models.append(model)

    return models



file_name_list = ['model/cnn_gru_full.h5','model/cnn_lstm_full.h5','model/cnn_dnn_full.h5']

models = load_models(file_name_list)



# train results

train_x = train_x.reshape(np.shape(train_x)[0], np.shape(train_x)[1],1)

pred_y = legendary_ensemble_train(models, train_x)

fpr, tpr, thresholds = roc_curve(train_y, pred_y)

train_auc = auc(fpr, tpr)

print("train auc", train_auc)

print("train f1 score", f1_score(train_y, pred_y, average='macro'))



# test results

test = test.reshape(np.shape(test)[0], np.shape(test)[1], 1)

pred_test = legendary_ensemble_train(models, test)

print("positives:", pred_test.sum())

    

generate_csv(pred_test, test)