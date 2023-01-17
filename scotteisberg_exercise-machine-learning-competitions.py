#!pip install datawig



#import datawig

#from datawig.column_encoders import ( SequentialEncoder, NumericalEncoder, CategoricalEncoder, BowEncoder )

#from datawig.imputer import (Imputer, INSTANCE_WEIGHT_COLUMN)

#from datawig.mxnet_input_symbols import (BowFeaturizer, EmbeddingFeaturizer,

#                                         LSTMFeaturizer, NumericalFeaturizer)

#from datawig.utils import random_split
import os

import pandas as pd

import numpy as np

import pytz

from datetime import datetime

from functools import reduce

tz = pytz.timezone('Europe/Berlin')



from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex7 import *

                                     

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import (

    SimpleImputer, KNNImputer, IterativeImputer, MissingIndicator)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler)



from scipy.stats import (skew, boxcox_normmax)

from scipy.special import boxcox1p



from sklearn.pipeline import make_pipeline, make_union

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import (mean_squared_error, mean_absolute_error, accuracy_score)

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from mlxtend.regressor import StackingCVRegressor



from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM

from keras.wrappers.scikit_learn import (KerasClassifier, KerasRegressor)

from keras.optimizers import (Adam)

from keras import regularizers



import matplotlib.pyplot as plt



# pd options

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)



numeric_dtypes    = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

object_dtypes     = ['object']

category_dtypes   = ['category']

cardinality_lim   = 10



# Set up code checking

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 



def dt_now():

    return datetime.now(tz)



def log(*args):

    print(dt_now(), " ".join(map(str, args)))

    pass
log('Starting...')



train = pd.read_csv('../input/train.csv', index_col='Id')

test = pd.read_csv('../input/test.csv')

test_id = test['Id']

test.drop(['Id'], axis=1, inplace=True)



log("Train set size:", train.shape)

log("Test set size :", test.shape)

log('Read files')
train.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = train.SalePrice #np.log1p(train.SalePrice) # log1p which applies log(1+x) to SalePrice

log('sale price stat.')

y.describe()

y_real  = y.to_numpy().reshape(-1,1)



X = train.copy()

# drop irrelevant

X.drop(['Utilities', 'PoolQC'], axis=1, inplace=True)

T = test.copy()

T.drop(['Utilities', 'PoolQC'], axis=1, inplace=True)

# drop target

X.drop(['SalePrice'], axis=1, inplace=True)    



X_full = pd.concat([X, T]).reset_index(drop=True)

X_full.info()



log('Generated dataset')
#https://towardsdatascience.com/preprocessing-encode-and-knn-impute-all-categorical-features-fast-b05f50b4dfaa

#https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779#https://hackernoon.com/build-your-first-neural-network-to-predict-house-prices-with-keras-3fb0839680f4

#https://medium.com/analytics-vidhya/using-scikit-learns-iterative-imputer-694c3cca34de

#https://stackoverflow.com/questions/44132652/keras-how-to-perform-a-prediction-using-kerasregressor

#https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/#adding-k-fold-cross-validation

#https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233

#https://machinelearningmastery.com/how-to-reduce-generalization-error-in-deep-neural-networks-with-activity-regularization-in-keras/



def category_objects(dataset):

    for c in dataset.columns:

        if dataset[c].dtype.name in category_dtypes:

            dataset[c] = dataset[c].astype(object, axis=0)

            log('category -> object:', c)

    pass



def ordinal_encode(dataset):

    imputer = OrdinalEncoder()

    copy = dataset.copy()

    for c in copy.columns:

        copy_c = copy[c].dropna()

        if copy_c.dtype.name not in object_dtypes:

            continue

        #retains only non-null values

        nonulls = np.array(copy_c)

        #reshapes the data for encoding

        impute_reshape = nonulls.reshape(-1,1)

        #encode date

        impute_ordinal = imputer.fit_transform(impute_reshape)

        #Assign back encoded values to non-null values

        dataset.loc[dataset[c].notnull(), c] = np.squeeze(impute_ordinal)    

        log('encoded:', c)

    pass    
def pre_prepare(dataset):

    df = dataset.copy()

    category_objects(df)

    ordinal_encode(df)

    return df



# IterativeImputer

def missing_values_impution(dataset, m_iter=10, strategy='mean'):

    df = dataset.copy()

    imputer = IterativeImputer(sample_posterior=True, 

                               max_iter=m_iter, 

                               verbose=0, 

                               random_state=42, 

                               initial_strategy=strategy)

    imputer.fit(df)

    return pd.DataFrame(imputer.transform(df), columns=df.columns)

    

#DeepNetwork: 

def missing_values_deep(dataset):

    df = dataset.copy()

    nan_count = {c:df[c].isna().sum() for c in df.columns if df[c].isna().any()}

    nan_count_sorted = sorted(nan_count.items(), key=lambda kv: kv[1], reverse=True)

    num_cols = [c for c in df.columns if df[c].dtype.name in numeric_dtypes]

    obj_cols = [c for c in df.columns if df[c].dtype.name in object_dtypes]

    cat_cols = [c for c in df.columns if df[c].dtype.name in category_dtypes]

    for k,v in nan_count_sorted:

        data_encoder_cols = ([NumericalEncoder(c) for c in num_cols if k != c] +

            [SequentialEncoder(c) for c in cat_cols if k != c] +

            [SequentialEncoder(c) for c in obj_cols if k != c])

        data_featurizer_cols = ([NumericalFeaturizer(c) for c in num_cols if k != c] +

            [LSTMFeaturizer(c) for c in cat_cols if k != c] +

            [LSTMFeaturizer(c) for c in obj_cols if k != c])

        label_encoder_cols = [ NumericalEncoder(k) if k in num_cols else CategoricalEncoder(k) ]

        imputer = Imputer(

            data_featurizers=data_featurizer_cols,

            label_encoders=label_encoder_cols,

            data_encoders=data_encoder_cols,

            output_path='imputer_model'

        )

        imputer.fit(train_df=df)

        prob_dict_topk = imputer.predict_proba_top_k(df, top_k=5)

        log('Sum nan', df[k].isna().sum())

    pass

def plots(history, ps, prefix='val_'):

    fig, axs = plt.subplots(ps.shape[0], ps.shape[1])

    ps_flatt = ps.flatten()

    for i in range(len(ps_flatt)):

        a = fig.get_axes()[i]

        a.plot(history.get(ps_flatt[i]))

        a.plot(history.get(prefix+ps_flatt[i]))

        a.set_title("Model {}".format(ps_flatt[i]))

        a.set_ylabel(ps_flatt[i])

        a.set_xlabel('epoch')

        a.legend([ps_flatt[i], prefix+ps_flatt[i]], loc='upper right')

        a.set_ylim(bottom=0)

    plt.tight_layout()

    plt.show()

    pass



# test

#plots({}, np.array([['loss', 'mse']]))  



def create_model(input_shape, reg_l2_rate=0.005, dropout_rate=0.3, adam_learn_rate=0.0025):

    #log(input_shape, reg_l2_rate, dropout_rate, adam_learn_rate)

    model = Sequential([

        Dense(units=1000, activation='relu', kernel_regularizer=regularizers.l2(reg_l2_rate), input_shape=input_shape),

        Dropout(dropout_rate),

        Dense(units=1000, activation='relu', kernel_regularizer=regularizers.l2(reg_l2_rate),),

        Dropout(dropout_rate),

        Dense(units=1000, activation='relu', kernel_regularizer=regularizers.l2(reg_l2_rate),),

        Dropout(dropout_rate),

        Dense(units=1000, activation='relu', kernel_regularizer=regularizers.l2(reg_l2_rate),),

        Dropout(dropout_rate),

        Dense(units=1000, activation='relu', kernel_regularizer=regularizers.l2(reg_l2_rate),),

        Dropout(dropout_rate),

        Dense(units=1, activation='elu', kernel_regularizer=regularizers.l2(reg_l2_rate),),

    ])

    adam = Adam(learning_rate=adam_learn_rate, beta_1=0.9, beta_2=0.999, amsgrad=True, epsilon=1e-8)

    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse'])

    return model
def find_score(

    dataset,

    iterative_params=[

        {'iter': 10, 'strategy': 'mean'},

        {'iter': 10, 'strategy': 'median'},

        {'iter': 50, 'strategy': 'mean'},

        {'iter': 50, 'strategy': 'median'},

        {'iter': 100, 'strategy': 'mean'},

        {'iter': 100, 'strategy': 'median'}

    ], 

    keras_params=[

        [.005,250,300], [.005,300,250], [.005,300,300], [.005,250,250],

        [.0025,200,300], [.0025,300,200], [.0025,300,250], [.0025,250,300], 

        [.001,300,300], [.001,400,350], [.001,350,400], [.001,400,400]

    ]):

    df = pre_prepare(dataset)

    model_results = {}

    for ip in iterative_params:

        #log(X_full['Electrical'].unique())

        X_full_iterative_imputed = missing_values_impution(df.copy(), m_iter=ip.get('iter'), strategy=ip.get('strategy'))        

        #log(X_full_iterative_imputed['Electrical'].unique())

        #X_full_deep_imputed     = missing_values_deep(df)

        #log(X_full_deep_imputed['Electrical'].unique())

         

        # split train and test again

        X_train       = X_full_iterative_imputed.iloc[:len(y), :]

        X_test        = X_full_iterative_imputed.iloc[len(X_train):, :]

        # scale features

        X_train_scale = StandardScaler().fit_transform(X_train.copy())

        X_test_scale  = StandardScaler().fit_transform(X_test.copy())

        

        ## loop model with some diff. rates

        for fit_rate in keras_params:

            estimator  = create_model((X_train_scale.shape[1],), adam_learn_rate=fit_rate[0])

            fit        = estimator.fit(X_train_scale, y, epochs=fit_rate[1], batch_size=fit_rate[2], verbose=0, validation_data=(X_train_scale, y),)

            prediction = estimator.predict(X_train_scale).reshape(-1,1)

            mae        = mean_absolute_error(y_real, prediction)

            test_pred  = estimator.predict(X_test_scale).reshape(-1,1).flatten()

            key        = "{}:{}:{}".format(str(ip.get('iter')), ip.get('strategy'), ":".join(map(str, fit_rate)))

            log('Key.', key, 'MAE. score:', mae)

            model_results[key] = {

                'estimator'  : estimator,

                'history'    : fit.history,

                'prediction' : prediction,

                'test_pred'  : test_pred,

                'mae'        : mae 

            }

    return model_results
log('Start prediction...')

# find out all scores

results = find_score(X_full)

# find out the possible best rate    

best_rate = min(results.keys(), key=(lambda k: results.get(k).get('mae')))

log('rate stat.', {k:v.get('mae') for k,v in results.items()})

log('best rate', best_rate, results.get(best_rate).get('mae'))

best_result = results.get(best_rate)

# plot history

plots(best_result.get('history'), np.array([['loss', 'mse']]))

# plot prediction

plots({'prices': y_real, 'predict_prices': best_result.get('prediction')}, np.array([['prices']]), 'predict_')
## make predictions which we will submit. 

# get back the value after log(1+x) at the beginning

#test_preds = mms_y.inverse_transform(estimator.predict(np_X_test)).flatten()

test_preds = best_result.get('test_pred')

log('End prediction...')



# The lines below shows how to save predictions in format used for competition scoring

output = pd.DataFrame({'Id': test_id, 'SalePrice': test_preds})

print(output.head())

output.to_csv('submission.csv', index=False)

log('Saved file')
"""

#mms_y = MinMaxScaler()

#y_scale       = mms_y.fit_transform(y.copy()[:,np.newaxis])



#np_X          = np.reshape(np_X, (np_X.shape[0], np_X.shape[1], 1))

#np_X_test     = np.reshape(np_X_test, (np_X_test.shape[0], np_X_test.shape[1], 1))

#np_y          = np.reshape(np_y, (np_y.shape[0], np_y.shape[1], 1))



def lstm_model():

    model = Sequential([

        LSTM(units=50,return_sequences=True, input_shape=(np_X.shape[1],1)),

        Dropout(0.2),

        LSTM(units=50,return_sequences=True),

        Dropout(0.2),

        LSTM(units=50,return_sequences=True),

        Dropout(0.2),

        LSTM(units=50),

        Dropout(0.2),

        Dense(units=1),

    ])

    pass

    

#prediction = mms_y.inverse_transform(estimator.predict(X_train_scale).reshape(-1,1))

#y_real = mms_y.inverse_transform(y_scale)

#log(prediction.shape, prediction.flatten()[:5])

#log(y_real.shape, y_real.flatten()[:5])

"""

    

"""

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

estimator = KerasRegressor(build_fn=create_model, nb_epoch=100, batch_size=100, verbose=False)

cvs = cross_val_score(estimator, X_train_scale, y, cv=kfolds)

log("Mean. {:0.4f}, Std. {:0.4f}".format(cvs.mean(), cvs.std()))

"""



"""

fold_nr = 1

kfolds_metrics = {}

for train, test in kfolds.split(np_X): 

    history = model.fit(np_X[train], np_y[train],

              batch_size=32,

              epochs=100,

              verbose=False,

              validation_data=(np_X[test], np_y[test]))

              #validation_split=0.2)

    scores = model.evaluate(np_X, np_y, verbose=0)

    log(f'Score for fold-{fold_nr}: {model.metrics_names[0]} of {scores[0]:0.4f}; {model.metrics_names[1]} of {scores[1]:0.4f}')

    kfolds_metrics[model.metrics_names[1]] = kfolds_metrics.get(model.metrics_names[1]).append(scores[1]) if kfolds_metrics.get(model.metrics_names[1]) else [scores[1]]

    kfolds_metrics[model.metrics_names[0]] = kfolds_metrics.get(model.metrics_names[0]).append(scores[0]) if kfolds_metrics.get(model.metrics_names[0]) else [scores[0]]

    kfolds_metrics['history']              = kfolds_metrics.get('history').append(history) if kfolds_metrics.get('history') else [history]



    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc='upper right')

    plt.ylim(bottom=0)

    plt.show()



    fold_nr = fold_nr + 1

    

log(kfolds_metrics)

"""