# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing core libraries

import numpy as np

import pandas as pd

from time import time

import pprint

import joblib



# Model selection

from sklearn.model_selection import StratifiedKFold



# Metrics

from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.metrics import make_scorer



# Data transformation pipelines

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import RobustScaler, StandardScaler



# Graphics

import matplotlib.pyplot as plt

import seaborn as sns
# TensorFlow 

import tensorflow as tf

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam, Nadam

from keras.layers import Input, Embedding, Reshape, GlobalAveragePooling1D

from keras.layers import Flatten, concatenate, Concatenate, Lambda, Dropout, SpatialDropout1D

from keras.layers import Reshape, MaxPooling1D,BatchNormalization, AveragePooling1D, Conv1D

from keras.layers import Activation, LeakyReLU

from keras.optimizers import SGD, Adam, Nadam

from keras.models import Model, load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.regularizers import l2, l1_l2

from keras.losses import binary_crossentropy

from keras.utils import get_custom_objects

from keras.layers import Activation, LeakyReLU

from keras.models import load_model
# Reading the data

X = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")

Xt = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")
# Separating target and ids

y = X.target.values

id_train = X.id

id_test = Xt.id



X.drop(['id', 'target'], axis=1, inplace=True)

Xt.drop(['id'], axis=1, inplace=True)
# Classifying variables in binary, high and low cardinality nominal, ordinal and dates

binary_vars = [c for c in X.columns if 'bin_' in c]



nominal_vars = [c for c in X.columns if 'nom_' in c]

high_cardinality = [c for c in nominal_vars if len(X[c].unique()) > 16]

low_cardinality = [c for c in nominal_vars if len(X[c].unique()) <= 16]



ordinal_vars = [c for c in X.columns if 'ord_' in c]



time_vars = ['day', 'month']
# Some feature engineering

X['ord_5_1'] = X['ord_5'].apply(lambda x: x[0] if type(x) == str else np.nan)

X['ord_5_2'] = X['ord_5'].apply(lambda x: x[1] if type(x) == str else np.nan)

Xt['ord_5_1'] = Xt['ord_5'].apply(lambda x: x[0] if type(x) == str else np.nan)

Xt['ord_5_2'] = Xt['ord_5'].apply(lambda x: x[1] if type(x) == str else np.nan)



ordinal_vars += ['ord_5_1', 'ord_5_2']
# Converting ordinal labels into ordered values

ordinals = {

    'ord_1' : {

        'Novice' : 0,

        'Contributor' : 1,

        'Expert' : 2,

        'Master' : 3,

        'Grandmaster' : 4

    },

    'ord_2' : {

        'Freezing' : 0,

        'Cold' : 1,

        'Warm' : 2,

        'Hot' : 3,

        'Boiling Hot' : 4,

        'Lava Hot' : 5

    }

}



def return_order(X, Xt, var_name):

    mode = X[var_name].mode()[0]

    el = sorted(set(X[var_name].fillna(mode).unique())|set(Xt[var_name].fillna(mode).unique()))

    return {v:e for e, v in enumerate(el)}



for mapped_var in ordinal_vars:

    if mapped_var not in ordinals:

        mapped_values = return_order(X, Xt, mapped_var)

        X[mapped_var].replace(mapped_values, inplace=True)

        Xt[mapped_var].replace(mapped_values, inplace=True)

    else:

        X[mapped_var].replace(ordinals[mapped_var], inplace=True)

        Xt[mapped_var].replace(ordinals[mapped_var], inplace=True)
# Creating a list of numpy values from high cardinality variables

X_cat, Xt_cat = list(), list()

categorical_counts = dict()



for hc in binary_vars+nominal_vars+ordinal_vars+time_vars:

    # Finding out the levels in each high cardinality variable

    levels = set(X[hc].astype(str).fillna("NAN").unique())|set(Xt[hc].astype(str).fillna("NAN").unique())

    levels = np.array(list(levels))

    # Counting the levels

    categorical_counts[hc] = len(levels)

    # Converting the levels into numeric values

    le = LabelEncoder()

    le.fit(np.ravel(levels.reshape(-1,1)))

    X_cat.append(le.transform(X[hc].astype(str).fillna("NAN").values))

    Xt_cat.append(le.transform(Xt[hc].astype(str).fillna("NAN").values))



print("Countings for high cardinality variables:")

print(categorical_counts)
# Enconding frequencies instead of labels (so we have some numeric variables)



def frequency_encoding(column, df, df_test=None):

    frequencies = df[column].value_counts().reset_index()

    df_values = df[[column]].merge(frequencies, how='left', 

                                   left_on=column, right_on='index').iloc[:,-1].values

    if df_test is not None:

        df_test_values = df_test[[column]].merge(frequencies, how='left', 

                                                 left_on=column, right_on='index').fillna(1).iloc[:,-1].values

    else:

        df_test_values = None

    return df_values, df_test_values



freq_encoded = list()



for column in X.columns:

    train_values, test_values = frequency_encoding(column, X, Xt)

    X[column+'_counts'] = train_values

    Xt[column+'_counts'] = test_values

    freq_encoded.append(column+'_counts')
# Target encoding of selected variables

import category_encoders as cat_encs



cat_feat_to_encode = binary_vars + ordinal_vars + nominal_vars + time_vars

smoothing = 0.3



enc_x = np.zeros(X[cat_feat_to_encode].shape)



for tr_idx, oof_idx in StratifiedKFold(n_splits=3, random_state=42, shuffle=True).split(X, y):

    encoder = cat_encs.TargetEncoder(cols=cat_feat_to_encode, smoothing=smoothing)

    

    encoder.fit(X[cat_feat_to_encode].iloc[tr_idx], y[tr_idx])

    enc_x[oof_idx, :] = encoder.transform(X[cat_feat_to_encode].iloc[oof_idx], y[oof_idx])

    

encoder.fit(X[cat_feat_to_encode], y)

enc_xt = encoder.transform(Xt[cat_feat_to_encode]).values



target_encoded = list()



for idx, new_var in enumerate(cat_feat_to_encode):

    new_var = new_var + '_enc'

    X[new_var] = enc_x[:,idx]

    Xt[new_var] = enc_xt[:, idx]

    target_encoded.append(new_var)
# The values are normalized using the Standard Scaler

ssc = StandardScaler()

selection = freq_encoded + target_encoded + ordinal_vars + time_vars

X_ohe = ssc.fit_transform(X[selection].fillna(X[selection].median()))

Xt_ohe = ssc.transform(Xt[selection].fillna(X[selection].median()))
# Adding the GELU and LEAKY RELU functions as custom objects 

# (see: https://datascience.stackexchange.com/questions/49522/what-is-gelu-activation)



def gelu(x):

    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))



get_custom_objects().update({'gelu': Activation(gelu)})



# Add leaky-relu so we can use it as a string

get_custom_objects().update({'leaky-relu': Activation(LeakyReLU(alpha=0.2))})
# Parametric DNN architecture



def tabular_dnn(numeric_variables, categorical_variables, categorical_counts,

                feature_selection_dropout=0.2, categorical_dropout=0.1,

                first_dense = 256, second_dense = 256, dense_dropout = 0.2, 

                activation_type=gelu):

    

    # Numeric inputs pipeline

    numerical_inputs = Input(shape=(numeric_variables,))

    numerical_normalization = BatchNormalization()(numerical_inputs)

    numerical_feature_selection = Dropout(feature_selection_dropout)(numerical_normalization)



    # Categorical inputs pipeline

    categorical_inputs = []

    categorical_embeddings = []

    for category in categorical_variables:

        categorical_inputs.append(Input(shape=[1], name=category))

        category_counts = categorical_counts[category]

        categorical_embeddings.append(

            Embedding(category_counts+1, 

                      min(int(category_counts/1.5 + 1), 3), 

                      name = category + "_embed")(categorical_inputs[-1]))



    categorical_logits = Concatenate(name = "categorical_conc")([Flatten()(SpatialDropout1D(categorical_dropout)(cat_emb)) 

                                                                 for cat_emb in categorical_embeddings])



    # Fully connected layers

    x = concatenate([numerical_feature_selection, categorical_logits])

    x = BatchNormalization()(x)

    

    x = Dense(first_dense, activation=activation_type)(x)

    x = BatchNormalization()(x)

    x = Dropout(dense_dropout)(x)

    

    x = Dense(second_dense, activation=activation_type)(x)

    x = Dropout(dense_dropout)(x)

    x = BatchNormalization()(x)

    

    # Sigmoid final activation

    output = Dense(1, activation="sigmoid")(x)

    

    # Composing the model -> input list of numeric and each high cardinality variable

    model = Model([numerical_inputs] + categorical_inputs, output)

    

    return model
# Useful functions for training DNNs



def auroc(y_true, y_pred):

    try:

        return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

    except:

        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)



get_custom_objects().update({'auroc': auroc})  



def mAP(y_true, y_pred):

    try:

        return tf.py_function(average_precision_score, (y_true, y_pred), tf.double)

    except:

        return tf.py_func(average_precision_score, (y_true, y_pred), tf.double)

    

get_custom_objects().update({'mAP': mAP})



def compile_model(model, loss, metrics, optimizer):

    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    return model



def plot_keras_history(history, measures):

    """

    history: Keras training history

    measures = list of names of measures

    """

    rows = len(measures) // 2 + len(measures) % 2

    fig, panels = plt.subplots(rows, 2, figsize=(15, 5))

    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)

    try:

        panels = [item for sublist in panels for item in sublist]

    except:

        pass

    for k, measure in enumerate(measures):

        panel = panels[k]

        panel.set_title(measure + ' history')

        panel.plot(history.epoch, history.history[measure], label="Train "+measure)

        panel.plot(history.epoch, history.history["val_"+measure], label="Validation "+measure)

        panel.set(xlabel='epochs', ylabel=measure)

        panel.legend()

        

    plt.show(fig)
# train/validation batch generator

def batch_generator(X_ohe, X_cat, y, cv=5, batch_size=64, random_state=None):

    '''

    Returns a batch from X, y

    random_state allows determinism

    different scikit-learn CV strategies are possible

    '''

    folds = len(y) // batch_size

    if isinstance(cv, int):

        kf = StratifiedKFold(n_splits=cv, 

                              shuffle=True, 

                              random_state=random_state)

    else:

        kf = cv

    

    while True:

        for _, batch_index in kf.split(X_ohe, y):

            numeric_input = X_ohe[batch_index].astype(np.float32)

            categorical_input = [X_cat[i][batch_index] for i in range(len(X_cat))]

            target = y[batch_index]

            yield [numeric_input] + categorical_input, target
# Global training settings

SEED = 42

FOLDS = 20

MAX_EPOCHS = 100

BATCH_SIZE = 1024 * 4
# Defining callbacks

measure_to_monitor = 'val_auroc' 

modality = 'max'



early_stopping = EarlyStopping(monitor=measure_to_monitor, 

                               mode=modality, 

                               patience=5, 

                               verbose=0)



model_checkpoint = ModelCheckpoint('best.model', 

                                   monitor=measure_to_monitor, 

                                   mode=modality, 

                                   save_best_only=True, 

                                   verbose=0)



model_reduce_lr = ReduceLROnPlateau(monitor=measure_to_monitor,

                                    mode=modality,

                                    factor=0.25,

                                    patience=2, 

                                    min_lr=1e-6, 

                                    verbose=1)
# Defining model

model_params = {

    "numeric_variables" : X_ohe.shape[1], 

    "categorical_variables" : categorical_counts.keys(),

    "categorical_counts" : categorical_counts, 

    "feature_selection_dropout" : 0.0,

    "categorical_dropout" : 0.3,

    "first_dense" : 512,

    "second_dense" : 512,

    "dense_dropout" : 0.3,

    "activation_type" : 'relu'

}
# Setting the CV strategy

skf = StratifiedKFold(n_splits=FOLDS, 

                      shuffle=True, 

                      random_state=SEED)



# CV Iteration: we store best epochs, oof and cv testv prediciton

roc_auc = list()

average_precision = list()

oof = np.zeros(len(X))

best_iteration = list()

cv_test_preds = np.zeros(len(Xt))



for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

    

    # Re-instantiating the model

    model = compile_model(tabular_dnn(**model_params), 

                          binary_crossentropy, 

                          [auroc, mAP], 

                          Adam(learning_rate=0.0001))

    

    # Creating the train and validation sets

    X_cv_ohe = X_ohe[train_idx].astype(np.float32)

    X_cv_cat = [X_cat[i][train_idx] for i in range(len(X_cat))]

    y_cv = y[train_idx]

    X_oof_ohe = X_ohe[test_idx].astype(np.float32)

    X_oof_cat = [X_cat[i][test_idx] for i in range(len(X_cat))]

    y_oof = y[test_idx]

    

    # Instantiating the train and validation generators

    train_batch = batch_generator(X_cv_ohe,

                                  X_cv_cat,

                                  y_cv,

                                  batch_size=BATCH_SIZE,

                                  random_state=SEED)

    

    val_batch = batch_generator(X_oof_ohe,

                                X_oof_cat,

                                y_oof,

                                batch_size=BATCH_SIZE,

                                random_state=SEED)

    

    train_steps = len(y_cv) // BATCH_SIZE 

    validation_steps = len(y_oof) // BATCH_SIZE

    

    # Training

    history = model.fit_generator(train_batch,

                                  validation_data=val_batch,

                                  epochs=MAX_EPOCHS,

                                  steps_per_epoch=train_steps,

                                  validation_steps=validation_steps,

                                  callbacks=[model_checkpoint, early_stopping, model_reduce_lr],

                                  #class_weight=[1.0, (np.sum(y_cv==0) / np.sum(y_cv==1))],

                                  verbose=1)

    

    # Reporting

    print("\nFOLD %i" % fold)

    plot_keras_history(history, measures=['auroc', 'loss'])

    

    # OOF prediction

    best_iteration.append(np.argmax(history.history['val_auroc']) + 1)

    

    model = load_model('best.model')

    

    preds = model.predict([X_oof_ohe] + X_oof_cat,

                          verbose=1,

                          batch_size=1024).flatten()



    oof[test_idx] = preds



    roc_auc.append(roc_auc_score(y_true=y_oof, y_score=preds))

    

    average_precision.append(average_precision_score(y_true=y_oof, y_score=preds))

    

    # CV test prediction

    cv_test_preds += model.predict([Xt_ohe] + Xt_cat,

                                   verbose=1,

                                   batch_size=1024).flatten() / FOLDS
# Storing results to disk

oof = pd.DataFrame({'id':id_train, 'dnn_oof': oof})

oof.to_csv("oof.csv", index=False)



submission = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/sample_submission.csv")

submission.target = cv_test_preds

submission.to_csv("./dnn_cv_submission.csv", index=False)
# Reporting from training

print("Average cv roc auc score %0.3f ± %0.3f" % (np.mean(roc_auc), np.std(roc_auc)))

print("Average cv roc average precision %0.3f ± %0.3f" % (np.mean(average_precision), np.std(average_precision)))



print("Roc auc score OOF %0.3f" % roc_auc_score(y_true=y, y_score=oof.dnn_oof))

print("Average precision OOF %0.3f" % average_precision_score(y_true=y, y_score=oof.dnn_oof))
# We now train on all the examples, using a rule of thumb for the number of iterations

train_batch = batch_generator(X_ohe, X_cat, y,

                              batch_size=BATCH_SIZE,

                              random_state=SEED)



train_steps = len(y) // BATCH_SIZE



model = compile_model(tabular_dnn(**model_params), 

                      binary_crossentropy, 

                      [auroc, mAP], 

                      Adam(learning_rate=0.0001))



history = model.fit_generator(train_batch,

                              epochs=int(np.median(best_iteration)),

                              steps_per_epoch=train_steps,

                              #class_weight=[1.0, (np.sum(y==0) / np.sum(y==1))],

                              verbose=1)
# Predicting and final submission

test_preds = model.predict([Xt_ohe] + Xt_cat,

                           verbose=1,

                           batch_size=1024).flatten()



submission = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/sample_submission.csv")

submission.target = test_preds

submission.to_csv("./submission.csv", index=False)
# Blend

submission.target = (submission.target + pd.read_csv("./dnn_cv_submission.csv").target) / 2

submission.to_csv("./blend_submission.csv", index=False)