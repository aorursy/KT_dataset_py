from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Lambda, Flatten

from keras.optimizers import Adam, SGD, RMSprop

from keras.callbacks import Callback, EarlyStopping

import keras.backend as K

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from keras.regularizers import l1, l2, l1_l2

import warnings

from sklearn import metrics

from sklearn.utils.class_weight import compute_class_weight

from sklearn.preprocessing import StandardScaler, MinMaxScaler



from imblearn.keras import BalancedBatchGenerator



import time

import gc



# Charts

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
import tflearn



def roc_loss(y_true, y_pred):

    return tflearn.objectives.roc_auc_score(y_pred, y_true)



alpha=0.25

gamma=2.0

    

def focal_loss(y_true, y_pred):

    bce = K.binary_crossentropy(y_true, y_pred)



    y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())

    p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))

    alpha_t =  y_true*alpha + ((1-alpha)*(1-y_true))



    return K.mean(alpha_t*K.pow((1-p_t), gamma)*bce, axis=-1)
epochs = 1000



batch_size = 1000

patience = 10

scaler = StandardScaler()

expand_data = True

n_splits = 10



max_evals = 5



model_type = "conv1d"

conv_activation = "relu"

batch_normalization = True

dropout = True



import featuretools as ft

trans_primitives = ['divide_by_feature']
from sklearn.preprocessing import power_transform



train_data = pd.read_csv('../input/train.csv')



def rank_gauss(data):

    df = pd.DataFrame()

    for i in data.columns:

        df[i] = rank_gauss_element(data[i].values)

    return df



def rank_gauss_element(x):

    from scipy.special import erfinv

    N = x.shape[0]

    temp = x.argsort()

    rank_x = temp.argsort() / N

    rank_x -= rank_x.mean()

    rank_x *= 2

    efi_x = erfinv(rank_x)

    efi_x -= efi_x.mean()

    return efi_x



def features_preprocessing(data):

    if expand_features:

        features = expand_features(data)

        #features = deep_feature_synthesis(data)

    else:

        features = data

    return scale_features(features)



def deep_feature_synthesis(data):

    # Deep Feature Synthesis

    columns = [c for c in data.columns if c not in ['target']]

    es = ft.EntitySet(id = 'transactions')

    es.entity_from_dataframe(entity_id = 'data', 

                             dataframe = data[columns], 

                             index = 'ID_code')

    

    feature_matrix, feature_names = ft.dfs(entityset=es, 

                   target_entity = 'data', 

                   max_depth = 1,

                   agg_primitives = [],

                   trans_primitives = trans_primitives,

                   verbose = 1,

                   n_jobs = -1,

                   chunk_size = 100)

    

    return feature_matrix

    

def scale_features(data):

    columns = [c for c in data.columns if c not in ['ID_code', 'target']]

    if scaler is not None:

        if scaler == "rank_gauss":

            return rank_gauss(data[columns])

        else:

            return pd.DataFrame(scaler.fit_transform(data[columns]), columns=columns)

    else:

        return data[columns]



def expand_features(train_features):

    features = [c for c in train_features.columns if c not in ['ID_code', 'target']]

    for feature in features:

        train_features['mean_'+feature] = (train_features[feature].mean()-train_features[feature])

        train_features['z_'+feature] = (train_features[feature] - train_features[feature].mean())/train_features[feature].std(ddof=0)

        train_features['sq_'+feature] = (train_features[feature])**2

        train_features['sqrt_'+feature] = (train_features['sq_'+feature])**(1/4)

        train_features['log_'+feature] = np.log(train_features['sq_'+feature]+10)/2

        

    return train_features
preprocess_train_data = features_preprocessing(train_data)



number_features = len(preprocess_train_data.columns)



X_train, X_test, y_train, y_test = train_test_split(preprocess_train_data, train_data['target'], stratify=train_data['target'], train_size = 0.9, test_size = 0.1, random_state = 1337)



preprocess_train_data.describe()
def class_weights(y) :

    return dict(enumerate(compute_class_weight("balanced", np.unique(y), y)))
import sklearn



def roc_chart(fpr, tpr, auc_score, label):

    plt.figure(1)

    plt.plot(fpr, tpr, label='{0} (area = {1:.3f})'.format(label, auc_score))

    plt.xlabel('False positive rate')

    plt.ylabel('True positive rate')

    plt.title('ROC curve')

    plt.legend(loc='best')

    plt.show()

    

def roc_calculation_chart(x, y, label):

    y_pred = model.predict(x)

    roc_score = sklearn.metrics.roc_auc_score(y, y_pred)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, y_pred)

    roc_chart(fpr, tpr, roc_score, label)

    

def roc_calculation(model, x_train, y_train, x_test, y_test):

    roc_calculation_chart(x_train, y_train, "roc-auc")

    roc_calculation_chart(x_test, y_test, "val-roc-auc")



class roc_callback(Callback):

    def __init__(self,training_data,validation_data):

        self.x = training_data[0]

        self.y = training_data[1]

        self.x_val = validation_data[0]

        self.y_val = validation_data[1]



    def on_train_begin(self, logs={}):

        return



    def on_train_end(self, logs={}):

        return



    def on_epoch_begin(self, epoch, logs={}):

        return



    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict(self.x)

        roc = sklearn.metrics.roc_auc_score(self.y, y_pred)

        y_pred_val = self.model.predict(self.x_val)

        roc_val = sklearn.metrics.roc_auc_score(self.y_val, y_pred_val)

        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')

        return



    def on_batch_begin(self, batch, logs={}):

        return



    def on_batch_end(self, batch, logs={}):

        return
def add_dense_layer(model, dense_size = 1024, dense_activation = 'relu', dropout_percentage = 0.5, kernel_regularizer = None, batch_normalization = False, dropout = False, hasInput = False):

    layer_name = 'dense_' + str(dense_size)

    if hasInput:

        model.add(Dense(dense_size, activation=dense_activation, kernel_regularizer = kernel_regularizer, name=layer_name, input_shape=(number_features,)))

    else:

        model.add(Dense(dense_size, activation=dense_activation, kernel_regularizer = kernel_regularizer, name=layer_name))

    model.add(BatchNormalization()) if (batch_normalization) else False 

    model.add(Dropout(dropout_percentage)) if (dropout) else False



def add_conv1d_layer(model, filter_size, batch_normalization, dropout, hasInput, number_per_layer_conv, dropout_percentage, kernel_size, pooling): 

    if hasInput:

        layer_name = 'conv1d_' + str(filter_size) + '_input_layer'

        model.add(Conv1D(filters = filter_size, kernel_size = (kernel_size), padding = 'Same', activation = conv_activation, name=layer_name, input_shape=(number_features,1))) 

        model.add(BatchNormalization()) if (batch_normalization) else False 

        model.add(Dropout(dropout_percentage)) if (dropout) else False

        number_per_layer_conv -= 1

    

    for layer_number in range(number_per_layer_conv):

        layer_name = 'conv1d_' + str(filter_size) + '_' + str(layer_number + 1)

        model.add(Conv1D(filters = filter_size, kernel_size = (kernel_size), padding = 'Same', activation = conv_activation, name=layer_name))

        model.add(BatchNormalization()) if (batch_normalization) else False   

        model.add(Dropout(dropout_percentage)) if (dropout) else False

    

    model.add(pooling(pool_size=(2),strides=(2)))

    model.add(BatchNormalization()) if (batch_normalization) else False   

    model.add(Dropout(dropout_percentage)) if (dropout) else False

    

def add_all_conv1d_layers(model, number_conv1d_layers, number_dense_layers, number_per_layer_conv, first_dense_layer, first_filter_layer, dense_activation, dropout_percentage, kernel_regularizer, batch_normalization, dropout, kernel_size, pooling, global_pooling):

    first_layer = first_filter_layer

    for layer_number in range(number_conv1d_layers):

        add_conv1d_layer(model, int(first_layer), False, False, first_layer == first_filter_layer, number_per_layer_conv, dropout_percentage, kernel_size, pooling)

        first_layer *= 2

    

    if (number_conv1d_layers > 0):

        model.add(global_pooling())

        #model.add(BatchNormalization()) if (batch_normalization) else False  

        model.add(Dropout(dropout_percentage)) if (dropout) else False

    

    add_all_dense_layers(model, number_dense_layers, first_dense_layer, dense_activation, dropout_percentage, kernel_regularizer, batch_normalization, dropout, number_conv1d_layers <= 0)

    

    

def add_all_dense_layers(model, number_layers = 5, first_dense_layer = 512, dense_activation = 'relu', dropout_percentage = 0.5, kernel_regularizer = None, batch_normalization = False, dropout = False, addInput = True):

    first_layer = first_dense_layer

    for layer_number in range(number_layers):

        add_dense_layer(model, int(first_layer), dense_activation, dropout_percentage, kernel_regularizer, batch_normalization, dropout, first_layer == first_dense_layer and addInput)

        first_layer /= 2



def create_dense_model(model):

    add_all_dense_layers(model, number_of_dense_layers, kernel_regularizer, batch_normalization, dropout)



def create_conv1d_model(model, number_of_dense_layers, number_of_conv1d_layers, number_per_layer_conv, first_dense_layer, first_filter_layer, dense_activation, dropout_percentage, kernel_size, kernel_regularizer, pooling, global_pooling):

    if (number_of_conv1d_layers > 0):

        model.add(Lambda(lambda i: K.expand_dims(i, axis=2), input_shape=(number_features,)))

    add_all_conv1d_layers(model, number_of_conv1d_layers, number_of_dense_layers, number_per_layer_conv, first_dense_layer, first_filter_layer, dense_activation, dropout_percentage, kernel_regularizer, batch_normalization, dropout, kernel_size, pooling, global_pooling)



def create_model(number_of_dense_layers, number_of_conv1d_layers, number_per_layer_conv, first_dense_layer, first_filter_layer, dense_activation, dropout_percentage, kernel_size, kernel_regularizer, pooling, global_pooling):

    model = Sequential()

    if model_type == 'dense':

        create_dense_model(model)

    else:

        create_conv1d_model(model, number_of_dense_layers, number_of_conv1d_layers, number_per_layer_conv, first_dense_layer, first_filter_layer, dense_activation, dropout_percentage, kernel_size, kernel_regularizer, pooling, global_pooling)

    

    return model



def build_model(number_of_dense_layer, 

                number_of_conv1d_layers, 

                number_per_layer_conv,

                first_dense_layer,

                first_filter_layer,

                dense_activation,

                dropout_percentage,

                kernel_size,

                kernel_regularizer,

                pooling,

                global_pooling,

                optimizer,

                loss):

    model = create_model(number_of_dense_layer, number_of_conv1d_layers, number_per_layer_conv, first_dense_layer, first_filter_layer, dense_activation, dropout_percentage, kernel_size, kernel_regularizer, pooling, global_pooling)

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid', name='binary_classification'))

    

    model.compile(loss=loss['function'], optimizer=optimizer, metrics=[roc_loss,'acc',focal_loss])

    

    return model



def train_model(model, X_train, X_test, y_train, y_test, batch_size, epochs, weights):

    validation_steps = int(len(X_test) / batch_size) + 1

    steps_per_epoch = int(len(X_train) / batch_size) + 1

    training_generator = BalancedBatchGenerator(X_train, y_train,

                                                batch_size=batch_size,

                                                random_state=42)

    earlystopper = EarlyStopping(monitor='val_focal_loss', patience=patience, verbose=1, restore_best_weights=True, mode='auto')

    roc = roc_callback(training_data=(X_train, y_train),validation_data=(X_test, y_test))

    return model.fit_generator(generator=training_generator, 

                              validation_data=(X_test,y_test),

                              callbacks=[earlystopper],

                              epochs=epochs, 

                              verbose=0,

                              class_weight = weights,

                              steps_per_epoch=steps_per_epoch,

                              validation_steps=validation_steps)
from sklearn.model_selection import StratifiedShuffleSplit



def train_k_folds(model, epochs, batch_size, weights, n_splits, train_features, train_targets):

    sss = StratifiedShuffleSplit(n_splits=n_splits)

    fold_number = 0

    for train_index, test_index in sss.split(train_features, train_targets):

        start_time = time.time()

        X_train, X_test = train_features[train_index], train_features[test_index]

        y_train, y_test = train_targets[train_index], train_targets[test_index]

        

        training = train_model(model, X_train, X_test, y_train, y_test, batch_size, epochs, weights)

        

        eval_model(model, training, X_test, y_test)

        test_pred = model.predict_classes(X_test)

        eval_accuracy(y_test, test_pred)

        

        del X_train, X_test, y_train, y_test, training, test_pred

        gc.collect()

        elapsed_time = time.time() - start_time

        print("Fold {0} run in {1}".format(fold_number + 1, elapsed_time))

        fold_number += 1

    

    return
def eval_model(model, training, X_test, y_test):

    """

    Model evaluation: plots, classification report

    @param training: model training history

    @param model: trained model

    """

    if training is not None:

        ## Trained model analysis and evaluation

        f, ax = plt.subplots(3,1, figsize=(10,10))

        ax[0].plot(training.history['loss'], label="Loss")

        ax[0].plot(training.history['val_loss'], label="Validation loss")

        ax[0].set_title('Loss')

        ax[0].set_xlabel('Epoch')

        ax[0].set_ylabel('Loss')

        ax[0].legend()



        # Accuracy

        ax[1].plot(training.history['acc'], label="Accuracy")

        ax[1].plot(training.history['val_acc'], label="Validation accuracy")

        ax[1].set_title('Accuracy')

        ax[1].set_xlabel('Epoch')

        ax[1].set_ylabel('Accuracy')

        ax[1].legend()



        # Roc Loss

        ax[2].plot(training.history['roc_loss'], label="Roc Loss")

        ax[2].plot(training.history['val_roc_loss'], label="Validation roc loss")

        ax[2].set_title('Roc Loss')

        ax[2].set_xlabel('Epoch')

        ax[2].set_ylabel('Roc Loss')

        ax[2].legend()

        plt.tight_layout()

        plt.show()

    

    test_res = model.evaluate(X_test, y_test, verbose=0)

    for i, model_name in enumerate(model.metrics_names):

        print('{0}: {1}'.format(model_name, test_res[i]))



    return test_res



def eval_accuracy(test_truth, test_pred):

    # Print metrics

    print("Classification report")

    

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        print(metrics.classification_report(test_truth, test_pred))
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval



space = { 

    'number_of_dense_layers': hp.choice('number_of_dense_layers',[3]),

    'number_of_conv1d_layers': hp.choice('number_of_conv1d_layers',[2,3]),

    'number_per_layer_conv': hp.choice('number_per_layer_conv',[2]),

    'first_dense_layer': hp.choice('first_dense_layer', [1024, 512]),

    'first_filter_layer': hp.choice('first_filter_layer', [32, 64]),

    'dense_activation': hp.choice('dense_activation', ['tanh']),

    'dropout_percentage': hp.choice('dropout_percentage',[0.75, 0.90]),

    'kernel_size': hp.choice('kernel_size', [3,5]),

    'regularizer': hp.choice('regularizer', [l1_l2(0.01)]),

    'pooling': hp.choice('pooling', [AveragePooling1D]),

    'global_pooling': hp.choice('global_pooling', [Flatten]),

    'optimizer': hp.choice('optimizer', [RMSprop(lr=1e-4)]),

    'loss': hp.choice('loss', [{ 'name': 'focal_loss', 'function': focal_loss }])

}
def objective(params):

    print(params)

    model = build_model(

        params['number_of_dense_layers'],

        params['number_of_conv1d_layers'],

        params['number_per_layer_conv'],

        params['first_dense_layer'],

        params['first_filter_layer'],

        params['dense_activation'],

        params['dropout_percentage'],

        params['kernel_size'],

        params['regularizer'],

        params['pooling'],

        params['global_pooling'],

        params['optimizer'],

        params['loss']

    )

    train_k_folds(model, epochs, batch_size, weights, n_splits, X_train.values, y_train.values)

    

    loss = eval_model(model, None, X_test.values, y_test.values)

    

    y_pred = model.predict(X_test)

    roc_score = sklearn.metrics.roc_auc_score(y_test, y_pred)

    

    del y_pred

    gc.collect()

    

    return {'loss': loss[model.metrics_names.index(params['loss']['name'])], 'status': STATUS_OK, 'roc_score': roc_score, 'best_model': model}
weights = class_weights(train_data['target'].values)

print(weights)
trials = Trials()



best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
print (trials.best_trial['result'])

print(best)

print(space_eval(space, best))

model = trials.best_trial['result']['best_model']
params = dict()

params['number_of_dense_layers'] = 2

params['number_of_conv1d_layers'] = 2

params['number_per_layer_conv'] = 2

params['first_dense_layer'] = 512

params['first_filter_layer'] = 16

params['dense_activation'] = 'relu'

params['dropout_percentage'] = 0.50

params['kernel_size'] = 3

params['regularizer'] = l2(0.01)

params['pooling'] = MaxPooling1D

params['global_pooling'] = Flatten

params['optimizer'] = Adam(lr=1e-4)

params['loss'] = { 'name': 'focal_loss', 'function': focal_loss }



model = build_model(

        params['number_of_dense_layers'],

        params['number_of_conv1d_layers'],

        params['number_per_layer_conv'],

        params['first_dense_layer'],

        params['first_filter_layer'],

        params['dense_activation'],

        params['dropout_percentage'],

        params['kernel_size'],

        params['regularizer'],

        params['pooling'],

        params['global_pooling'],

        params['optimizer'],

        params['loss']

    )

#training = train_model(model, X_train.values, X_test.values, y_train.values, y_test.values, batch_size, epochs, None)
model.summary()
eval_model(model, None, X_test.values, y_test.values)
test_pred = model.predict_classes(X_test)

eval_accuracy(y_test, test_pred)
roc_calculation(model, X_train, y_train, X_test, y_test)
del X_train, X_test, y_train, y_test, train_data, preprocess_train_data

gc.collect()
test_data = pd.read_csv('../input/test.csv')

X_predictions = test_data.drop(columns=['ID_code'], axis=1)

predictions = model.predict(features_preprocessing(X_predictions))

file_submission = pd.DataFrame(test_data['ID_code'])

file_submission['target'] = predictions

file_submission.to_csv('santander_predictions.csv', index=False)