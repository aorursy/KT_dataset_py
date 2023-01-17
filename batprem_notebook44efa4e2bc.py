import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
seed_value= 0



import os

os.environ['PYTHONHASHSEED']=str(seed_value)



import random

random.seed(seed_value)



import numpy as np

np.random.seed(seed_value)



import tensorflow as tf

tf.random.set_seed(seed_value)

train_feature = pd.read_csv('../input/lish-moa/train_features.csv')

X_train = train_feature.drop('sig_id', axis = 1)
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

Y_train = train_targets_scored.drop('sig_id', axis = 1)
Y_train
combine = pd.concat([X_train, Y_train], axis=1, sort=False)

combine
def get_conditional_prob(feature, observation):

    '''

    Function for returning a resulting dataframe

    '''

    return (combine[[feature, observation]].groupby([feature], as_index=False).

             mean().

             sort_values(by=observation, ascending=False))



x_col = X_train.columns

y_col = Y_train.columns



print("I have an assumsion that if cp_type is ctl_vehicle then y will be all zero")

ctl_vehicle_is_zero = True

for y in y_col:

    

    temp = get_conditional_prob(feature=x_col[0], observation=y)

    if temp[temp['cp_type'] == 'ctl_vehicle'][y][0] > 0:

        # If y result > 0 was found

        ctl_vehicle_is_zero = False

        print(temp)

    # print(temp[temp['cp_type'] == 'ctl_vehicle'][y][0])



if ctl_vehicle_is_zero:

    print("That's true")

else:

    print("Just Misunderstood")
cp_time_to_MOA = pd.DataFrame()

import scipy.stats as stats





for y in y_col:

    temp = get_conditional_prob(feature=x_col[1], observation=y)

    temp_2 = temp.T.rename(columns=temp.T.iloc[0]).iloc[1:]

    cp_time_to_MOA = pd.concat([cp_time_to_MOA, temp_2])

cp_time_to_MOA
stats.ttest_rel(cp_time_to_MOA[24],cp_time_to_MOA[48])
stats.ttest_rel(cp_time_to_MOA[24],cp_time_to_MOA[72])
stats.ttest_rel(cp_time_to_MOA[48],cp_time_to_MOA[72])
stats.f_oneway(cp_time_to_MOA[24], cp_time_to_MOA[48], cp_time_to_MOA[72])
cp_dose_to_MOA = pd.DataFrame()

import scipy.stats as stats



# stats.f_oneway(cp_time_to_MOA[24], cp_time_to_MOA[48], cp_time_to_MOA[72])

for y in y_col:

    temp = get_conditional_prob(feature=x_col[2], observation=y)

    temp_2 = temp.T.rename(columns=temp.T.iloc[0]).iloc[1:]

    cp_dose_to_MOA = pd.concat([cp_dose_to_MOA, temp_2])

cp_dose_to_MOA
stats.ttest_rel(cp_dose_to_MOA['D1'],cp_dose_to_MOA['D2'])
stats.ttest_rel([10,10],[1,1])
cp_time_to_MOA[24].values
X_train[X_train.columns[3:]]
y_col = Y_train.columns[1]

combine.loc[Y_train[Y_train[y_col]==1].index]
all_zero = True

for yc in Y_train.columns:

    temp = (Y_train[yc] == 0)

    if type(all_zero) is bool :

        all_zero = temp

    else:

        all_zero = all_zero & temp

        
X_train_wraggled = X_train[X_train['cp_type'] != 'ctl_vehicle'][X_train.columns[3:]]
Y_train_wraggled = Y_train.loc[X_train_wraggled.index]
X_train_wraggled = X_train_wraggled.loc[Y_train_wraggled.index]
X_train_wraggled_g = X_train_wraggled[X_train_wraggled.columns[pd.Series(X_train_wraggled.columns).str.startswith('g')]] 



X_train_wraggled_c = X_train_wraggled[X_train_wraggled.columns[pd.Series(X_train_wraggled.columns).str.startswith('c')]] 
print(X_train_wraggled.shape[1])

print(X_train_wraggled_g.shape[1])

print(X_train_wraggled_c.shape[1])
Y_train_wraggled[Y_train_wraggled.sum(axis=1)== 1]
os.listdir('../input/moa-lstm')
X_train_wraggled_c
def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

              

def calculating_class_weights(y_true):

    from sklearn.utils.class_weight import compute_class_weight

    number_dim = np.shape(y_true)[1]

    weights = np.empty([number_dim, 2])

    for i in range(number_dim):

        weights[i] = compute_class_weight('balanced', [0,1], y_true.iloc[:, i])

    return weights



def get_weighted_loss(weights):

    def weighted_loss(y_true, y_pred):

        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)

    return weighted_loss



import tensorflow as tf



def f1(y_true, y_pred):

    y_pred = K.round(y_pred)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)

    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f1 = 2*p*r / (p+r+K.epsilon())

    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)



def f1_loss(y_true, y_pred):

    

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)

    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f1 = 2*p*r / (p+r+K.epsilon())

    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    return 1 - K.mean(f1)



class_weights = calculating_class_weights(Y_train_wraggled)
class_weights_2 = np.copy(class_weights)

class_weights[:,0] =1

class_weights[:,1] = 1.15
class_weights
# Fully connect

import keras

from keras import backend as K

from numpy import loadtxt

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.callbacks import ModelCheckpoint,EarlyStopping

from keras.layers import Input, Concatenate, concatenate, BatchNormalization

from keras.models import Model

from keras.layers import Dense, Conv1D, GlobalMaxPooling1D

from keras.layers import LSTM, Reshape

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from keras.callbacks import TensorBoard

from tensorflow_addons.layers import WeightNormalization



model_name = "conv"

model_name = "parallel_conv"

#model_name = "LSTM"



SPLIT_RATIO = 0.75

# model_name = 'g_conv'

# model_name = 'c_lstm'

#Load pretrained model

from keras.models import load_model







#Create model function

def getModel(model_name):

    if model_name == 'pretrained':

        model = load_model('../input/moa-lstm/best_weights (1).hdf5')

        toBeCompiled = False

    elif model_name == 'conv':

        InputLayer = Input(shape=(X_train_wraggled.shape[1], 1))

        ConvLayer = Conv1D(filters=20,

                           kernel_size=10,

                           padding='valid',

                           activation='relu',

                           strides=1)(InputLayer)

        PoolingLayer = GlobalMaxPooling1D()(ConvLayer)

        OutputLayer = Dense(206, activation='sigmoid')(PoolingLayer)

        model = Model(inputs=InputLayer, outputs=OutputLayer)

        toBeCompiled = True

        

    elif model_name == 'parallel_conv':

        InputLayer_g = Input(shape=(X_train_wraggled_g.shape[1], 1))

        InputLayer_c = Input(shape=(X_train_wraggled_c.shape[1], 1))

        

        

        ConvLayer_g = Conv1D(filters=120,

                           kernel_size=50,

                           padding='valid',

                           activation='relu',

                           strides=1)(InputLayer_g)

        PoolingLayer_g = GlobalMaxPooling1D()(ConvLayer_g)

        PoolingLayer_g = BatchNormalization()(PoolingLayer_g)

        PoolingLayer_g = Dropout(0.4)(PoolingLayer_g)

        

        ConvLayer_c = Conv1D(filters=120,

                           kernel_size=50,

                           padding='valid',

                           activation='relu',

                           strides=1)(InputLayer_c)

        PoolingLayer_c = GlobalMaxPooling1D()(ConvLayer_c)

        PoolingLayer_c = BatchNormalization()(PoolingLayer_c)

        PoolingLayer_c = Dropout(0.4)(PoolingLayer_c)

        

        merged = concatenate([PoolingLayer_g, PoolingLayer_c], axis=1)

        merged = BatchNormalization()(merged)

        merged = Reshape((240,1))(merged)

        

        merged = WeightNormalization(LSTM(500))(merged)

        merged = BatchNormalization()(merged)

        merged = Dropout(0.2)(merged)

        

        merged = WeightNormalization(Dense(1000, activation='relu'))(merged)

        merged = BatchNormalization()(merged)

        merged = Dropout(0.2)(merged)

        



        

#         merged = WeightNormalization(Dense(1000, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.2)(merged)

        

#         merged = WeightNormalization(Dense(1000, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.2)(merged)

        

#         merged = WeightNormalization(Dense(1000, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.2)(merged)

        

#         merged = WeightNormalization(Dense(1000, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.2)(merged)

        

#         merged = WeightNormalization(Dense(1000, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.2)(merged)

        

#         merged = WeightNormalization(Dense(1000, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.2)(merged)

        

#         merged = WeightNormalization(Dense(1000, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.2)(merged)

        

#         merged = WeightNormalization(Dense(1000, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.2)(merged)

        

#         merged = WeightNormalization(Dense(1000, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.2)(merged)

        

#         merged = WeightNormalization(Dense(1000, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.2)(merged)

        

        OutputLayer = WeightNormalization(Dense(206, activation='sigmoid'))(merged)

        model = Model(inputs=[InputLayer_g,InputLayer_c], outputs=OutputLayer)

        toBeCompiled = True

        

    elif model_name == 'LSTM':

        InputLayer_g = Input(shape=(X_train_wraggled_g.shape[1], 1))

        InputLayer_c = Input(shape=(X_train_wraggled_c.shape[1], 1))

        

        LSTM_g = LSTM(200)(InputLayer_g)

        LSTM_g = BatchNormalization()(LSTM_g)

        LSTM_c = LSTM(200)(InputLayer_c)

        LSTM_c = BatchNormalization()(LSTM_c)

        merged = concatenate([LSTM_g, LSTM_c], axis=1)

        merged = BatchNormalization()(merged)

#         merged = Reshape((400,1))(merged)

        

#         merged = WeightNormalization(LSTM(400, dropout = 0.4))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.2)(merged)

        

#         merged = WeightNormalization(Dense(300, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.4)(merged)

        

#         merged = WeightNormalization(Dense(300, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.4)(merged)

        

#         merged = WeightNormalization(Dense(300, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.4)(merged)

        

#         merged = WeightNormalization(Dense(300, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.4)(merged)

        

#         merged = WeightNormalization(Dense(300, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.4)(merged)

        

#         merged = WeightNormalization(Dense(300, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.4)(merged)

        

#         merged = WeightNormalization(Dense(300, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.4)(merged)

        

#         merged = WeightNormalization(Dense(300, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.4)(merged)

        

#         merged = WeightNormalization(Dense(300, activation='relu'))(merged)

#         merged = BatchNormalization()(merged)

#         merged = Dropout(0.4)(merged)

        

        merged = WeightNormalization(Dense(300, activation='relu'))(merged)

        merged = BatchNormalization()(merged)

        merged = Dropout(0.4)(merged)

        

        OutputLayer = WeightNormalization(Dense(206, activation='sigmoid'))(merged)

        model = Model(inputs=[InputLayer_g,InputLayer_c], outputs=OutputLayer)

        toBeCompiled = True

    return toBeCompiled,model



toBeCompiled,model = getModel(model_name)

METRICS = [

        'accuracy',

        "binary_crossentropy",

        f1_m,

        keras.metrics.TruePositives(name='tp'),

        keras.metrics.FalsePositives(name='fp'),

        keras.metrics.TrueNegatives(name='tn'),

        keras.metrics.FalseNegatives(name='fn'), 

        keras.metrics.Precision(name='precision'),

        keras.metrics.Recall(name='recall'),

        keras.metrics.AUC(name='auc'),

    ]

if toBeCompiled:

    model.compile(loss=get_weighted_loss(class_weights),

                  #loss="binary_crossentropy",

                  optimizer='adam',

                  metrics=METRICS)







if model_name in ['parallel_conv','LSTM']:

    #Set Checkpoint

    filepath="best_weights.hdf5"

    Monitor = 'val_binary_crossentropy'

    #Monitor = 'binary_crossentropy'

    checkpoint = ModelCheckpoint(filepath, monitor=Monitor, verbose=1, save_best_only=True, mode='min')

    early_stop =EarlyStopping(monitor=Monitor, mode = 'min', patience=30)

    

    validation = (

                      [

                        X_train_wraggled_g.iloc[int(len(X_train_wraggled_g)*SPLIT_RATIO):].values.astype('float32'),

                        X_train_wraggled_c.iloc[int(len(X_train_wraggled_c)*SPLIT_RATIO):].values.astype('float32')

                    ],

                      Y_train_wraggled.iloc[int(len(X_train_wraggled)*SPLIT_RATIO):].values.astype('float32')

                  )

    # validation = None

    

    model.summary()

    model.fit([

                X_train_wraggled_g.iloc[0:int(len(X_train_wraggled_g)*SPLIT_RATIO)].values.astype('float32'),

                X_train_wraggled_c.iloc[0:int(len(X_train_wraggled_c)*SPLIT_RATIO)].values.astype('float32')

                ],

              Y_train_wraggled.iloc[0:int(len(X_train_wraggled)*SPLIT_RATIO)].values.astype('float32'),

              epochs=600,

        

              validation_data= validation,

              batch_size=200,

                shuffle = True,

              callbacks = [checkpoint,early_stop]

             )

    

# from keras.models import load_model

# os.listdir('../input/best-weights')

# try:

#     best_model = load_model('../input/best-weights/best_weights.hdf5', custom_objects={'weighted_loss': get_weighted_loss(class_weights), 'f1_m':f1_m})

# except OSError:

#     best_model = model
# Y_predicts = (best_model.predict([X_train_wraggled_g, X_train_wraggled_c]))
# Y_predicts_df = pd.DataFrame(Y_predicts)

# Y_predicts_df.columns = Y_train_wraggled.columns

# Y_predicts_df
# Y_predicts_df[Y_train_wraggled[]]
# Y_train_wraggled.index = Y_predicts_df.index
# Y_predicts_df[(Y_train_wraggled['5-alpha_reductase_inhibitor']==0) & (Y_predicts_df['5-alpha_reductase_inhibitor'] > 0.005)].iloc[0].idxmax()
# Y_train_wraggled.loc[1024].idxmax()
# min(Y_predicts_df[(Y_train_wraggled['5-alpha_reductase_inhibitor']==1)]['5-alpha_reductase_inhibitor'])
test_feature = pd.read_csv('../input/lish-moa/test_features.csv')

from keras.models import load_model

try:

    best_model = load_model('best_weights.hdf5', custom_objects={'weighted_loss': get_weighted_loss(class_weights), 'f1_m':f1_m})

except OSError:

    best_model = model
default_result = {}

for y_c in Y_train_wraggled.columns:

    default_result[y_c] = 0
# predicts = []



X_test = test_feature



X_test_wraggled = X_test[X_test.columns[4:]]



X_test_wraggled_g = X_test_wraggled[X_test_wraggled.columns[pd.Series(X_test_wraggled.columns).str.startswith('g')]] 



X_test_wraggled_c = X_test_wraggled[X_test_wraggled.columns[pd.Series(X_test_wraggled.columns).str.startswith('c')]] 



if model_name in ['conv']:

    result = model.predict(X_test_wraggled.values)

elif model_name in ['parallel_conv', 'LSTM']:

    result = model.predict([X_test_wraggled_g.values, X_test_wraggled_c.values])

elif model_name in ['g_conv']:

    result = model.predict([X_test_wraggled_g.values])

elif model_name in ['c_lstm']:

    result = model.predict([X_test_wraggled_c.values])
predicts_df = pd.DataFrame(result)
predicts_df[test_feature['cp_type'] == "ctl_vehicle"] = 0
predicts_df.columns = Y_train_wraggled.columns
predicts_df['sig_id']=test_feature['sig_id']

predicts_df = predicts_df[['sig_id']+list(Y_train_wraggled.columns)]

predicts_df
predicts = pd.DataFrame(predicts_df)

predicts.to_csv('submission.csv', index=False)

predicts