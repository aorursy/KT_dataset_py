import subprocess

import sys

import datetime

import os

import random as rn

import numpy as np

import pandas as pd



from matplotlib import pyplot

from sklearn.model_selection import train_test_split



import keras

from keras.utils import to_categorical

from keras import models

from keras import layers

from keras.models import Model, load_model

from keras import regularizers, initializers

# from keras import regularizers, initializers

# from keras import regularizers, initializers

from keras.layers import Dense, Dropout

# from keras.models import load_model

from keras.layers.normalization import BatchNormalization

from keras.callbacks import ModelCheckpoint



from keras.callbacks import TensorBoard

import keras.backend as K



from keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization



import shutil



batch_size=128



keras.__version__

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

pred_df = pd.read_csv('../input/sample_submission.csv')

# log_dir = '/'
if os.path.exists("./test1") == False:

    shutil.os.mkdir('../work')

print(os.listdir("../"))

print(os.listdir("../work/"))
log_dir = '../work'
print( pred_df.shape )

print( pred_df.columns )

pred_df = pred_df.drop(columns='Label')

print( pred_df.shape )

print( pred_df.columns )
print( type( train_df ) )

print( type( test_df ) )

print( train_df.shape )

print( test_df.shape )
train_label_df = train_df.iloc[:,0:1]

print( train_df.shape )

train_label_df
print( train_df.shape )

train_df = train_df.drop(columns="label")

print( train_df.shape )
train_df
train_df_n = train_df.astype('float32') / 255

test_df_n = test_df.astype('float32') / 255



train_np_n = train_df_n.values

test_np_n  = test_df_n.values



print( train_np_n.shape )

print( test_np_n.shape )
print( type( train_label_df ) )

print( train_label_df.shape )

train_label_np = train_label_df.values

print( type( train_label_np ) )

print( train_label_np.shape )
X_train, X_val, y_train, y_val = train_test_split(train_np_n, train_label_np, test_size=0.25, random_state=42)

print( X_train.shape )

print( y_train.shape )

print( X_val.shape )

print( y_val.shape )
X_test = test_np_n

print( X_test.shape )
print( y_train.shape )

print( y_val.shape )

Y_train = to_categorical( y_train )

Y_val = to_categorical( y_val )

print( Y_train.shape )

print( Y_val.shape )
print( X_train.shape )

print( X_val.shape )

print( X_test.shape )

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))

X_val = X_val.reshape((X_val.shape[0], 28, 28, 1))

X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

print( X_train.shape )

print( X_val.shape )

print( X_test.shape )
def check_result(history):



  import statistics

  for _k in history.keys():



      if _k == 'val_acc':

          # print(f'{_k.ljust(8)} {"max :"} {max(history[_k]):.10f}    {"avg :"} {statistics.mean(history[_k]):.10f}   {"last :"} {history[_k][-1]:.10f}')

          histmax = max(history[_k])

          histlast = history[_k][-1]

          histgap = (histmax-histlast) / histmax

          print(f'{_k.ljust(8)} {"max :"} {histmax:.10f}    {"avg :"} {statistics.mean(history[_k]):.10f}   {"last :"} {histlast:.10f}  {"gap :"} {histgap:.2%}')

        
# # batch_size=128

# tensorboard = TensorBoard(

#                             log_dir=log_dir,

#                             histogram_freq=1,

#                             write_graph=True,

#                             write_grads=True,

#                             batch_size=batch_size,

#                             write_images=True

#                           )

# K.clear_session()
from keras.callbacks import LearningRateScheduler



def step_decay_for_conv2(epoch):

    x = 0.0005

    if epoch >= 20: x = 0.0001

    if epoch >= 40: x = 0.00005

    

    return x



lr_decay = LearningRateScheduler(step_decay_for_conv2,verbose=0)
# from keras.callbacks import  EarlyStopping



# # es_cb = EarlyStopping( monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto')   

# # es_cb = EarlyStopping( monitor='val_loss', min_delta=0, patience=3, verbose=2, mode='auto')   

# es_cb = EarlyStopping( monitor='val_last_fc_acc', min_delta=0, patience=3, verbose=2, mode='auto')   
def create_model2():

    inputs_mnist = Input( shape=(28,28,1) )

    inputs = Conv2D(filters=64, kernel_size=(3,3), padding='same', bias_regularizer=regularizers.l2(0.005) )(inputs_mnist)

    inputs = Conv2D(filters=128, kernel_size=(3,3), padding='same', bias_regularizer=regularizers.l2(0.005) )(inputs)

    inputs = BatchNormalization()( inputs )

    inputs = MaxPooling2D(pool_size=(2,2))(inputs)

    inputs = Conv2D(filters=128, kernel_size=(3,3), padding='same', bias_regularizer=regularizers.l2(0.005) )(inputs)

    inputs = Conv2D(filters=128, kernel_size=(3,3), padding='same', bias_regularizer=regularizers.l2(0.005) )(inputs)

    inputs = BatchNormalization()( inputs )

    inputs = MaxPooling2D(pool_size=(2,2))(inputs)

    inputs = Conv2D(filters=128, kernel_size=(3,3), padding='same', bias_regularizer=regularizers.l2(0.005) )(inputs)

    inputs = Conv2D(filters=128, kernel_size=(3,3), padding='same', bias_regularizer=regularizers.l2(0.005) )(inputs)

    inputs = BatchNormalization()( inputs )

    inputs = MaxPooling2D(pool_size=(2,2))(inputs)

    inputs = Conv2D(filters=128, kernel_size=(3,3), padding='same', bias_regularizer=regularizers.l2(0.005) )(inputs)

    inputs = Conv2D(filters=128, kernel_size=(3,3), padding='same', bias_regularizer=regularizers.l2(0.005) )(inputs)

    inputs = BatchNormalization()( inputs )

    inputs_last = MaxPooling2D(pool_size=(2,2))(inputs)

   



    inputs2 = Conv2D(filters=64, kernel_size=(5,5), padding='same', bias_regularizer=regularizers.l2(0.005) )(inputs_mnist)

    inputs2 = Conv2D(filters=128, kernel_size=(5,5), padding='same', bias_regularizer=regularizers.l2(0.005) )(inputs2)

    inputs2 = BatchNormalization()( inputs2 )

    inputs2 = MaxPooling2D(pool_size=(2,2))(inputs2)

    inputs2 = Conv2D(filters=128, kernel_size=(5,5), padding='same', bias_regularizer=regularizers.l2(0.005) )(inputs2)

    inputs2 = Conv2D(filters=128, kernel_size=(5,5), padding='same', bias_regularizer=regularizers.l2(0.005) )(inputs2)

    inputs2 = BatchNormalization()( inputs2 )

    inputs2 = MaxPooling2D(pool_size=(2,2))(inputs2)

    inputs2 = Conv2D(filters=128, kernel_size=(5,5), padding='same', bias_regularizer=regularizers.l2(0.005) )(inputs2)

    inputs2 = Conv2D(filters=128, kernel_size=(5,5), padding='same', bias_regularizer=regularizers.l2(0.005) )(inputs2)

    inputs2 = BatchNormalization()( inputs2 )

    inputs2 = MaxPooling2D(pool_size=(2,2))(inputs2)

    inputs2 = Conv2D(filters=128, kernel_size=(5,5), padding='same', bias_regularizer=regularizers.l2(0.005) )(inputs2)

    inputs2 = Conv2D(filters=128, kernel_size=(5,5), padding='same', bias_regularizer=regularizers.l2(0.005) )(inputs2)

    inputs2 = BatchNormalization()( inputs2 )

    inputs2_last = MaxPooling2D(pool_size=(2,2))(inputs2)





    inputs_ad_ls = Flatten()(inputs_last)

    inputs_ad_ls = Dense(units=4096, activation='relu')(inputs_ad_ls)

    inputs_ad_ls = Dropout(rate=0.5)(inputs_ad_ls)

    inputs_ad_ls = Dense(units=4096, activation='relu')(inputs_ad_ls)

    inputs_ad_ls = Dropout(rate=0.5)(inputs_ad_ls)

    outputs_ad_ls = Dense(units=10, activation='softmax', name='1st_fc')(inputs_ad_ls)  

    

    inputs2_ad_ls = Flatten()(inputs2_last)

    inputs2_ad_ls = Dense(units=4096, activation='relu')(inputs2_ad_ls)

    inputs2_ad_ls = Dropout(rate=0.5)(inputs2_ad_ls)

    inputs2_ad_ls = Dense(units=4096, activation='relu')(inputs2_ad_ls)

    inputs2_ad_ls = Dropout(rate=0.5)(inputs2_ad_ls)

    outputs2_ad_ls = Dense(units=10, activation='softmax', name='2nd_fc')(inputs2_ad_ls)   

    

    inputs3 = keras.layers.concatenate([inputs_last, inputs2_last]) 



    inputs_2_fc = Flatten()(inputs3)

    

    inputs_2_fc = Dense(units=8192, activation='relu')(inputs_2_fc)

    inputs_2_fc = Dropout(rate=0.5)(inputs_2_fc)

    inputs_2_fc = Dense(units=4096, activation='relu')(inputs_2_fc)

    inputs_2_fc = Dropout(rate=0.5)(inputs_2_fc)

    inputs_2_fc = Dense(units=4096, activation='relu')(inputs_2_fc)

    inputs_2_fc = Dropout(rate=0.5)(inputs_2_fc)

    outputs = Dense(units=10, activation='softmax', name='last_fc')(inputs_2_fc)



    model = Model(inputs=[inputs_mnist], outputs=[outputs, outputs_ad_ls, outputs2_ad_ls])

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], loss_weights=[1.0, 0.2, 0.2] )



    model.summary()



    return model

_model = create_model2()
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



SVG(model_to_dot(_model).create(prog='dot', format='svg'))
filepath="../work/weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"

# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)

checkpoint = ModelCheckpoint(filepath, monitor='val_last_fc_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
def fit_the_model(_model, _epochs):

#     original_hist = _model.fit(X_train, Y_train, epochs=_epochs, batch_size=batch_size, 

    original_hist = _model.fit(np.array(X_train), [np.array(Y_train),np.array(Y_train),np.array(Y_train)], epochs=_epochs, batch_size=batch_size, 

                verbose=1,

                callbacks=[lr_decay],

#                 validation_data=(X_val, Y_val))

                validation_data=(np.array(X_val), [np.array(Y_val),np.array(Y_val),np.array(Y_val)] ))

    return original_hist
def fit_the_model_with_data(_model, _epochs, X_train, Y_train, X_val, Y_val, _cp):

#     original_hist = _model.fit(X_train, Y_train, epochs=_epochs, batch_size=batch_size, 

    original_hist = _model.fit(np.array(X_train), [np.array(Y_train),np.array(Y_train),np.array(Y_train)], epochs=_epochs, batch_size=batch_size,                                

#                 verbose=1,

                verbose=0,

                callbacks=[lr_decay, _cp],

#                 validation_data=(X_val, Y_val))

                validation_data=(np.array(X_val), [np.array(Y_val),np.array(Y_val),np.array(Y_val)] ))

    return original_hist
# epochs=100; _m = create_model2( 'same', 'glorot_uniform', None, 'zeros', None, 4 );  history = fit_the_model(_m, epochs); check_result(history.history); #_m.save(mygdrive+"models/"+"normal"+'.h5')

epochs=100; _m = create_model2();  history = fit_the_model(_m, epochs); check_result(history.history); #_m.save(mygdrive+"models/"+"normal"+'.h5')
from sklearn.model_selection import KFold

import numpy as np
X = np.copy( train_np_n )

y = np.copy( train_label_np )



kf = KFold(n_splits=4,shuffle=True)

kf.get_n_splits(X)



print(kf)  
# _m = create_model2( 'same', 'glorot_uniform', None, 'zeros', regularizers.l2(0.005), 4 );

_m = create_model2( );
_i = 0

for train_index, val_index in kf.split(X):

  filepath="../work/kfold_cp"+str(_i)+".hdf5"

#   _cp = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)

  _cp = ModelCheckpoint(filepath, monitor='val_last_fc_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)

  

#   _m = create_model2( 'same', 'glorot_uniform', None, 'zeros', regularizers.l2(0.005), 4 );

  _m = create_model2( );

#   print("TRAIN:", train_index, "TEST:", val_index)

  X_t, X_v = X[train_index], X[val_index]

  y_t, y_v = y[train_index], y[val_index]

  

  X_t = X_t.reshape(X_t.shape[0],28,28,1)

  X_v = X_v.reshape(X_v.shape[0],28,28,1)

  Y_t = to_categorical( y_t )

  Y_v = to_categorical( y_v )

  print( X_t.shape, y_t.shape, Y_t.shape, X_v.shape, y_v.shape, Y_v.shape )

  

  epochs=100

#   history = fit_the_model_with_data(_m, epochs, X_t, Y_t, X_v, Y_v) 

  history = fit_the_model_with_data(_m, epochs, X_t, Y_t, X_v, Y_v, _cp)

  check_result(history.history); 

#   _m.save(mygdrive+"models/work/"+"kfold"+str(_i)+'.h5')

  _m.save("../work/"+"kfold"+str(_i)+'.h5')

  _i += 1
print(os.listdir("../work"))
# model0 = load_model( "../work/kfold0.h5" )

# model1 = load_model( "../work/kfold1.h5" )

# model2 = load_model( "../work/kfold2.h5" )

# model3 = load_model( "../work/kfold3.h5" )



model_cp0 = load_model( "../work/kfold_cp0.hdf5" )

model_cp1 = load_model( "../work/kfold_cp1.hdf5" )

model_cp2 = load_model( "../work/kfold_cp2.hdf5" )

model_cp3 = load_model( "../work/kfold_cp3.hdf5" )
print(os.listdir("../work"))
def pred_argmax( _model, _data, _label=np.array([0]) ):

#   c_prob = _model.predict( _data )

  c_prob, c_prob1, c_prob2 = _model.predict( _data )

  pred= np.argmax(c_prob, axis=1)

  if len(_label) != 1:

    correct = np.argmax(_label,axis=1)

  else:

    correct = 0

  return c_prob, pred, correct



def pred_argmax2( _model,_gen, _data, _label=np.array([0]) ):

  _batch_size=128

  _s = len(_data) / _batch_size

  c_prob = _model.predict_generator( _gen.flow(_data,batch_size=_batch_size,shuffle=None), steps=_s)

  

  pred= np.argmax(c_prob, axis=1)

  if len(_label) != 1:

    correct = np.argmax(_label,axis=1)

  else:

    correct = 0

  return c_prob, pred, correct



def disp_image(_data, _pred, _correct, _list, _isdisplay="off"):

  pair_list = []

  wrong_list = []

  

  for _i in _list:

    if _isdisplay == "on":

      plt.imshow(_data[_i].reshape(28, 28), cmap=plt.get_cmap('gray'))

      plt.show()



    pair_list.append( str(_correct[_i])+" as "+str(_pred[_i]) )

    wrong_list.append( _pred[_i] )

    

  return pair_list, wrong_list
print( train_np_n.shape )

print( train_label_np.shape )
X_all = train_np_n.reshape( train_np_n.shape[0], 28, 28, 1 )

Y_all = to_categorical( train_label_np )

# Y_all = train_label_np 

print( X_all.shape )

print( Y_all.shape )

print( X_test.shape )
# X0_all_prob, X0_all_pred, X0_all_correct = pred_argmax(model0, X_all,Y_all)

# X1_all_prob, X1_all_pred, X1_all_correct = pred_argmax(model1, X_all,Y_all)

# X2_all_prob, X2_all_pred, X2_all_correct = pred_argmax(model2, X_all,Y_all)

# X3_all_prob, X3_all_pred, X3_all_correct = pred_argmax(model3, X_all,Y_all)

# _d0 = np.not_equal(X0_all_pred, X0_all_correct)

# _d1 = np.not_equal(X1_all_pred, X1_all_correct)

# _d2 = np.not_equal(X2_all_pred, X2_all_correct)

# _d3 = np.not_equal(X3_all_pred, X3_all_correct)

# print(sum(_d0))

# print(sum(_d1))

# print(sum(_d2))

# print(sum(_d3))

# DF = np.argmin([sum(_d0),sum(_d1),sum(_d2),sum(_d3)], axis=0)

# print( DF )

X0_cp_all_prob, X0_cp_all_pred, X0_cp_all_correct = pred_argmax(model_cp0, X_all,Y_all)

X1_cp_all_prob, X1_cp_all_pred, X1_cp_all_correct = pred_argmax(model_cp1, X_all,Y_all)

X2_cp_all_prob, X2_cp_all_pred, X2_cp_all_correct = pred_argmax(model_cp2, X_all,Y_all)

X3_cp_all_prob, X3_cp_all_pred, X3_cp_all_correct = pred_argmax(model_cp3, X_all,Y_all)

_d0_cp = np.not_equal(X0_cp_all_pred, X0_cp_all_correct)

_d1_cp = np.not_equal(X1_cp_all_pred, X1_cp_all_correct)

_d2_cp = np.not_equal(X2_cp_all_pred, X2_cp_all_correct)

_d3_cp = np.not_equal(X3_cp_all_pred, X3_cp_all_correct)

print(sum(_d0_cp))

print(sum(_d1_cp))

print(sum(_d2_cp))

print(sum(_d3_cp))

DF_cp = np.argmin([sum(_d0_cp),sum(_d1_cp),sum(_d2_cp),sum(_d3_cp)], axis=0)

print( DF_cp )
X0_test_prob, X0_test_pred, X0_test_correct = pred_argmax(model_cp0, X_test)

X1_test_prob, X1_test_pred, X1_test_correct = pred_argmax(model_cp1, X_test)

X2_test_prob, X2_test_pred, X2_test_correct = pred_argmax(model_cp2, X_test)

X3_test_prob, X3_test_pred, X3_test_correct = pred_argmax(model_cp3, X_test)

print( X0_test_pred )

print( X1_test_pred )

print( X2_test_pred )

print( X3_test_pred )
print( train_label_np.shape )

# print( X0_all_pred.shape )

# pred_info = np.concatenate([train_label_np, X0_all_pred.reshape(X0_all_pred.shape[0],1), X1_all_pred.reshape(X1_all_pred.shape[0],1), X2_all_pred.reshape(X2_all_pred.shape[0],1), X3_all_pred.reshape(X3_all_pred.shape[0],1)], axis=1 )

pred_info = np.concatenate([X0_cp_all_pred.reshape(X0_cp_all_pred.shape[0],1), X1_cp_all_pred.reshape(X1_cp_all_pred.shape[0],1), X2_cp_all_pred.reshape(X2_cp_all_pred.shape[0],1), X3_cp_all_pred.reshape(X3_cp_all_pred.shape[0],1)], axis=1 )

print( pred_info.shape )

print( pred_info )
pred_info_test = np.concatenate([X0_test_pred.reshape( X0_test_pred.shape[0],1),  X1_test_pred.reshape( X1_test_pred.shape[0],1),  X2_test_pred.reshape( X2_test_pred.shape[0],1),  X3_test_pred.reshape( X3_test_pred.shape[0],1)], axis=1 )

print( pred_info_test.shape )

print( pred_info_test )
DF_cp
from collections import Counter

# DF = 2

def return_result(preds):

  _c = Counter(preds)

  _v = _c.most_common(1)[0][0]

  _n = _c.most_common(1)[0][1]

  if _n == 4 or _n ==3:

    return _v

  else:

    return preds[DF_cp]

##########################################################

# rdf = pd.DataFrame(pred_info).apply(return_result, axis=1)

# rdf
rdf_test = pd.DataFrame(pred_info_test).apply(return_result, axis=1)

rdf_test
pred_df["label"]=rdf_test

pred_df
pred_df.to_csv('submission.csv', index=False)