PATH = '../input/lish-moa/'
#!pip install tensorflow-addons
import gc

import random

import os

import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

#ptimizer=tfa.optimizers.AdamW(lr = 2e-3, weight_decay = 1e-5, clipvalue = 700)

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD, Adam, RMSprop

from keras.layers import Dropout

from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping,ReduceLROnPlateau

from keras.optimizers import SGD, Adam, RMSprop



import tensorflow_addons as tfa



from keras.utils import plot_model

from keras.models import Model

from keras.layers import Input

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers.merge import concatenate





from sklearn.metrics import log_loss
import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
def seed_everything(seed=999):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    

seed_everything(49)
train_features = pd.read_csv(PATH+'train_features.csv')

test_features =pd.read_csv(PATH+'test_features.csv')

train_targets_nonscored =pd.read_csv(PATH+'train_targets_nonscored.csv')

train_targets_scored =pd.read_csv(PATH+'train_targets_scored.csv')

sample_submission =pd.read_csv(PATH+'sample_submission.csv')
train_features.describe()
#Let us look at train_features table: first 5 rows and shape

print(train_features.shape)

train_features.head(5)
#Let us look at test_features table: first 5 rows and shape

print(test_features.shape)

test_features.head(5)
#test_features.loc[test_features['cp_type'] == 1].head()
#Let us look at train_targets_nonscored table: first 5 rows and shape

print(train_targets_nonscored.shape)

train_targets_nonscored.head(5)
#Let us look at train_targets_scored table: first 5 rows and shape

print(train_targets_scored.shape)

train_targets_scored.head(5)
#Let us look closely at train_targets_scored table

train_targets_scored.describe()

#Just want to be sure that scored and nonscored dont have overalpping features

print(np.intersect1d(train_targets_scored.columns, train_targets_nonscored.columns))

#Just want to be sure that train features  and nonscored dont have overalpping features

print(np.intersect1d(train_features.columns, train_targets_nonscored.columns))

#Let us look at sample_submission table: first 5 rows and shape

print(sample_submission.shape)

sample_submission.head(5)
#If this is equal to 207, we have same columns in train_targets_scored and sample_submission

print(len(np.intersect1d(train_targets_scored.columns, sample_submission.columns)))
#Let us check data types of train_features

train_features.select_dtypes(include=['object','category','int','float']).dtypes
#Which are the object datatypes

train_features.select_dtypes(include=['object']).dtypes
#This means we have have cp_type and cp_dose as categorical variables, also cp_time which has only three values 24, 48 and 72

tr_features= [col for col in train_features.columns if col!='sig_id']

print("Length of train features list without 'sig_id' is:", len(tr_features))
# Now use label encoder to convert train_features and test_features df together

for train_feat in ['cp_type', 'cp_dose','cp_time']:

    le = LabelEncoder()

    le.fit(list(train_features[train_feat].astype(str).values) + list(test_features[train_feat].astype(str).values))

    train_features[train_feat] = le.transform(list(train_features[train_feat].astype(str).values))

    test_features[train_feat] = le.transform(list(test_features[train_feat].astype(str).values))
#Let us check whether label encoding done

print(train_features.head(5))

#Let us check whether label encoding done

print(test_features.head(5))
#Let us keep on train_feat columns for NN

train = train_features[tr_features]

test= test_features[tr_features]

train_target = train_targets_scored.drop(['sig_id'], axis=1)

train_target_aux = train_targets_nonscored.drop(['sig_id'], axis=1)
#So label encoding is done, let us move towards building NN

#We are using StandardScalar to transform all features and feed a balanced input to the neural net



sc = StandardScaler()

#train = sc.fit_transform(train)

#test = sc.transform(test)
len_X =int(train.shape[1])

print(len_X)
target_feat_len= train_target.shape[1]

target_aux_len= train_target_aux.shape[1]

#print(target_feat_len,train_target_aux)
## You can play around with network architecture and check what works



#Using functional API, we will create a model with two outputs. First for scored columns and second for nonscored columns.

#Nonscored output is forcing model to consider this targets and I "hope" that it will produce more better accuracy

#Using batch normalization and high dropouts so that we prevent overfitting



#9/11 added tfa.layers.WeightNormalization



def getModel2(input_dim,target_feature_length,target_auxiliary_length):

    visible = Input(shape=(input_dim,))

    hidden1 = tfa.layers.WeightNormalization(Dense(875, activation='relu'))(visible)

    batchnorm1= BatchNormalization()(hidden1)

    dropout1= Dropout(0.5)(batchnorm1)

    hidden2 = tfa.layers.WeightNormalization(Dense(1750, activation='relu'))(dropout1)

    batchnorm2= BatchNormalization()(hidden2)

    dropout2= Dropout(0.5)(batchnorm2)

    hidden3 = tfa.layers.WeightNormalization(Dense(100, activation='relu'))(dropout2)

    batchnorm3= BatchNormalization(name="batchnorm3")(hidden3)

    dropout3= Dropout(0.5,name="dropout3")(batchnorm3)    

    output_2 = tfa.layers.WeightNormalization(Dense(target_auxiliary_length, activation='sigmoid',name="ouput_2"))(dropout3)

    

    hidden4 = tfa.layers.WeightNormalization(Dense(206, activation='relu',name="hidden4"))(output_2)

    batchnorm4= BatchNormalization(name="batchnorm4")(hidden4)

    dropout4= Dropout(0.5,name="dropout4")(batchnorm4)

    

    concat1 = tf.keras.layers.Concatenate(name="concate_nonscored_feat")([output_2, dropout4])

    

    output1 = tfa.layers.WeightNormalization(Dense(target_feature_length, activation='sigmoid', name="outputscore"))(concat1)

    output2 = tfa.layers.WeightNormalization(Dense(target_auxiliary_length, activation='sigmoid',name="outputnonscore"))(concat1)

    model = Model(inputs=visible, outputs=[output1,output2])

    return model
#Create a sample model and check summay

modelNN= getModel2(len_X,target_feat_len,target_aux_len)

modelNN.summary()



def metric(y_true, y_pred):

    #print('y_true:', y_true.head(3))

    #print('y_pred:',y_pred.head(3))

    metrics = []

    for _target in train_target.columns:

        metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float), labels=[0,1]))

    return np.mean(metrics)
#Plot model and delete from memory

plot_model(modelNN, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

del modelNN
"""

from sklearn.model_selection import KFold

NFOLD = 10

kf = KFold(n_splits=NFOLD)



BATCH_SIZE=32

EPOCHS=50



pout = np.zeros((test_features.shape[0], target_feat_len))

paux = np.zeros((test_features.shape[0], target_aux_len))

#Train is already a numpy as it went through StandardScalar

train_features = train

#train_targets needs to be converted to numpy

train_targets = train_target.values

train_target_aux_np=train_target_aux.values



pred = np.zeros((train_features.shape[0], target_feat_len))



cnt=0

for tr_idx, val_idx in kf.split(train_features):

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=0.0001, mode='auto')

    cnt += 1

    print(f"FOLD {cnt}")

    #net = getModel(len_X,target_feat_len)

    net= getModel2(len_X,target_feat_len,target_aux_len)

    #net.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    net.compile(optimizer = tfa.optimizers.AdamW(lr = 2e-3, weight_decay = 1e-5, clipvalue = 700), loss = 'binary_crossentropy', metrics = ['accuracy'])

    net.fit(train_features[tr_idx], [train_targets[tr_idx],train_target_aux_np[tr_idx]], batch_size=BATCH_SIZE, epochs=EPOCHS, 

            validation_data=(train_features[val_idx], [train_targets[val_idx],train_target_aux_np[val_idx]]), verbose=0, callbacks=[early_stopping,reduce_lr_loss])

    print("Training for this fold", net.evaluate(train_features[tr_idx], [train_targets[tr_idx],train_target_aux_np[tr_idx]], verbose=0, batch_size=BATCH_SIZE))

    print("Validation for this fold", net.evaluate(train_features[val_idx], [train_targets[val_idx],train_target_aux_np[val_idx]], verbose=0, batch_size=BATCH_SIZE))

    print("Predict on validation data for this fold")

    pred[val_idx] = net.predict(train_features[val_idx], batch_size=BATCH_SIZE, verbose=0)[0]

    print(f'OOF Metric log_loss for this FOLD {cnt} : {metric(pd.DataFrame(train_targets[val_idx]), pd.DataFrame(pred[val_idx], columns=train_target.columns))}')

    #print(f'OOF Metric log_loss for this FOLD {cnt} : {metric(train_targets[val_idx], pred[val_idx])}')

    print("Predict test for nonscored targets")

    paux += net.predict(test, batch_size=BATCH_SIZE, verbose=0)[1] / NFOLD

    print("Predict test with scored targets")

    pout += net.predict(test, batch_size=BATCH_SIZE, verbose=0)[0] / NFOLD

    

"""
from sklearn.model_selection import KFold

#from sklearn.model_selection import MultilabelStratifiedKFold

NFOLD = 10



BATCH_SIZE=16

EPOCHS=100



pout = np.zeros((test_features.shape[0], target_feat_len))

paux = np.zeros((test_features.shape[0], target_aux_len))

foldmetric=0

#Train is already a numpy as it went through StandardScalar

train_features = train

#train_targets needs to be converted to numpy

train_targets = train_target

train_target_aux_np=train_target_aux



pred = np.zeros((train_features.shape[0], target_feat_len))



cnt=0

kf = KFold(n_splits=NFOLD)

#kf = MultilabelStratifiedKFold(n_splits=NFOLD,shuffle=True, random_state=49)

for tr_idx, val_idx in kf.split(train_features):

#for tr_idx, val_idx in enumerate(kf.split(train_targets,train_targets)):

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=0.0001, mode='auto')

    cnt += 1

    print(f"FOLD {cnt}")

    #net = getModel(len_X,target_feat_len)

    net= getModel2(len_X,target_feat_len,target_aux_len)

    #net.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    net.compile(optimizer = tfa.optimizers.AdamW(lr = 2e-3, weight_decay = 1e-5, clipvalue = 700), loss = 'binary_crossentropy', metrics = ['accuracy'])

    net.fit(train_features.iloc[tr_idx], [train_targets.iloc[tr_idx],train_target_aux_np.iloc[tr_idx]], batch_size=BATCH_SIZE, epochs=EPOCHS, 

            validation_data=(train_features.iloc[val_idx], [train_targets.iloc[val_idx],train_target_aux_np.iloc[val_idx]]), verbose=0, callbacks=[early_stopping,reduce_lr_loss])

    print("Training for this fold", net.evaluate(train_features.iloc[tr_idx], [train_targets.iloc[tr_idx],train_target_aux_np.iloc[tr_idx]], verbose=0, batch_size=BATCH_SIZE))

    print("Validation for this fold", net.evaluate(train_features.iloc[val_idx], [train_targets.iloc[val_idx],train_target_aux_np.iloc[val_idx]], verbose=0, batch_size=BATCH_SIZE))

    print("Predict on validation data for this fold")

    pred[val_idx] = net.predict(train_features.iloc[val_idx], batch_size=BATCH_SIZE, verbose=0)[0]

    fold_metric = metric(train_targets.iloc[val_idx], pd.DataFrame(pred[val_idx],columns=train_target.columns))

    print(f'OOF Metric log_loss for this FOLD {cnt} : {fold_metric}')

    foldmetric += fold_metric/ NFOLD

    print("Average metric is:",foldmetric)

    #print(f'OOF Metric log_loss for this FOLD {cnt} : {metric(train_targets[val_idx], pred[val_idx])}')

    print("Predict test for nonscored targets")

    paux += net.predict(test, batch_size=BATCH_SIZE, verbose=0)[1] / NFOLD

    print("Predict test with scored targets")

    pout += net.predict(test, batch_size=BATCH_SIZE, verbose=0)[0] / NFOLD
foldmetric.head()
submission = pd.DataFrame(data=pout, columns=train_target.columns)

subpaux = pd.DataFrame(data=paux, columns=train_target_aux.columns)
submission.insert(0, column = 'sig_id', value=sample_submission['sig_id'])

subpaux.insert(0, column = 'sig_id', value=sample_submission['sig_id'])
submission.loc[test_features['cp_type'] == 1, train_target.columns] = 0
submission.to_csv('submission.csv', index=False)

subpaux.to_csv('subaux.csv', index=False)
print(submission.shape)

print(subpaux.shape)