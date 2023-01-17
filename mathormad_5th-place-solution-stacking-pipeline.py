import os

import pandas as pd

import numpy as np

from tqdm import *

from sklearn.metrics import log_loss

from sklearn.model_selection import GroupKFold

import math



import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.models import Sequential, load_model, Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,

                          BatchNormalization, Input, Conv2D, Multiply, Lambda,

                          Concatenate, GlobalAveragePooling2D, Softmax)

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



from lightgbm import LGBMClassifier

import gc





NUM_CLASSES = 6

NUM_MODELS = 9
train_df = pd.read_csv('../input/rsna-oof-data-for-stacking/oof_{}models_post_with_meta.csv'.format(NUM_MODELS))

test_df = pd.read_csv('../input/rsna-oof-data-for-stacking/sub_{}models_post_with_meta.csv'.format(NUM_MODELS))

train_meta = pd.read_csv('../input/rsna-oof-data-for-stacking/train_meta_with_label_stage2.csv')



train_df = pd.merge(train_df, train_meta, on="sop_instance_uid")

train_df.rename(columns={"patient_id_x": "patient_id"}, inplace=True)

train_df.drop(['patient_id_y'], axis=1, inplace=True)



print(train_df.shape, test_df.shape)
X_train_lgbm = train_df.iloc[:, 1:(6*NUM_MODELS+1)].values

X_train = X_train_lgbm.reshape((len(train_df), NUM_MODELS, NUM_CLASSES, 1))

Y_train = train_df.iloc[:, -6:].values.astype(float)

X_test_lgbm = test_df.iloc[:, 1:(6*NUM_MODELS+1)].values

X_test = X_test_lgbm.reshape((len(test_df), NUM_MODELS, NUM_CLASSES, 1))

Y_pred = np.zeros((X_test.shape[0], NUM_CLASSES)).astype(float)

print(X_train.shape, Y_train.shape, X_test.shape)

print(X_train_lgbm.shape, X_test_lgbm.shape, Y_pred.shape)
# create new features for lightgbm

base_train_pred = np.mean(X_train, axis=1).reshape(len(X_train), NUM_CLASSES)

base_test_pred = np.mean(X_test, axis=1).reshape(len(X_test), NUM_CLASSES)



X_train_lgbm = np.concatenate((X_train_lgbm, base_train_pred), axis=1)

X_test_lgbm = np.concatenate((X_test_lgbm, base_test_pred), axis=1)

print(X_train_lgbm.shape, X_test_lgbm.shape)
# parameters for LGBM model

params = {

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'log_loss',

    'n_estimators': 2000,

    'learning_rate': 0.01,

    'num_leaves': 31,

    'max_depth': -1, 

    'n_jobs': -1,

    'subsample': 0.5, 

    'subsample_freq': 2,

    'colsample_bytree': 0.9,

}
def create_stacking_model():

    

    input_tensor = Input(shape=(NUM_MODELS, NUM_CLASSES, 1))

    x = Conv2D(128, kernel_size=(NUM_MODELS, 1), activation='relu')(input_tensor)

    x = Dropout(0.3)(x)

    x = Conv2D(256, (1,NUM_CLASSES), activation='relu')(x)

    x = Dropout(0.3)(x)

    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)

    x = Dropout(0.5)(x)

    output1 = Dense(NUM_CLASSES, activation='sigmoid',

               name='output1')(x)

    x = Dense(256, activation='relu')(x)

    x = Dropout(0.5)(x)

    output2 = Dense(NUM_CLASSES, activation='sigmoid',

                   name='output2')(x)

    model = Model(input_tensor, [output1, output2])

    

    return model
# weighted log loss function for keras

def _weighted_log_loss(y_true, y_pred):

    

    class_weights = np.array([2, 1, 1, 1, 1, 1])



    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1.0-tf.keras.backend.epsilon())

    out = -(         y_true  * tf.keras.backend.log(      y_pred) * class_weights

            + (1.0 - y_true) * tf.keras.backend.log(1.0 - y_pred) * class_weights)

    

    return tf.keras.backend.mean(out, axis=-1)



# weighted log loss function for evaluating

def multilabel_logloss(y_true, y_pred):

    class_weights = np.array([2, 1, 1, 1, 1, 1])

    eps = 1e-15

    y_pred = np.clip(y_pred, eps, 1.0-eps)

    out = -(         y_true  * np.log(      y_pred) * class_weights

            + (1.0 - y_true) * np.log(1.0 - y_pred) * class_weights)

    

    return np.mean(out)
# mix-up generator for NN



def mixup_data(x, y, alpha=0.4):

    

    # 50% chance to keep original data

    if(np.random.randint(2) == 1):

        return x, y

    

    # 50% chance to apply mix-up augmentation

    lam = np.random.beta(alpha, alpha)

    sample_size = x.shape[0]

    index_array = np.arange(sample_size)

    np.random.shuffle(index_array)

    

    mixed_x = lam * x + (1 - lam) * x[index_array]

    mixed_y = (lam * y) + ((1 - lam) * y[index_array])

    

    return mixed_x, mixed_y



def make_batches(size, batch_size):

    nb_batch = int(np.ceil(size/float(batch_size)))

    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]



def batch_generator(X,y,batch_size=128,shuffle=True,mixup=False):

    sample_size = X.shape[0]

    index_array = np.arange(sample_size)

    

    while True:

        if shuffle:

            np.random.shuffle(index_array)

        batches = make_batches(sample_size, batch_size)

        for batch_index, (batch_start, batch_end) in enumerate(batches):

            batch_ids = index_array[batch_start:batch_end]

            X_batch = X[batch_ids]

            y_batch = y[batch_ids]

            

            if mixup:

                X_batch, y_batch = mixup_data(X_batch, y_batch)

            

            yield X_batch, {'output1': y_batch, 'output2': y_batch} 
BATCH_SIZE = 512 * 4

REPEAT = 1 # you can repeat many times to improve the stability

NN_score = []

base_score = []

LGBM_score = []

STACK_score = []

EPOCH = 50

NUM_FOLDS = 5



for num_repeat in range(REPEAT):

    GKF = GroupKFold(n_splits=NUM_FOLDS)

    for fold, (train_index, test_index) in enumerate(GKF.split(X_train, Y_train, train_df['patient_id'])):



        print('***************  Fold %d  ***************'%(fold))



        # dataset for NN

        x_train_nn, x_valid_nn = X_train[train_index], X_train[test_index]

        y_train_fold, y_valid_fold = Y_train[train_index], Y_train[test_index]

        print(x_train_nn.shape, x_valid_nn.shape)

        print(y_train_fold.shape, y_valid_fold.shape)

        

        # dataset for lgbm

        x_train_lgbm, x_valid_lgbm = X_train_lgbm[train_index], X_train_lgbm[test_index]

        print(x_train_lgbm.shape, x_valid_lgbm.shape)

        

        ####### average ################

        base_fold_pred = np.mean(x_valid_nn, axis=1).reshape(len(x_valid_nn), NUM_CLASSES)

        base_score.append(multilabel_logloss(y_valid_fold, base_fold_pred))

        print('simple average score for this fold: ', multilabel_logloss(y_valid_fold, base_fold_pred))



        ######## train NN ##################

        early_stoping = EarlyStopping(monitor='val_loss', patience=7, verbose=0)

        WEIGHTS_PATH = 'cnn_stacking_weights_repeat{}_fold{}.hdf5'.format(num_repeat, fold)

        save_checkpoint = ModelCheckpoint(WEIGHTS_PATH, monitor = 'val_loss', verbose = 0,

                                          save_best_only = True, save_weights_only = True, mode='min')

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr = 1e-8, verbose=0)

        callbacks = [save_checkpoint, early_stoping, reduce_lr]

        

        tr_gen = batch_generator(x_train_nn,y_train_fold,

                                 batch_size=BATCH_SIZE,

                                 shuffle=True, mixup=True)

        val_gen = batch_generator(x_valid_nn,y_valid_fold,

                                 batch_size=BATCH_SIZE,

                                 shuffle=False)

        

        train_gen_dataset = tf.data.Dataset.from_generator(

            lambda:tr_gen,

            output_types=('float32', {'output1': 'float32', 'output2': 'float32'}),

            output_shapes=(tf.TensorShape((None, NUM_MODELS, NUM_CLASSES, 1)),

                           {'output1':tf.TensorShape((None, NUM_CLASSES)),

                            'output2':tf.TensorShape((None, NUM_CLASSES))}))

        

        val_gen_dataset = tf.data.Dataset.from_generator(

            lambda:val_gen,

            output_types=('float32', {'output1': 'float32', 'output2': 'float32'}),

            output_shapes=(tf.TensorShape((None, NUM_MODELS, NUM_CLASSES, 1)),

                           {'output1':tf.TensorShape((None, NUM_CLASSES)),

                            'output2':tf.TensorShape((None, NUM_CLASSES))}))

        

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():

            model = create_stacking_model()

            model.compile(loss=_weighted_log_loss, optimizer=Adam(lr=1e-3))

            model.fit(train_gen_dataset,

                    steps_per_epoch=math.ceil(float(len(y_train_fold)) / float(BATCH_SIZE)),

                    validation_data=val_gen_dataset,

                    validation_steps=math.ceil(float(len(y_valid_fold)) / float(BATCH_SIZE)),

                    epochs=EPOCH, callbacks=callbacks,

                    workers=2, max_queue_size=10,

                    use_multiprocessing=True,

                    verbose=0)

        

        model.load_weights(WEIGHTS_PATH)

        valid_nn = model.predict(x_valid_nn, batch_size=BATCH_SIZE, verbose=0)

        valid_nn = np.sum(valid_nn, axis=0)/2

        nn_score = multilabel_logloss(y_valid_fold, valid_nn)

        NN_score.append(nn_score)

        print('NN score for this fold: ', nn_score)

        tmp = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)

        Y_pred += np.sum(tmp, axis=0)/(2*NUM_FOLDS)



        ###### train lgbm #############################

        valid_lgbm = np.zeros((y_valid_fold.shape))

        for i in range(NUM_CLASSES):

            lgbm_model = LGBMClassifier(**params)

            lgbm_model.fit(x_train_lgbm, y_train_fold[:,i],

                           eval_set=(x_valid_lgbm, y_valid_fold[:,i]),

                           eval_metric='logloss',

                           early_stopping_rounds=100,

                           verbose=0)

            valid_lgbm[:, i] += (lgbm_model.predict_proba(x_valid_lgbm,

                                                num_iteration=lgbm_model.best_iteration_)[:,1])

            Y_pred[:, i] += (lgbm_model.predict_proba(X_test_lgbm,

                                   num_iteration=lgbm_model.best_iteration_)[:,1])/NUM_FOLDS

        lgbm_score = multilabel_logloss(y_valid_fold, valid_lgbm)

        LGBM_score.append(lgbm_score)

        print('LGBM score for this fold: ', lgbm_score)

        

        stack_score = multilabel_logloss(y_valid_fold, (valid_lgbm+valid_nn)/2)

        print('LGBM + NN score for this fold: ', stack_score)

        STACK_score.append(stack_score)



        del (x_train_nn, x_valid_nn, y_train_fold, y_valid_fold,

             x_train_lgbm, valid_nn, valid_lgbm, tmp)

        gc.collect()

    

Y_pred = Y_pred/(2*REPEAT)
print('mean simple average score: ', np.mean(base_score))

print('mean NN score: ', np.mean(NN_score))

print('mean LGBM score: ', np.mean(LGBM_score))

print('mean NN + LGBM score: ', np.mean(STACK_score))
def read_testset(filename="stage_1_sample_submission.csv"):

    df = pd.read_csv(filename)

    df["Image"] = df["ID"].str.slice(stop=12)

    df["Diagnosis"] = df["ID"].str.slice(start=13)

    

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]

    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    

    return df



submit = read_testset(filename="../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_sample_submission.csv")



submit.iloc[:, :] = Y_pred

submit = submit.stack().reset_index()

submit.insert(loc=0, column='ID', value=submit['Image'].astype(str) + "_" + submit['Diagnosis'])

submit = submit.drop(["Image", "Diagnosis"], axis=1)

submit.to_csv('stacking_{}_models_repeat_{}_times_stage2.csv'.format(NUM_MODELS, REPEAT), index=False)
submit.head()