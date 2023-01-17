from google.colab import drive
drive.mount('/content/drive')
#!pip install tensorflow_addons
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Input, Dense, Add, Multiply,Bidirectional,GRU,Dropout
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_squared_error, f1_score, confusion_matrix, recall_score,precision_score
#from tensorflow.compat.v1.keras.layers import CuDNNLSTM,CuDNNGRU
import pandas as pd
import numpy as np
import random
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses, models, optimizers
import tensorflow_addons as tfa
import gc

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
from scipy import signal
import scipy as sp
import os

#add by NorwayPing
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
import math
from keras.constraints import maxnorm
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
import math
from sklearn.metrics import confusion_matrix,  plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Any results you write to the current directory are saved as output.

from typing import Tuple
def augment(X: np.array, y:np.array,z:np.array) -> Tuple[np.array, np.array,np.array]:
    
    X = np.vstack((X, np.flip(X, axis=1)))
    y = np.vstack((y, np.flip(y, axis=1)))
    z = np.vstack((z, np.flip(z, axis=1)))
    
    return X, y,z
# configurations and main hyperparammeters
EPOCHS = 150
NNBATCHSIZE = 16
GROUP_BATCH_SIZE = 4000
SEED = 321
LR = 0.001
SPLITS = 5

#add by NorwayPing
TRAIN_LEN = 5000000
SEEDS = [2020,321, 52]
COUNTER_RUN =3
VALID_SCORE_LIST = []
F1_SCORE_LIST = []

#add by NorwayPing
class_weight_dic= {0: 4.03185976,1:5.07089749,2:9.03344407,3:7.47632808,4:12.42043161,
  5:17.96057653,6:26.54068687, 7:18.86623369, 8:20.37396411,9:36.72318978,10: 140.23770291}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

def create_class_weight(y, mu=0.5):
        classes = np.unique(y)
        #class_weights = class_weight.compute_class_weight("balanced", classes, y)
        value_counts = pd.value_counts(y)

        total = len(y)
        keys = classes
        class_weights = dict()
        class_weight_log = []

        for key in keys:
            score = total / (value_counts[key])
            score_log = math.log(mu * total /(value_counts[key]))
            class_weights[key] = round(score, 2) if score > 1.0 else round(1.0, 2)
            class_weight_log.append( round(score_log, 2) if score_log > 1.0 else round(1.0, 2))

        return class_weights, class_weight_log

        
# read data
def read_data():
    train = pd.read_csv('/content/drive/My Drive/ion_kaggle/data/train_clean_trend.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test  = pd.read_csv('/content/drive/My Drive/ion_kaggle/data/test_clean_trend.csv', dtype={'time': np.float32, 'signal': np.float32})
    sub  = pd.read_csv('/content/drive/My Drive/ion_kaggle/data/sample_submission.csv', dtype={'time': np.float32}) 
    del train['type'],test['type']
    train['group'] = np.arange(train.shape[0])//500_000
    aug_df = train[train["group"] == 5].copy()
    aug_df["group"] = 10

    for col in ["signal", "open_channels"]:
        aug_df[col] += train[train["group"] == 8][col].values

    train = train.append(aug_df, sort=False).reset_index(drop=True)
    del aug_df
    gc.collect()

    train_pred=np.load("/content/drive/My Drive/ion_kaggle/data/lgb_reg_aug_opt.npz")['valid']  
    test_pred=np.load("/content/drive/My Drive/ion_kaggle/data/lgb_reg_aug_opt.npz")['test']   
    train_pred1=np.load("/content/drive/My Drive/ion_kaggle/data/lgb_pred_trend.npz")['valid'] 
    test_pred1=np.load("/content/drive/My Drive/ion_kaggle/data/lgb_pred_trend.npz")['test'] 

    return train, test, sub,train_pred,test_pred,train_pred1,test_pred1   





# create batches of 4000 observations
def batching(df, batch_size):
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df

# normalize the data (standard scaler). We can also try other scalers for a better score!
def normalize(train, test,features=['signal','oof_fea']):
    for item in features:
        train_input_mean = train[item].mean()
        train_input_sigma = train[item].std()
        train[item]= (train[item] - train_input_mean) / train_input_sigma
        test[item] = (test[item] - train_input_mean) / train_input_sigma
    return train, test

# get lead and lags features
def lag_with_pct_change(df, windows):
    for window in windows:    
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
                                                                                                    
    return df

# main module to run feature engineering. Here you may want to try and add other features and check if your score imporves :).
def run_feat_engineering(df, batch_size):
    # create batches
    df = batching(df, batch_size = batch_size)

    df = lag_with_pct_change(df, [1,2,3])

    df['signal_2'] = df['signal'] ** 2
    
    
    
    return df

# fillna with the mean and select features for training
def feature_selection(train, test):
    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    for feature in features:
        feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()
        train[feature] = train[feature].fillna(feature_mean)
        test[feature] = test[feature].fillna(feature_mean)
    return train, test, features

# model function (very important, you can try different arquitectures to get a better score. I believe that top public leaderboard is a 1D Conv + RNN style)
def Classifier(shape_):
    
    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = Conv1D(filters = filters,
                   kernel_size = 1,
                   padding = 'same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same', 
                              activation = 'tanh', 
                              dilation_rate = dilation_rate)(x)
            sigm_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same',
                              activation = 'sigmoid', 
                              dilation_rate = dilation_rate)(x)
            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters = filters,
                       kernel_size = 1,
                       padding = 'same')(x)
            res_x = Add()([res_x, x])
        return res_x
    
    inp = Input(shape = (shape_))

    x = wave_block(inp, 16, 3, 12)
    x = wave_block(x, 32, 3, 8)
    #x = wave_block(x, 64, 3, 4)
    x = wave_block(x, 128, 3, 1)
    #x = Bidirectional(CuDNNGRU(128,return_sequences=True))(x)

    x = Dropout(0.4)(x)
    out = Dense(11, activation = 'softmax', name = 'out')(x)
    
    model = models.Model(inputs = inp, outputs = out)
    
    opt = Adam(lr = LR)
    opt = tfa.optimizers.SWA(opt) 
    

    model.compile(loss = losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['accuracy'])

    return model

# function that decrease the learning as epochs increase (i also change this part of the code)

def lr_schedule(epoch):
    if epoch < 40:
        lr = LR
    elif epoch < 50:
        lr = LR / 3
    elif epoch < 60:
        lr = LR / 6
    elif epoch < 70:
        lr = LR / 9
    elif epoch < 80:
        lr = LR / 12
    elif epoch < 100:
        lr = LR / 15
    else:
        lr = LR / 100
    return lr

def lr_schedule(epoch):
    if epoch < 80:
        lr = LR
    elif epoch < 90:
        lr = LR / 4
    elif epoch < 100:
        lr = LR / 7
    elif epoch < 110:
        lr = LR / 12
    else:
        lr = LR / 100
    return lr

def lr_schedule(epoch):
    if epoch < 70:
      lr = LR
    elif epoch < 80:
      lr = LR / 3
    elif epoch < 90:
      lr = LR / 6
    elif epoch < 100:
      lr = LR / 9
    elif epoch < 110:
      lr = LR / 12
    elif epoch < 120:
      lr = LR / 15
    else:
      lr = LR / 100
    return lr

# class to get macro f1 score. This is not entirely necessary but it's fun to check f1 score of each epoch (be carefull, if you use this function early stopping callback will not work)
class MacroF1(Callback):
    def __init__(self, model, inputs, targets):
        self.model = model
        self.inputs = inputs
        self.targets = np.argmax(targets, axis = 2).reshape(-1)
        
    def on_epoch_end(self, epoch, logs):
        pred = np.argmax(self.model.predict(self.inputs), axis = 2).reshape(-1)
        score = f1_score(self.targets, pred, average = 'macro')
        print(f'F1 Macro Score: {score:.5f}')
'''
class MacroF1(Callback):
    def __init__(self, model, inputs, targets,filepath):
        self.model = model
        self.inputs = inputs
        self.targets = np.argmax(targets, axis = 2).reshape(-1)
        self.best_f1=0
        self.filepath=filepath
        
    def on_epoch_end(self, epoch, logs):
        pred = np.argmax(self.model.predict(self.inputs), axis = 2).reshape(-1)
        score = f1_score(self.targets, pred, average = 'macro')
        print(f'F1 Macro Score: {score:.5f}')
        if score>=self.best_f1:
            self.best_f1=score
            self.model.save_weights(self.filepath)
'''
class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        # X_train, y_train = self.data[0][0], self.data[0][1]
        # y_train = np.argmax(y_train, axis=2).reshape(-1)
        # train_pred = np.argmax(self.model.predict(X_train), axis=2).reshape(-1)
        # train_score = recall_score(y_train, train_pred, average="macro")        
        # train_score = f1_score(y_train, train_pred, average="macro")        
        # logs['Train_F1Macro'] = train_score

        X_valid, y_valid = self.data[1][0], self.data[1][1]
        y_valid = np.argmax(y_valid, axis=2).reshape(-1)
        valid_pred = np.argmax(self.model.predict(X_valid), axis=2).reshape(-1)
        valid_recall = recall_score(y_valid, valid_pred, average="macro")        
        valid_precision = precision_score(y_valid, valid_pred, average="macro")        
        valid_score = f1_score(y_valid, valid_pred, average="macro")        
        logs['Valid_F1Macro'] = valid_score
        logs['Valid_Recall'] = valid_recall
        logs['Valid_Precision'] = valid_precision

        print(f"Validation F1 Macro {valid_score:1.6f} Validation Recall {valid_recall:1.6f} Validation Precision {valid_precision:1.6f}")
        # print(' Train F1 Macro', train_score, 'Validation F1 Macro', valid_score)

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)
        
        gc.collect()  

    

        

def run_cv_model_by_batch(run_number, train, test, splits, batch_col, feats, sample_submission, nn_epochs, nn_batch_size,is_10_channel=False):
    
    seed = SEEDS[run_number]
    seed_everything(seed)
    K.clear_session()
    print(f'seed:{SEED}')


    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    #change by NorwayPing
    oof_ = np.zeros((len(train)//GROUP_BATCH_SIZE, GROUP_BATCH_SIZE, 11)) # build out of folds matrix with 11 columns, they represent our target variables classes (from 0 to 10)
    org_oof_ = np.zeros((len(train)//GROUP_BATCH_SIZE, GROUP_BATCH_SIZE, 11)) # build out of folds matrix with 11 columns, they represent our target variables classes (from 0 to 10)

    preds_ = np.zeros((len(test), 11))
    target = ['open_channels']
     
    #add by NorwayPing        
    train_open_channels = train[target]
    len_train = len(train)
    group_counts = train['group'].nunique()   
    group_unique =np.arange(group_counts)    
    org_group = np.arange(group_counts)    
    np.random.shuffle(group_unique)
    dic_group = dict(dict(zip(group_unique, org_group)))    
    group_id =np.repeat(group_unique,GROUP_BATCH_SIZE)   
    train['group'] = group_id
 
    
    group = train['group']
    kf = GroupKFold(n_splits=5)
    splits = [x for x in kf.split(train, train[target], group)]

    new_splits = []
    for sp in splits:
        new_split = []
        new_split.append(np.unique(group[sp[0]]))
        new_split.append(np.unique(group[sp[1]]))
        new_split.append(sp[1])    
        new_splits.append(new_split)
    # pivot target columns to transform the net to a multiclass classification estructure (you can also leave it in 1 vector with sparsecategoricalcrossentropy loss function)
    tr = pd.concat([pd.get_dummies(train.open_channels), train[['group']]], axis=1)
     
    tr.columns = ['target_'+str(i) for i in range(11)] + ['group']
    target_cols = ['target_'+str(i) for i in range(11)]

    train_tr = np.array(list(tr.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)

    #add by NorwayPing    
    train_target = np.array(list(train.groupby('group').apply(lambda x: x[target].values)))   
  


    train = np.array(list(train.groupby('group').apply(lambda x: x[feats].values)))
    test = np.array(list(test.groupby('group').apply(lambda x: x[feats].values)))

    #add by NorwayPing
    #test_flip = augment(test)

    for n_fold, (tr_idx, val_idx, val_orig_idx) in enumerate(new_splits[0:], start=0):
        train_x, train_y = train[tr_idx], train_tr[tr_idx]
        valid_x, valid_y = train[val_idx], train_tr[val_idx]
        
      
        train_target_y = train_target[tr_idx]
        train_x, train_y,train_target_y = augment(train_x, train_y,train_target_y)

        #add by NorwayPing                         
        org_group_x =   pd.Series(dic_group)[val_idx].values
        #  org_group_x =   pd.Series(dic_group)[val_idx].values
        class_weights, class_weight_log = create_class_weight(train_target[tr_idx].reshape(-1))

        #change by NorwayPing to improve class 6,7,8
        #class_weight_log =create_class_weight(train_y)
        
        sample_weight = np.ones(shape=(train_y.shape[0],train_y.shape[1],1))
        #train_target_y = train_target[tr_idx]        
        for i in range(11):
          sample_weight[train_target_y == i] = class_weight_log[i]

                      
        gc.collect()
        shape_ = (None, train_x.shape[2]) # input is going to be the number of feature we are using (dimension 2 of 0, 1, 2)
        model = Classifier(shape_)

        # using our lr_schedule function
        cp = ModelCheckpoint(f"model_run{run_number}_fold{n_fold+1}.h5", monitor='Valid_F1Macro', mode='max',save_best_only=True, verbose=1, period=1)
        cp.set_model(model)
        cp_loss = ModelCheckpoint(f"model_run{run_number}_fold{n_fold+1}.h5", monitor='val_loss', mode='min',save_best_only=True, verbose=1, period=1)
        cp_loss.set_model(model)
        es = EarlyStopping(monitor='val_loss',mode='min',restore_best_weights=True,verbose=1,patience=20)
        es.set_model(model)

        cb_lr_schedule = LearningRateScheduler(lr_schedule)               
        metric = Metric(model, [cp,cp_loss,es], [(train_x, train_y), (valid_x, valid_y)]) # ,es

        history = model.fit(train_x, train_y,epochs = nn_epochs,
                  callbacks = [cb_lr_schedule, metric], # MacroF1(model, [valid_x_sig, valid_x_oof], valid_y)
                  batch_size = nn_batch_size, verbose = 0,
                  validation_data = (valid_x, valid_y),
                  sample_weight=sample_weight)
        '''history = model.fit(train_x, train_y,epochs = nn_epochs,
                  callbacks = [cb_lr_schedule, metric], # MacroF1(model, [valid_x_sig, valid_x_oof], valid_y)
                  batch_size = nn_batch_size, verbose = 0,
                  validation_data = (valid_x, valid_y))'''
        # plot history
        # summarize history for accuracy
        plt.figure(figsize=(20,15))
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(20,15))
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        #model.load_weights(f'run{run_number}-fold{n_fold}.h5')


        preds_f = model.predict(valid_x)
         
         
        f1_score_ = f1_score(np.argmax(valid_y, axis=2).reshape(-1),  np.argmax(preds_f, axis=2).reshape(-1), average = 'macro') # need to get the class with the biggest probability
        print(f'Training run {run_number}  fold {n_fold + 1} completed. macro f1 score : {f1_score_ :1.5f}')

       
        #change by NorwayPing
        #preds_f = preds_f.reshape(-1, preds_f.shape[-1])
        oof_[val_idx,:] += preds_f      
        org_oof_[org_group_x,:] += preds_f 

        te_preds = model.predict(test)
        te_preds = te_preds.reshape(-1, te_preds.shape[-1])           
        preds_ += te_preds / SPLITS

        
         #add by NorwayPing
        if is_10_channel:
          VALID_SCORE_10_CHANNEL_LIST.append(f1_score_)
        else:
          VALID_SCORE_LIST.append(f1_score_)

    # calculate the oof macro f1_score
  
    f1_score_ = f1_score(np.argmax(train_tr, axis=2).reshape(-1),  np.argmax(oof_, axis = 2).reshape(-1), average = 'macro') # axis 2 for the 3 Dimension array and axis 1 for the 2 Domension Array (extracting the best class)
    print(f'Training run {run_number} completed. oof macro f1 score : {f1_score_:1.5f}')
 
    f1_score_ = f1_score(train_open_channels,  np.argmax(org_oof_, axis = 2).reshape(-1), average = 'macro') # axis 2 for the 3 Dimension array and axis 1 for the 2 Domension Array (extracting the best class)
    print(f'Training run {run_number} completed. oof macro f1 org score : {f1_score_:1.5f}')

    #add by NorwayPing 
    if is_10_channel:
          F1_SCORE_10_CHANNEL_LIST.append(f1_score_)
    else:
          F1_SCORE_LIST.append(f1_score_)
    
    return org_oof_.reshape(-1, 11),preds_
# this function run our entire program
def run_everything():
    
    print('Reading Data Started...')
    train, test, sample_submission,train_pred,test_pred,train_pred1,test_pred1= read_data()
     

    train['oof_fea']=train_pred
    test['oof_fea']=test_pred
    train['oof_fea1']=train_pred1
    test['oof_fea1']=test_pred1

    '''train=pd.concat([train[0:3600000],train[4000000:]])
    train=train.reset_index(drop=True)
    test=test.reset_index(drop=True)'''

   
    #test=test.reset_index(drop=True)
    
    train, test = normalize(train, test,features=['signal','oof_fea','oof_fea1'])
    print('Reading and Normalizing Data Completed')
        
    print('Creating Features')
    print('Feature Engineering Started...')
    train = run_feat_engineering(train, batch_size = GROUP_BATCH_SIZE)
    test = run_feat_engineering(test, batch_size = GROUP_BATCH_SIZE)
    print(train.shape)
    train, test, features = feature_selection(train, test)
    print(train.shape)
    print('Feature Engineering Completed...')
        
   
    print(f'Training Wavenet model with {SPLITS} folds of GroupKFold Started...')
    #add by NorwayPing
    preds_ = np.zeros((len(test), 11))
    oof_ = np.zeros((len(train), 11))
    for run in range(COUNTER_RUN):
        
        #preds_ +=run_cv_model_by_batch(run, train, test, SPLITS, 'group', features, sample_submission, EPOCHS, NNBATCHSIZE)/COUNTER_RUN
        oof_run,preds_run = run_cv_model_by_batch(run, train, test, SPLITS, 'group', features, sample_submission, EPOCHS, NNBATCHSIZE)
        oof_ += oof_run/COUNTER_RUN
        preds_ += preds_run/COUNTER_RUN

    f1_score_ = f1_score(train['open_channels'],  np.argmax(oof_, axis = 1), average = 'macro') # axis 2 for the 3 Dimension array and axis 1 for the 2 Domension Array (extracting the best class)
    print(f'Training completed. overall oof macro f1 score : {f1_score_:1.5f}')
    
    
    print(f'Training valid macro f1 score list: {VALID_SCORE_LIST}')
    print(f'Training oof macro f1 score list: {F1_SCORE_LIST}')

    print(f'Training valid macro f1 score mean: {np.mean(VALID_SCORE_LIST)}')
    print(f'Training oof macro f1 score mean: {np.mean(F1_SCORE_LIST)}')
    
    print(f'Training valid macro f1 score std: {np.std(VALID_SCORE_LIST)}')
    print(f'Training oof macro f1 score std: {np.std(F1_SCORE_LIST)}')

    sample_submission['open_channels'] = np.argmax(preds_, axis = 1).astype(int)
    print(sample_submission['open_channels'].value_counts())
    sample_submission.to_csv('rank 6-modify-highest -3runs.csv', index=False, float_format='%.4f')
    np.savez_compressed('rank 6-modify-highest -3runs.npz',valid=oof_, test=preds_)
    plot_cm(train['open_channels'],  np.argmax(oof_, axis = 1), 'all data wavenet \n f1=' + str('%.4f' %f1_score_))
run_everything()

# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud
def plot_cm(y_true, y_pred, title):
    figsize=(16,16)
    y_pred = y_pred.astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap='viridis', annot=annot, fmt='', ax=ax)
train = pd.read_csv('/content/drive/My Drive/ion_kaggle/data/train_clean_trend.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
test  = pd.read_csv('/content/drive/My Drive/ion_kaggle/data/test_clean_trend.csv', dtype={'time': np.float32, 'signal': np.float32})
sub  = pd.read_csv('/content/drive/My Drive/ion_kaggle/data/sample_submission.csv', dtype={'time': np.float32}) 
del train['type'],test['type']
train['group'] = np.arange(train.shape[0])//500_000
aug_df = train[train["group"] == 5].copy()
aug_df["group"] = 10

for col in ["signal", "open_channels"]:
    aug_df[col] += train[train["group"] == 8][col].values

train = train.append(aug_df, sort=False).reset_index(drop=True)
del aug_df
gc.collect()

oof=np.load("rank 6-modify-highest -3runs.npz")['valid'] 

oof_df = train[['signal','open_channels','group']].copy()
oof_df["oof"] = np.argmax(oof, axis=1)
oof_df = oof_df[oof_df["group"]<10]
gc.collect()

oof_f1 = f1_score(oof_df['open_channels'],oof_df['oof'],average = 'macro')
oof_recall = recall_score(oof_df['open_channels'],oof_df['oof'],average = 'macro')
oof_precision = precision_score(oof_df['open_channels'],oof_df['oof'],average = 'macro')

print(f"OOF F1 Macro Score: {oof_f1:.6f} - OOF Recall Score: {oof_recall:.6f} - OOF Precision Score: {oof_precision:.6f}")


plot_cm(oof_df['open_channels'],  (oof_df['oof']), 'all data wavenet \n f1=' + str('%.4f' %oof_f1))