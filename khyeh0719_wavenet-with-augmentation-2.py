!!pip install tensorflow_addons==0.9.1
import tensorflow_addons as tfa
import tensorflow as tf

from tensorflow.keras.layers import *

import pandas as pd

import numpy as np

import random

from tensorflow.keras.callbacks import Callback, LearningRateScheduler

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



import os

# Any results you write to the current directory are saved as output.

FOLD = 4 # fold 1 3 tends to be broken...

AUG_CNT = 15



# configurations and main hyperparammeters

EPOCHS = 180

NNBATCHSIZE = 16

GROUP_BATCH_SIZE = 4000

SEED = 321

LR = 0.001

SPLITS = 5



def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)

    

# read data

def read_data():

    train = pd.read_csv('../input/data-without-drift/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})

    test  = pd.read_csv('../input/data-without-drift/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})

    sub  = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})



    '''

    oof_files = ['rf_shift_20_per4000_group_split.pkl',

                 'wavenet_stacked_bigru_oof.pkl']

    

    from tqdm import tqdm

    for o_i, oof in enumerate(tqdm(oof_files)):

        Y_train_proba = pd.read_pickle('../oof/{}'.format(oof))['train'] #np.load("/kaggle/input/ion-shifted-rfc-proba/Y_train_proba.npy")

        Y_test_proba = pd.read_pickle('../oof/{}'.format(oof))['test']  #np.load("/kaggle/input/ion-shifted-rfc-proba/Y_test_proba.npy")

        

        for i in range(11):

            train[f"proba_{o_i}_{i}"] = Y_train_proba[:, i]

            test[f"proba_{o_i}_{i}"] = Y_test_proba[:, i]

    '''

    return train, test, sub



# create batches of 4000 observations

def batching(df, batch_size):

    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values

    df['group'] = df['group'].astype(np.uint16)

    return df



# normalize the data (standard scaler). We can also try other scalers for a better score!

def normalize(train, test):

    train_input_mean = train.signal.mean()

    train_input_sigma = train.signal.std()

    train['signal'] = (train.signal - train_input_mean) / train_input_sigma

    test['signal'] = (test.signal - train_input_mean) / train_input_sigma

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

    # create leads and lags (1, 2, 3 making them 6 features)



    #df = lag_with_pct_change(df, [i for i in range(1, 3)])

    # create signal ** 2 (this is the new feature)

    df['signal_2'] = df['signal']** 2

    

    return df



def train_update_groups(df):

    df['group_ind'] = -1

    step_size = 500_000

    groups = [0, 0, 1, 2, 4, 3, 1, 2, 3, 4]

    

    for i in range(df.shape[0]//step_size):

        a = i * step_size

        b = (i+1) * step_size

        df.loc[df.index[a:b], 'group_ind'] = groups[i]

    

    df['group_ind'] = df['group_ind'].astype(int)

    return df



def test_update_groups(df):

    df['group_ind'] = -1

    step_size = 100_000

    groups = [0, 2, 3, 0, 1, 4, 3, 4, 0, 2,

              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    

    for i in range(df.shape[0]//step_size):

        a = i * step_size

        b = (i+1) * step_size

        df.loc[df.index[a:b], 'group_ind'] = groups[i]

    

    df['group_ind'] = df['group_ind'].astype(int)

    return df



def convert_signal_into_probs(df, is_train=True, fit_dict=None):



    from scipy.stats import norm

    if is_train:

        df['group_ind'] = train_update_groups(df.copy())['group']



        filt = ((df.time>=364.229) & (df.time<382.343)) | ((df.time>=47.857) & (df.time<47.863))

        keep_filt= ~filt

        

        fit_dict = {}

        for gp in df['group_ind'].unique():

            

            x = df.loc[(df['group_ind'] == gp) & keep_filt, 'signal'].values

            y = df.loc[(df['group_ind'] == gp) & keep_filt, 'open_channels'].values

            

            fit_dict[gp] = {}

            fit_dict[gp]['open_channels'] = y.max()+1

            fit_dict[gp]['per_oc_dist'] = {}

            

            for s in np.arange(fit_dict[gp]['open_channels']):

                filt = y==s

                mean_, std_ = np.mean(x[filt]), np.std(x[filt])

                fit_dict[gp]['per_oc_dist'][s] = (mean_, std_)

                print("GP = {:2d}, OC = {:2d}, Mean={:.4f}, Std={:.4f}".format(gp, s, mean_, std_))

                    

    else:

        assert fit_dict is not None, "fit_dict should be provided when is_train is False"

        df['group_ind'] = test_update_groups(df.copy())['group']



    

    p_signal = np.zeros((df.shape[0], 11))

    for gp in df['group_ind'].unique():

        

        filt = df['group_ind'] == gp

        x = df.loc[filt, 'signal'].values

            

        for s in range(fit_dict[gp]['open_channels']):

            dist = fit_dict[gp]['per_oc_dist'][s]

            p_signal[filt, s] = norm.cdf(x, *dist)

            p_signal[filt, s] = np.where(p_signal[filt, s]>0.5, 1-p_signal[filt, s], p_signal[filt, s])

        

        p_signal_sum = p_signal.sum(axis=1)

        zero_sum_filt = p_signal_sum==0

        filt_1 = (filt)&(~zero_sum_filt); p_signal[filt_1, :] = p_signal[filt_1, :] / p_signal_sum[filt_1].reshape((-1,1))

        filt_2 = (filt)&(zero_sum_filt); p_signal[filt_2, :] = 1./(fit_dict[gp]['open_channels']-1) # for outlier signal, the prob for each channel is zero

        

        #print(gp, p_signal[filt])

        

    # remove signal column

    df.drop(['signal', 'group_ind'], axis=1, inplace=True)

    

    for s in range(11):

        df[f'sig_prob_{s}'] = p_signal[:, s]

        

    return df, fit_dict



# fillna with the mean and select features for training

def feature_selection(train, test):

    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time', 'group_ind']]

    print(features)

    train = train.replace([np.inf, -np.inf], np.nan)

    test = test.replace([np.inf, -np.inf], np.nan)

    for feature in features:

        feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()

        train[feature] = train[feature].fillna(feature_mean).astype(np.float32)

        test[feature] = test[feature].fillna(feature_mean).astype(np.float32)

    return train, test, features



# model function (very important, you can try different arquitectures to get a better score. I believe that top public leaderboard is a 1D Conv + RNN style)



#####

# wavenet + gru: cv: .940, lb: .942

#####

def Classifier_Wavenet_Gru(shape_):

    

    def cbr(x, out_layer, kernel, stride, dilation):

        x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)

        x = BatchNormalization()(x)

        x = Activation("relu")(x)

        return x

    

    def wave_block(x, filters, kernel_size, n):

        dilation_rates = [2**i for i in range(n)]

        

        x = Conv1D(filters, kernel_size=1, dilation_rate=1, strides=1, padding="same")(x)

        

        res_x = x

        for dilation_rate in dilation_rates:

            x = Conv1D(filters = filters,

                       kernel_size = kernel_size,

                       padding = 'causal',

                       dilation_rate = dilation_rate)(x)

            

            tanh_out = Activation('tanh')(x)

            sigm_out = Activation('sigmoid')(x)

            

            x = Multiply()([tanh_out, sigm_out])

            

            x = SpatialDropout1D(0.1)(x)

            

            x = Conv1D(filters = filters,

                       kernel_size = 1,

                       padding = 'same')(x)

            res_x = Add()([res_x, x])

            

        x = Activation("relu")(x)

        return res_x

    

    def bidirectional_wave_block(x, filters, kernel_size, n):

        x_f = wave_block(x, filters, kernel_size, n)

        x_rev = Lambda(lambda t: t[:,::-1,:])(x)

        x_b = wave_block(x_rev, filters, kernel_size, n)

        x_b = Lambda(lambda t: t[:,::-1,:])(x_b)

        x = concatenate([x_f, x_b])

        return x

    

    inp = Input(shape = (shape_))

    x = inp



    # reverse the seq

    x_rev = Lambda(lambda t: t[:,::-1,:])(x)

    

    # forward features

    x = wave_block(x, 64, 2, 13)

    x = wave_block(x, 64, 2, 13)

    x = wave_block(x, 64, 2, 13)

    

    # backward features

    x_rev = wave_block(x_rev, 64, 2, 13)

    x_rev = wave_block(x_rev, 64, 2, 13)

    x_rev = wave_block(x_rev, 64, 2, 13)

    x_rev = Lambda(lambda t: t[:,::-1,:])(x_rev)

    

    # stack features

    x = concatenate([x, x_rev])

    

    

    x = Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(64, return_sequences=True))(x)

    x = Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(64, return_sequences=True))(x)

    out = Dense(11, activation = 'softmax', name = 'out')(x)

    

    model = models.Model(inputs = inp, outputs = out)

    

    opt = Adam(lr = LR)

    opt = tfa.optimizers.SWA(opt)

    model.compile(loss = losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['accuracy'])

    return model



def Classifier_Wavenet(shape_):

    

    def cbr(x, out_layer, kernel, stride, dilation):

        x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)

        x = BatchNormalization()(x)

        x = Activation("relu")(x)

        return x

    

    def wave_block(x, filters, kernel_size, n):

        dilation_rates = [2**i for i in range(n)]

        

        x = Conv1D(filters, kernel_size=1, dilation_rate=1, strides=1, padding="same")(x)

        

        res_x = x

        for dilation_rate in dilation_rates:

            x = Conv1D(filters = filters,

                       kernel_size = kernel_size,

                       padding = 'same',

                       dilation_rate = dilation_rate)(x)

            

            tanh_out = Activation('tanh')(x)

            sigm_out = Activation('sigmoid')(x)

            

            x = Multiply()([tanh_out, sigm_out])

            

            x = SpatialDropout1D(0.1)(x)

            

            x = Conv1D(filters = filters,

                       kernel_size = 1,

                       padding = 'same')(x)

            res_x = Add()([res_x, x])

            

        x = Activation("relu")(x)

        return res_x

    

    inp = Input(shape = (shape_))

    x = inp

    

    x = wave_block(x, 64, 3, 12)

    x = wave_block(x, 64, 3, 8)

    x = wave_block(x, 64, 3, 4)

    x = wave_block(x, 64, 3, 1)

    

    #x = Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(64, return_sequences=True))(x)

    #x = Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(64, return_sequences=True))(x)

    

    out = Dense(11, activation = 'softmax', name = 'out')(x)

    

    model = models.Model(inputs = inp, outputs = out)

    

    opt = Adam(lr = LR)

    opt = tfa.optimizers.SWA(opt, clipvalue=1.0)

    model.compile(loss = losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['accuracy'])

    return model

# function that decrease the learning as epochs increase (i also change this part of the code)

def lr_schedule(epoch):

    if epoch < 30:

        lr = LR

    elif epoch < 40:

        lr = LR / 3

    elif epoch < 50:

        lr = LR / 5

    elif epoch < 60:

        lr = LR / 7

    elif epoch < 70:

        lr = LR / 9

    elif epoch < 80:

        lr = LR / 11

    elif epoch < 90:

        lr = LR / 13

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



def do_augment(train, feats, trn_idx, aug_time=5):

    

    import matplotlib.pyplot as plt

    #np.random.seed(423)

    

    # for more group 3 and 4 data

    aug_gp_mapping = {

        3: 1,

        4: 3

    }

    

    all_dfs = []

    

    for gp in [0,1,2,3,4]:

        

        orig_trn = train.loc[trn_idx, ['signal', 'group_ind', 'open_channels', 'group']]

        orig_trn = orig_trn.loc[orig_trn.group_ind == gp]

        orig_trn_x = orig_trn['signal'].values

        orig_trn_y = orig_trn['open_channels'].values

        

        print(orig_trn_x.shape, orig_trn_y.shape)

        print(orig_trn['open_channels'].value_counts().sort_index())

        means_ = []

        vars_ = []

        for oc in range(orig_trn_y.max()+1):

            filt = orig_trn_y == oc

            signals = orig_trn_x[filt]

            means_ += [signals.mean()]

            vars_ += [signals.var()]

            print(oc, signals.mean(), signals.var())

            

        if gp not in aug_gp_mapping.keys() or aug_time == 0:

            res = pd.DataFrame()

            res['signal'] = orig_trn_x

            res['open_channels'] = orig_trn_y

            all_dfs += [res]

            continue

            

        trn = train.loc[trn_idx, ['signal', 'group_ind', 'open_channels', 'group']]

        trn = trn.loc[trn.group_ind == aug_gp_mapping[gp],]

        trn_x = trn['signal'].values

        trn_y = trn['open_channels'].values

        signal_segments = trn.groupby('group').apply(lambda x: x['signal'].values)

        open_chnannel_segments = trn.groupby('group').apply(lambda x: x['open_channels'].values)

        

        # double the training size, per 4000 split, collect trn_X_.shape[0]//4000 segments

        signal_segments = [s for s in signal_segments if len(s) == GROUP_BATCH_SIZE]

        open_chnannel_segments = [s for s in open_chnannel_segments if len(s) == GROUP_BATCH_SIZE]

    

        # collect statistics before augmentation

        plt.title('before augment signal'); plt.plot(orig_trn_x[::1000]); plt.show()

        plt.title('before augment label'); plt.plot(orig_trn_y[::1000]); plt.show()

        

        # linear regression to find mean\var per open channel

        seq_len = orig_trn_y.max()+1

        selected_len = int(seq_len * 2 / 3)

        from sklearn.linear_model import LinearRegression

        lr = LinearRegression()

        lr.fit(np.arange(seq_len-selected_len, seq_len, 1).reshape((-1,1)), means_[-selected_len:])

        new_means = lr.predict(np.arange(seq_len).reshape((-1,1))) 

        

        new_vars = [v if (orig_trn_y == i).sum()>30 else max(vars_) for i, v in enumerate(vars_)]

        

        # adjust mean for original data

        for oc in range(orig_trn_y.max()+1):

            filt = orig_trn_y == oc

            signals = orig_trn_x[filt]

            src_m, src_v = signals.mean(), signals.var()

            if (orig_trn_y == oc).sum()<=30:

                src_v = max(vars_)

                

            tar_m, tar_v = new_means[oc], new_vars[oc] 

        

            signals = (signals-src_m)/(src_v**.5)*(tar_v**.5) + tar_m

            orig_trn_x[orig_trn_y == oc] = signals

            

        # augment data

        aug_size = aug_time*orig_trn_x.shape[0]//GROUP_BATCH_SIZE

        selections = [np.random.choice(np.arange(len(signal_segments)), orig_trn_y.max()//trn['open_channels'].max(), replace=False) for _ in range(aug_size)]

        

        combined_signal_segments = []

        combined_oc_segments = []

        for inds in selections:

            combined_signal_segments += [np.vstack([signal_segments[ix] for ix in inds]).sum(axis=0)]

            combined_oc_segments += [np.vstack([open_chnannel_segments[ix] for ix in inds]).sum(axis=0)]

        

        aug_trn_x = np.concatenate(combined_signal_segments)

        aug_trn_y = np.concatenate(combined_oc_segments)

        print(pd.Series(aug_trn_y).value_counts().sort_index())

        

        for oc in range(orig_trn_y.max()+1):

            aug_signals = aug_trn_x[aug_trn_y == oc]

            src_m, src_v = aug_signals.mean(), aug_signals.var()

            if (aug_trn_y == oc).sum()<=30:

                src_v = max(vars_)

                

            tar_m, tar_v = new_means[oc], new_vars[oc]

            

            aug_signals = (aug_signals-src_m)/(src_v**.5)*(tar_v**.5) + tar_m

            aug_trn_x[aug_trn_y == oc] = aug_signals

          

        print('Augment Data Shape:', aug_trn_x.shape)

        

        trn_x = np.concatenate([orig_trn_x, aug_trn_x], axis=0)

        trn_y = np.concatenate([orig_trn_y, aug_trn_y], axis=0)

        #assert False

        

        plt.title('after augment'); plt.plot(trn_x[::1000]); plt.show()

        plt.title('after augment label'); plt.plot(trn_y[::1000]); plt.show()

        for oc in range(trn_y.max()+1):

            signals = trn_x[trn_y == oc]

            print(oc, signals.mean(), signals.var())

    

        res = pd.DataFrame()

        res['signal'] = trn_x

        res['open_channels'] = trn_y

        all_dfs += [res]

        

    df = pd.concat(all_dfs, axis=0).reset_index(drop=True)

    df = run_feat_engineering(df, batch_size=GROUP_BATCH_SIZE)

    

    tr = pd.concat([pd.get_dummies(df.open_channels), df[['group']]], axis=1)

    tr.columns = ['target_'+str(i) for i in range(11)] + ['group']

    target_cols = ['target_'+str(i) for i in range(11)]

    targets = np.array(list(tr.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)

    signals = np.array(list(df.groupby('group').apply(lambda x: x[feats].values)))

    

    return signals, targets



# main function to perfrom groupkfold cross validation (we have 1000 vectores of 4000 rows and 8 features (columns)). Going to make 5 groups with this subgroups.

def run_cv_model_by_batch(train, test, splits, batch_col, feats, sample_submission, nn_epochs, nn_batch_size):

    

    seed_everything(SEED)

    K.clear_session()

    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)

    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)

    tf.compat.v1.keras.backend.set_session(sess)

    oof_ = np.zeros((len(train), 11)) # build out of folds matrix with 11 columns, they represent our target variables classes (from 0 to 10)

    preds_ = np.zeros((len(test), 11))

    target = ['open_channels']

    group = train['group']

    kf = GroupKFold(n_splits=5)

    splits = [x for x in kf.split(train, train[target], group)]



    new_splits = []

    for sp in splits:

        new_split = []

        new_split.append(np.unique(group[sp[0]]))

        new_split.append(np.unique(group[sp[1]]))

        new_split.append(sp[0])   

        new_split.append(sp[1])    

        new_splits.append(new_split)

    # pivot target columns to transform the net to a multiclass classification estructure (you can also leave it in 1 vector with sparsecategoricalcrossentropy loss function)

    tr = pd.concat([pd.get_dummies(train.open_channels), train[['group']]], axis=1)



    orig_train = train.copy()

    tr.columns = ['target_'+str(i) for i in range(11)] + ['group']

    target_cols = ['target_'+str(i) for i in range(11)]

    train_tr = np.array(list(tr.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)

    train = np.array(list(train.groupby('group').apply(lambda x: x[feats].values)))

    test = np.array(list(test.groupby('group').apply(lambda x: x[feats].values)))

    f1_scores = []

    

    for n_fold, (tr_idx, val_idx, trn_orig_ix, val_orig_idx) in enumerate(new_splits[0:], start=0):

        print(f'Training fold {n_fold + 1} started')

        if n_fold not in [FOLD]:

            continue

        

        # augtime = 5: fold 1 is very bad

        # aug = 1: [0.9375797429851022, 0.8188899525577876, 0.936195957056339, 0.7684088439541757, 0.9371989134178409]

        # aug = 1 (gp4 aug only): fold 0 worse, fold 1 still pretty bad

        # aug = 1 (gp3 aug only): fold 0 worse, fold 1 still pretty bad

        # train starting for fold 1: fold 1 is good=>.9385

        # remove signal **2, train from fold 0: .9365

        # use signal absolute instead, train from fold 0: .9358

        # use grad clip value = 1, sig**2 feat, train from fold 0: 0.93785, fold 1 breaks...

        # use fixed random seed in the beginning of augmentation

        # separate train for each fold...: fold 1: .9392, fold 3: .9369

        

        #train_x, train_y = train[tr_idx], train_tr[tr_idx]

        train_x, train_y = do_augment(orig_train, feats, trn_orig_ix, aug_time=AUG_CNT)

        valid_x, valid_y = train[val_idx], train_tr[val_idx]

        print(f'Our training dataset shape is {train_x.shape}')

        print(f'Our validation dataset shape is {valid_x.shape}')



        gc.collect()

        shape_ = (None, train_x.shape[2]) # input is going to be the number of feature we are using (dimension 2 of 0, 1, 2)

        model = Classifier_Wavenet(shape_)

        # using our lr_schedule function

        cb_lr_schedule = LearningRateScheduler(lr_schedule)

        model.fit(train_x,train_y,

                  epochs = nn_epochs,

                  callbacks = [cb_lr_schedule, MacroF1(model, valid_x, valid_y)], # adding custom evaluation metric for each epoch

                  batch_size = nn_batch_size,verbose = 2,

                  validation_data = (valid_x,valid_y))

        preds_f = model.predict(valid_x)

        f1_score_ = f1_score(np.argmax(valid_y, axis=2).reshape(-1),  np.argmax(preds_f, axis=2).reshape(-1), average = 'macro') # need to get the class with the biggest probability

        print(f'Training fold {n_fold + 1} completed. macro f1 score : {f1_score_ :1.5f}')

        preds_f = preds_f.reshape(-1, preds_f.shape[-1])

        oof_[val_orig_idx,:] += preds_f

        te_preds = model.predict(test)

        te_preds = te_preds.reshape(-1, te_preds.shape[-1])           

        preds_ += te_preds / SPLITS

        f1_scores += [f1_score_]

    # calculate the oof macro f1_score

    

    f1_score_ = f1_score(np.argmax(train_tr, axis = 2).reshape(-1),  np.argmax(oof_, axis = 1), average = 'macro') # axis 2 for the 3 Dimension array and axis 1 for the 2 Domension Array (extracting the best class)

    print(f'Training completed. oof macro f1 score : {f1_score_:1.5f}')

    print('all fold f1 scores = ', f1_scores)

    sample_submission['open_channels'] = np.argmax(preds_, axis = 1).astype(int)

    sample_submission.to_csv('submission_wavenet.csv', index=False, float_format='%.4f')

    

    pd.to_pickle({

        'train': oof_,

        'test': preds_,

    }, 'wavenet_aug_oof_{}_{}.pkl'.format(FOLD, AUG_CNT))

    

    import matplotlib.pyplot as plt

    plt.plot(sample_submission['open_channels'].values[::1000]); plt.show()

    

# this function run our entire program

def run_everything():

    

    print('Reading Data Started...')

    train, test, sample_submission = read_data()

    #train, test = normalize(train, test)

    print('Reading and Normalizing Data Completed')

        

    print('Creating Features')

    print('Feature Engineering Started...')

    

    train = train_update_groups(train)

    test = test_update_groups(test)

    

    train = run_feat_engineering(train, batch_size = GROUP_BATCH_SIZE)

    test = run_feat_engineering(test, batch_size = GROUP_BATCH_SIZE)

    

    #train, fit_dict = convert_signal_into_probs(train, is_train=True, fit_dict=None)

    #test, _ = convert_signal_into_probs(test, is_train=False, fit_dict=fit_dict)

    

    train, test, features = feature_selection(train, test)

    

    print('Feature Engineering Completed...')

        

   

    print(f'Training Wavenet model with {SPLITS} folds of GroupKFold Started...')

    run_cv_model_by_batch(train, test, SPLITS, 'group', features, sample_submission, EPOCHS, NNBATCHSIZE)

    print('Training completed...')

        

run_everything()



# 7: 0.94086

# 8: 0.9397