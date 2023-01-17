!pip install tensorflow_addons==0.9.1
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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import gc
import os
from matplotlib import pyplot as plt
EPOCHS = 200
NNBATCHSIZE = 16
GROUP_BATCH_SIZE = 4000
LR = 0.0015
SPLITS = 5

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
def read_data():
    train = pd.read_csv('/kaggle/input/data-without-drift-with-kalman-filter/train.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test = pd.read_csv('/kaggle/input/data-without-drift-with-kalman-filter/test.csv', dtype={'time': np.float32, 'signal': np.float32})
    sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})
    
    Y_train_proba = np.load("/kaggle/input/ion-shifted-rfc-proba/Y_train_proba.npy")
    Y_test_proba = np.load("/kaggle/input/ion-shifted-rfc-proba/Y_test_proba.npy")
    
    for i in range(11):
        train[f"proba_{i}"] = Y_train_proba[:, i]
        test[f"proba_{i}"] = Y_test_proba[:, i]

    return train, test, sub
def batching(df, batch_size):
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df
def normalize(train, test):
    train_input_mean = train.signal.mean()
    train_input_sigma = train.signal.std()
    train['signal'] = (train.signal - train_input_mean) / train_input_sigma
    test['signal'] = (test.signal - train_input_mean) / train_input_sigma
    return train, test
def generate_signal_shift(df, windows):
    for window in windows:    
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
    return df
def run_feat_engineering(df, batch_size):
    df = batching(df, batch_size = batch_size)
    df = generate_signal_shift(df, [1, 2])
    df['signal_2'] = df['signal'] ** 2
    df['signal_3'] = df['signal'] ** 3
    return df
def feature_selection(train, test):
    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    for feature in features:
        feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()
        train[feature] = train[feature].fillna(feature_mean)
        test[feature] = test[feature].fillna(feature_mean)
    return train, test, features
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
    x = BatchNormalization()(inp)
    x = wave_block(x, 16, 3, 12)
    x = BatchNormalization()(x)
    x = wave_block(x, 32, 3, 8)
    x = BatchNormalization()(x)
    x = wave_block(x, 64, 3, 4)
    x = BatchNormalization()(x)
    x = wave_block(x, 128, 3, 1)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    out = Dense(11, activation = 'softmax', name = 'out')(x)
    
    model = models.Model(inputs = inp, outputs = out)
    
    opt = Adam(lr = LR)
    opt = tfa.optimizers.SWA(opt)
    model.compile(loss = losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['categorical_accuracy'])
    return model
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
class EarlyStoppingAtMaxMacroF1(Callback):
    """Stop training when the MacroF1 is at its max.
    """

    def __init__(self, model, inputs, targets, epochs, patience=0):
        super(EarlyStoppingAtMaxMacroF1, self).__init__()

        self.model = model
        self.inputs = inputs
        self.targets = np.argmax(targets, axis=2).reshape(-1)
        self.patience = patience

        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.last_epoch = epochs - 1

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as negative infinity.
        self.best = np.NINF

    def on_epoch_end(self, epoch, logs=None):
        pred = np.argmax(self.model.predict(self.inputs), axis=2).reshape(-1)
        current_score = f1_score(self.targets, pred, average='macro')
        print(f'F1 Macro Score: {current_score:.6f}')

        if np.greater(current_score, self.best):
            self.best = current_score
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)
        if epoch == self.last_epoch:
            self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
def run_cv_model_by_batch(train, test, n_splits, batch_col, feats, sample_submission, nn_epochs, nn_batch_size, seed):
    
    seed_everything(seed)
    K.clear_session()
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    oof_ = np.zeros((len(train), 11)) # build out of folds matrix with 11 columns, they represent our target variables classes (from 0 to 10)
    preds_ = np.zeros((len(test), 11))
    target = ['open_channels']
    group = train['group']
    kf = GroupKFold(n_splits=n_splits)
    splits = [x for x in kf.split(train, train[target], group)]

    new_splits = []
    for sp in splits:
        new_split = []
        new_split.append(np.unique(group[sp[0]]))
        new_split.append(np.unique(group[sp[1]]))
        new_split.append(sp[1])    
        new_splits.append(new_split)
    # pivot target columns to transform the net to a multiclass classification structure
    tr = pd.concat([pd.get_dummies(train.open_channels), train[['group']]], axis=1)

    tr.columns = ['target_'+str(i) for i in range(11)] + ['group']
    target_cols = ['target_'+str(i) for i in range(11)]
    train_tr = np.array(list(tr.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)
    train = np.array(list(train.groupby('group').apply(lambda x: x[feats].values)))
    test = np.array(list(test.groupby('group').apply(lambda x: x[feats].values)))

    for n_fold, (tr_idx, val_idx, val_orig_idx) in enumerate(new_splits[0:], start=0):
        train_x, train_y = train[tr_idx], train_tr[tr_idx]
        valid_x, valid_y = train[val_idx], train_tr[val_idx]
        print(f'Our training dataset shape is {train_x.shape}')
        print(f'Our validation dataset shape is {valid_x.shape}')

        gc.collect()
        shape_ = (None, train_x.shape[2])
        model = Classifier(shape_)
        
        cb_lr_schedule = LearningRateScheduler(lr_schedule)
        model.fit(train_x,train_y,
                  epochs = nn_epochs,
                  callbacks = [cb_lr_schedule, EarlyStoppingAtMaxMacroF1(model, valid_x, valid_y, nn_epochs, patience=40)],
                  batch_size = nn_batch_size,verbose = 2,
                  validation_data = (valid_x,valid_y))
        preds_f = model.predict(valid_x)
        f1_score_ = f1_score(np.argmax(valid_y, axis=2).reshape(-1),  np.argmax(preds_f, axis=2).reshape(-1), average = 'macro')
        print(f'Training fold {n_fold + 1} completed. macro f1 score : {f1_score_ :.6f}')
        preds_f = preds_f.reshape(-1, preds_f.shape[-1])
        oof_[val_orig_idx,:] += preds_f
        te_preds = model.predict(test)
        te_preds = te_preds.reshape(-1, te_preds.shape[-1])           
        preds_ += te_preds / n_splits
        model.save(f"model_seed_{seed}_fold_{n_fold}.mdl")
    # calculate the oof macro f1_score
    f1_score_ = f1_score(np.argmax(train_tr, axis = 2).reshape(-1),  np.argmax(oof_, axis = 1), average = 'macro')
    print(f'Training completed. oof macro f1 score : {f1_score_:.6f}')

    # save predictions
    np.save(f"preds_{seed}.npf", preds_)
    sample_submission['open_channels'] = np.argmax(preds_, axis = 1).astype(int)
    sample_submission.to_csv(f'submission_wavenet_{seed}.csv', index=False, float_format='%.4f')
def run_everything(seed):
    
    print('Reading Data Started.')
    train, test, sample_submission = read_data()
    print('Reading Data Completed.')
    
    print('Normalizing Data Started.')
    train, test = normalize(train, test)
    print('Normalizing Data Completed.')
        
    print('Feature Engineering Started.')
    train = train[:3640000].append(train[3840000:], ignore_index=True) # removed noise from train data
    print(f'Train shape: {train.shape}')
    
    train = run_feat_engineering(train, batch_size = GROUP_BATCH_SIZE)
    test = run_feat_engineering(test, batch_size = GROUP_BATCH_SIZE)
    train, test, features = feature_selection(train, test)
    print('Feature Engineering Completed.')

    print(f'Training Wavenet model with {SPLITS} folds of GroupKFold Started.')
    run_cv_model_by_batch(train, test, SPLITS, 'group', features, sample_submission, EPOCHS, NNBATCHSIZE, seed)
    print('Training completed.')
seeds = [1, 2, 3] # we used more seeds
for seed in seeds:
    run_everything(seed)
sample_sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')

file_name = 'submission_wavenet_{}.csv'
submissions = [pd.read_csv(file_name.format(i)) for i in seeds]

print(sample_sub.shape)
for s in submissions:
    print(s.shape)
predictions = [sub['open_channels'] for sub in submissions]
res_round_median = np.round(np.median(predictions, axis=0)).astype(int)
res_max = np.max(predictions, axis=0).astype(int)
# check the difference
diff = (res_max - res_round_median)
unique, counts = np.unique(diff, return_counts=True)
dict(zip(unique, counts))
train_df, test_df, sample_submission = read_data()
# check how we can split train data by groups
plt.figure(figsize=(15, 8))
plt.plot(train_df["time"], train_df["signal"], color="grey")
plt.title("Signals (Clean train data)", fontsize=20)
plt.xlabel("Time", fontsize=18)
plt.ylabel("Signal", fontsize=18)
plt.show()
def get_groups_data(train_df):
    A = train_df[:1000000]
    B = train_df[1000000:1500000].append(train_df[3000000:3500000], ignore_index=True)
    C = train_df[1500000:2000000].append(train_df[3500000:3640000], ignore_index=True)\
                                 .append(train_df[3840000:4000000], ignore_index=True) # removed noise for group C
    D = train_df[2000000:2500000].append(train_df[4500000:5000000], ignore_index=True)
    E = train_df[2500000:3000000].append(train_df[4000000:4500000], ignore_index=True)
    return A, B, C, D, E
A, B, C, D, E = get_groups_data(train_df)
plt.figure(figsize=(15, 8))
plt.plot(A.time, A.signal)
plt.plot(B.time, B.signal)
plt.plot(C.time, C.signal)
plt.plot(D.time, D.signal)
plt.plot(E.time, E.signal)
plt.title("Signals (Clean train data without noise, colored by groups)", fontsize=20)
plt.xlabel("Time", fontsize=18)
plt.ylabel("Signal", fontsize=18)
plt.show()
datasets = {
    'A': A,
    'B': B,
    'C': C,
    'D': D,
    'E': E,
}
def combine_predictions(df, batch_length, datasets):
    sub_final = pd.DataFrame()
    sub_final['time'] = sample_sub['time']
    sub_final['open_channels'] = sample_sub['open_channels']

    batch_len = batch_length
    for i in range(int(len(df) / batch_len)):
        df_batch = df[i * batch_len:i * batch_len + batch_len]

        batch_mean = df_batch['signal'].mean()
        default_group_name, default_group = next(iter(datasets.items()))
        min_mean_diff = np.abs(batch_mean - default_group['signal'].mean())
        clf_name = default_group_name

        for name, data in datasets.items():
            dataset_mean = data['signal'].mean()
            mean_diff = np.abs(batch_mean - dataset_mean)
            if mean_diff < min_mean_diff:
                min_mean_diff = mean_diff
                clf_name = name

        # use max(predictions) for the group with the high signal average number (D) and round(median(predictions)) for everything else (A, B, C, E).
        if (clf_name=='A'):
            sub_final.loc[i * batch_len:i * batch_len + batch_len - 1, 'open_channels'] = res_round_median[i * batch_len:i * batch_len + batch_len]
        elif (clf_name=='B'):
            sub_final.loc[i * batch_len:i * batch_len + batch_len - 1, 'open_channels'] = res_round_median[i * batch_len:i * batch_len + batch_len]
        elif (clf_name=='C'):
            sub_final.loc[i * batch_len:i * batch_len + batch_len - 1, 'open_channels'] = res_round_median[i * batch_len:i * batch_len + batch_len]
        elif (clf_name=='D'):
            sub_final.loc[i * batch_len:i * batch_len + batch_len - 1, 'open_channels'] = res_max[i * batch_len:i * batch_len + batch_len]
        elif (clf_name=='E'):
            sub_final.loc[i * batch_len:i * batch_len + batch_len - 1, 'open_channels'] = res_round_median[i * batch_len:i * batch_len + batch_len]
            
        print(f"group_{clf_name} prediction: {i} batch data")
    return sub_final
sub_final = combine_predictions(test_df, 100000, datasets)
# check if combine_predictions correctly defines each group
plt.figure(figsize=(15, 8))
plt.plot(test_df["time"], test_df["signal"], color="grey")
plt.title("Signals (Clean test data)", fontsize=20)
plt.xlabel("Time", fontsize=18)
plt.ylabel("Signal", fontsize=18)
plt.show()
# we can also combine our predictions by simple hard-coded solution, so let's check if there is no difference:
res_combined = np.concatenate([res_round_median[:500000], res_max[500000:600000], res_round_median[600000:700000], res_max[700000:800000], res_round_median[800000:]])

diff = (sub_final['open_channels'] - res_combined)
unique, counts = np.unique(diff, return_counts=True)
dict(zip(unique, counts))
sub_final.to_csv('wavenet_monte_carlo.csv', index=False, float_format='%.4f')
sub_final