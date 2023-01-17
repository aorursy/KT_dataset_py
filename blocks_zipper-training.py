import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import fastai

import math

import datetime

import matplotlib.pyplot as plt

import os

import lightgbm as lgb

from tqdm import tqdm, tqdm_notebook

import plotly.graph_objs as go
df_all = pd.read_pickle('../input/df_all_norm.pkl') 

size_per_episode = pd.read_pickle('../input/size_per_episode.pkl')
df_all.episode.unique().size
size_per_episode.head(3)
def select_train_test_episodes(size_per_episode, label = "walking",percent = 0.2):

    target = label # "open" or "walking"

    percent_test = percent

    size_per_episode= size_per_episode.reset_index().set_index([target,'episode']).sort_index()

    target_per_class = size_per_episode.groupby(target).sum()*percent_test

    test_episodes = []

    target_length = 0

    while target_length < target_per_class.iloc[0].length:

        sample_episode = size_per_episode.xs(0).sample()

        target_length += sample_episode.length.values[0]

        test_episodes.append(sample_episode.index.values[0])

    target_length = 0

    while target_length < target_per_class.iloc[1].length:

        sample_episode = size_per_episode.xs(1).sample()

        target_length += sample_episode.length.values[0]

        test_episodes.append(sample_episode.index.values[0])

    train_episodes = size_per_episode.index.levels[1].values

    train_episodes = set(train_episodes) - set(test_episodes)

    print(f"train episode:{train_episodes}, test episodes {test_episodes}")

    return train_episodes,test_episodes



def balance_training_set(X_trn, Y_trn):

    Y_trn = np.array(Y_trn)

    walking_count = np.count_nonzero(Y_trn)

    still_count = np.count_nonzero(1-Y_trn)

    print(f'walking {walking_count}, still {still_count}, ratio {walking_count/still_count:.2f}')

    Y_trn_still_idx = np.where(Y_trn == 0)[0]

    idx_rebalance = np.om.choice(Y_trn_still_idx, abs(walking_count-still_count))

    Y_trn = np.append(Y_trn,Y_trn[idx_rebalance])

    X_trn = np.vstack((X_trn,X_trn[idx_rebalance]))

    return X_trn, Y_trn



# A bit of feature engineering

def create_training_with_fft(window_size = 128, step_size = 10, episode_list=[],

                             features=["aa","ax","ay","az"], fft_only=False, 

                             remove_0_freq=True, add_fft=True, get_episodelist = False,

                             return_neg_freq = True, flatten = False,

                             wwa=0, nwwa = 0, wnstd=0, nwnstd=0):

    # ww: window warping augmentation (0.1 is 10% from original time scale max)

    x = [] ; y = [] ; blist = []

    if not episode_list: episode_list = df_all.episode.unique()

    sample_period = 30 #ms

    if wwa > 0:

        nresamples=nwwa; 

        resample_list = sample_period+np.arange(-nresamples,nresamples+1)*sample_period/nresamples*wwa 

    else:

        resample_list = [sample_period]

    for episode in tqdm(episode_list):

        for ts in resample_list:

            for nnoise in range(nwnstd+1):

                dftmp = df_all[df_all.episode==episode].copy()

                len_ep = dftmp.shape[0]

                dftmp[features]= dftmp[features] + np.random.normal(0,wnstd,[len_ep,len(features)])

                dftmp = dftmp.resample(f'{ts}ms').interpolate(method="linear")

                for i in range(0, dftmp.ma.size-window_size,step_size):

                    dfsub = dftmp.iloc[i:i+window_size]

                    feat_val=dfsub[features].values

                    if remove_0_freq:

                        feat_val_fft = (dfsub - dfsub.mean())[features].values

                    else:

                        feat_val_fft = feat_val

                    n = len(feat_val_fft)

                    fft_feat_val = abs(np.fft.fft(feat_val_fft.T)/n).T    

                    if not return_neg_freq:

                        fft_feat_val = fft_feat_val[np.arange(n/2,dtype=int)] 

                    if add_fft:

                        if fft_only:

                            if flatten: x.append(fft_feat_val.T.flatten())

                            else: x.append(fft_feat_val.T)

                        else:

                            if flatten:x.append(np.hstack((feat_val.T.flatten(), fft_feat_val.T.flatten())))

                            else: x.append(np.vstack((feat_val.T, fft_feat_val.T)))

                    else:

                        if flatten: x.append(feat_val.T.flatten())

                        else: x.append(feat_val.T)

                    y.append(dfsub[target].values[0])

                    blist.append(episode)

    if get_episodelist: return np.array(x),np.array(y),blist

    else: return np.array(x),np.array(y)

from sklearn.metrics import roc_curve, recall_score, precision_score, accuracy_score

def get_metrics(X,y,m):

    ypred = m.predict(X)

    return  accuracy_score(y,ypred), precision_score(y,ypred),recall_score(y,ypred)



import seaborn as sns

from sklearn.metrics import confusion_matrix,precision_score, recall_score, accuracy_score, f1_score

# recall for zipper open is maximize rate of detection the flyer is open (minimize false negatives).

# recall: real->detected max, precision: detected -> real max (minimize false positives)



def plot_confusion_matrix(Y_true, Y_pred, label="walking",ax=[]):

    cm = confusion_matrix( Y_true, Y_pred)

    prec= precision_score(Y_true, Y_pred); rec = recall_score(Y_true,Y_pred)

    acc = accuracy_score(Y_true,Y_pred); f1 = f1_score(Y_true,Y_pred)

    if label is "walking":

        labels = ["still","walking"]

    else:

        labels = ["close","open"]

    cm_df = pd.DataFrame(cm,index = labels, columns = labels)

    sns.heatmap(cm_df, annot=True,ax=ax)

    ax.set_title(f'Confusion Matrix, acc {acc:.2f} - f1 {f1:.2f} - rec {rec:.2f}');

    ax.set_ylabel('True label')

    ax.set_xlabel('Predicted label')



def print_quadrants_examples(X, Y, blist, m):

    # Analysis of the bad ones



    preds = m.predict(X)

    blist = np.array(blist); #Y_val = np.array(Y_val); 

    idx_fails = preds != Y

    idx_ok = preds == Y



    idx_fn = np.where(idx_fails & (Y == 1) == True)[0]

    idx_fp = np.where(idx_fails & (Y == 0) == True)[0]

    b_fn = blist[idx_fn]; b_fp = blist[idx_fp]

    b_fn, b_fn_count = np.unique(b_fn, return_counts=True)

    b_fp, b_fp_count = np.unique(b_fp, return_counts=True)

    idx_tn = np.where(idx_ok & (Y == 0) == True)[0]

    idx_tp = np.where(idx_ok & (Y == 1) == True)[0]

    print(f'false neg {b_fn}, counts {b_fn_count}. false pos {b_fp}, count {b_fp_count}')

    quadrants_dict = {'fn': 'false negatives', 'fp': 'false positives', 'tp': 'true positives', 'tn': 'true negatives'}

    def print_quadrant(quadrant='fp', idx=idx_fp, blist = blist):

        if len(idx) >= 6:

            fig,ax = plt.subplots(2,3,figsize=(15,5))

            for _, axi in enumerate(fig.axes):

                i=np.random.choice(idx)

                episode = blist[i]

                axi.plot(X[i].reshape(6,-1)[0]); 

                axi.set_title(f'ep:{episode}, {quadrants_dict[quadrant]}')

    print_quadrant(quadrant='tp', idx=idx_tp)

    print_quadrant(quadrant='tn', idx=idx_tn)

    print_quadrant(quadrant='fp', idx=idx_fp)

    print_quadrant(quadrant='fn', idx=idx_fn)
# targeting walking and non walking

target = "walking"

train_episodes,test_episodes = select_train_test_episodes(size_per_episode,label = target, percent = 0.2)

# lgb_params ={'boosting_type': 'gbdt', 'lambda_l1': 0.07013102270356747, 'lambda_l2': 0.8442787399572204, 'max_bin': 82, 'min_data_in_leaf': 310, 'num_leaves': 33, 'subsample': 0.8377796877710598, 'n_estimators': 556}
features = ["ax","ay","az"];

ws = 128; fft_only = False;  remove_0freq = True

X_trn,Y_trn, blist_trn = create_training_with_fft(episode_list=train_episodes,features=features, window_size=ws, fft_only=fft_only, remove_0_freq=remove_0freq, get_episodelist=True, return_neg_freq = True, flatten=True)

X_val,Y_val, blist_val = create_training_with_fft(episode_list=test_episodes,features=features, window_size=ws, fft_only=fft_only, remove_0_freq=remove_0freq, get_episodelist=True, return_neg_freq = True, flatten=True)

m = lgb.LGBMClassifier()

m.fit(X_trn,Y_trn)

fig,axes = plt.subplots(1,2,figsize=(24,6))

plot_confusion_matrix(Y_trn,m.predict(X_trn),label="walking",ax=axes[0]);axes[0].set_title("Training: " + axes[0].get_title());

plot_confusion_matrix(Y_val,m.predict(X_val),label="walking",ax=axes[1]);axes[1].set_title("Validation: " + axes[1].get_title());
print_quadrants_examples(X_val, Y_val, blist_val, m)
target = "open"

train_episodes,test_episodes = select_train_test_episodes(size_per_episode[size_per_episode.walking==1],label = target, percent = 0.2)
# setting episodes to typical sample for reproducibility

train_episodes= [ 0,  2,  5, 10, 11, 12, 18, 19, 20, 21, 22, 23, 27, 29, 30, 34, 36, 38, 39, 40, 43, 44, 45, 46, 47, 50, 51, 54, 55, 56]

test_episodes = [ 6, 13, 49,  9, 33, 35, 41, 17, 53,  1]
# features = ['ma',"mx","my","mz"]

dataset_params = {'window_size':128, 'fft_only':False, 'remove_0_freq': True,

                 'features': ["mx","my","mz","ax","ay","az"],'return_neg_freq':True, 'get_episodelist':True}

# m = lgb.LGBMClassifier(random_state=1,max_bins=4)

X_trn,Y_trn, blist_trn = create_training_with_fft(episode_list = train_episodes, flatten=True, **dataset_params) 

X_val,Y_val, blist_val = create_training_with_fft(episode_list = test_episodes, flatten=True, **dataset_params)

m = lgb.LGBMClassifier(random_state=1)

m.fit(X_trn,Y_trn)

fig,axes = plt.subplots(1,2,figsize=(24,6))

plot_confusion_matrix(Y_trn,m.predict(X_trn),label="open",ax=axes[0]);axes[0].set_title("Training: " + axes[0].get_title());

plot_confusion_matrix(Y_val,m.predict(X_val),label="open",ax=axes[1]);axes[1].set_title("Validation: " + axes[1].get_title());
augmentation_params = {'nwwa':1, 'wwa':0.05,'nwnstd':1, 'wnstd':0.02}

# m = lgb.LGBMClassifier(random_state=1,max_bins=4)

X_trn,Y_trn, blist_trn = create_training_with_fft(episode_list = train_episodes, flatten=True, **dataset_params, **augmentation_params) 

X_val,Y_val, blist_val = create_training_with_fft(episode_list = test_episodes, flatten=True, **dataset_params)

m = lgb.LGBMClassifier(random_state=1)

m.fit(X_trn,Y_trn)

fig,axes = plt.subplots(1,2,figsize=(24,6))

plot_confusion_matrix(Y_trn,m.predict(X_trn),label="open",ax=axes[0]);axes[0].set_title("Training: " + axes[0].get_title());

plot_confusion_matrix(Y_val,m.predict(X_val),label="open",ax=axes[1]);axes[1].set_title("Validation: " + axes[1].get_title());
Y_preds_lgbm = m.predict_proba(X_val)[:,1]

Y_preds_classes_lgbm = m.predict(X_val)

print_quadrants_examples(X_val, Y_val, blist_val, m)
import csv

from hyperopt import STATUS_OK

from timeit import default_timer as timer



# Objective definition

def objective(hyperparameters):

    """Objective function for Gradient Boosting Machine Hyperparameter Optimization.

       Writes a new line to `outfile` on every iteration"""

    

    # Keep track of evals

    global ITERATION

    if ITERATION == 0:

        # Adds headers to the csv file

        of_connection = open(OUT_FILE, 'w')

        writer = csv.writer(of_connection)

        # Write column names

        headers = ['loss', 'hyperparameters', 'iteration']

        writer.writerow(headers)

        of_connection.close()

    ITERATION += 1

    

    

    # Retrieve the subsample

    subsample = hyperparameters['boosting_type'].get('subsample', 1.0)

    

    # Extract the boosting type and subsample to top level keys

    hyperparameters['boosting_type'] = hyperparameters['boosting_type']['boosting_type']

    hyperparameters['subsample'] = subsample

    

    # Num boosting round <=> n estimators

    if 'n_estimators' in hyperparameters:

        del hyperparameters['n_estimators']

    

    # Make sure parameters that need to be integers are integers

    for parameter_name in ['num_leaves', 'min_data_in_leaf', 'max_bin']:

        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

    

    # Perform n_folds cross validation

    cv_results = lgb.cv(hyperparameters, train_set,  nfold = 5, metrics = 'binary_logloss', seed = 1,

                       num_boost_round = 10000, early_stopping_rounds = 10)



    # Extract the best score

    best_score = cv_results['binary_logloss-mean'][-1]

    loss = best_score

    n_estimators = len(cv_results['binary_logloss-mean'])

   

    hyperparameters['n_estimators'] = n_estimators

    

    # Write to the csv file ('a' means append)

    of_connection = open(OUT_FILE, 'a')

    writer = csv.writer(of_connection)

    writer.writerow([loss, hyperparameters, ITERATION])

    of_connection.close()



    # Dictionary with information for evaluation

    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION, 'status': STATUS_OK}



# Optimization space

from hyperopt import hp

from hyperopt.pyll.stochastic import sample

space = {

    'boosting_type': hp.choice('boosting_type', 

                                            [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 

                                             {'boosting_type': 'goss', 'subsample': 1.0}]),

    'num_leaves': hp.quniform('num_leaves', 2, 150, 1),

    'max_bin': hp.quniform('max_bin', 3, 100, 1),

    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 500, 5),

    'lambda_l1': hp.uniform('lambda_l1', 0.0, 1.0),

    'lambda_l2': hp.uniform('lambda_l2', 0.0, 1.0),

}
from hyperopt import Trials, tpe, fmin

opt_loops = 1 ## << change if optimizing, max before reaching time limit is 27 loops

# Record results

global  ITERATION; ITERATION = 0

trials = Trials()

OUT_FILE = 'zipper_lgb_optim.csv'



# Run optimization

m = lgb.LGBMClassifier(random_state=1)

train_set = lgb.Dataset(X_trn, label = Y_trn)

test_set = lgb.Dataset(X_val, label = Y_val)



best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,

            max_evals = opt_loops)
trials_dict = sorted(trials.results, key = lambda x: x['loss'])

opt_params = trials_dict[0]['hyperparameters']

print(opt_params)
#from overnight opt

opt_params = {'boosting_type': 'gbdt', 'lambda_l1': 0.002582327981327716, 'lambda_l2': 0.481905023933198, 'max_bin': 68, 'min_data_in_leaf': 475, 'num_leaves': 47, 'subsample': 0.8321264957420181, 'n_estimators': 4129}

m = lgb.LGBMClassifier(**opt_params, random_state=1)

m.fit(X_trn,Y_trn)



fig,axes = plt.subplots(1,2,figsize=(24,6))

plot_confusion_matrix(Y_trn,m.predict(X_trn),label="open",ax=axes[0]);axes[0].set_title("Training: " + axes[0].get_title());

plot_confusion_matrix(Y_val,m.predict(X_val),label="open",ax=axes[1]);axes[1].set_title("Validation: " + axes[1].get_title());
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.utils.data as tdatautils



from fastai.basic_data import DataBunch, DatasetType

from fastai.basic_train import Learner

from fastai.train import *

from fastai import *



import pdb
class ZipNet(nn.Module):

    ## input channel 128x1

    def __init__(self, ks = 5, ws = 128, feature_size = 6):

        # ws: Window size, which is the time series input size

        # feature_size: 6 (number of channels, time series first 3, fft last 3)

        # ks: kernel_size        

        

        

        super(ZipNet,self).__init__()

        num_conv = 3; padding = ks//2; num_layer_conv = 72 ; num_fc = 30; 

        self.flat_conv_size = ws

        self.feature_size = feature_size

        for i in range(num_conv):

            self.flat_conv_size = (self.flat_conv_size)/2

        self.flat_conv_size = int(self.flat_conv_size)*num_layer_conv

        self.flat_fft_size = ws * (feature_size // 2) 

        self.conv1 = nn.Conv1d(feature_size // 2,num_layer_conv,ks,padding=padding)

        self.conv1b = nn.Conv1d(num_layer_conv,num_layer_conv,ks,padding=padding)

        self.conv2 = nn.Conv1d(num_layer_conv,num_layer_conv,ks,padding=padding)

        self.conv2b = nn.Conv1d(num_layer_conv,num_layer_conv,ks,padding=padding)

        self.conv3 = nn.Conv1d(num_layer_conv,num_layer_conv,ks,padding=padding)

        self.conv3b = nn.Conv1d(num_layer_conv,num_layer_conv,ks,padding=padding)

        self.fc1 = nn.Linear(self.flat_conv_size + self.flat_fft_size,num_fc)

        self.fc2 = nn.Linear(num_fc,1)





    def forward(self, x):

        # input is episode, [feats,fft_feats], samples

        idx_feat = np.arange(self.feature_size // 2)

        idx_fft = self.feature_size // 2 + np.arange(self.feature_size // 2)

        x_feat  = x[:,idx_feat,:]; x_fft = x[:,idx_fft,:]

        # Conv on features

        xf = self.conv1(x_feat) # 128 

        xf = self.conv1b(F.relu(xf)) # 128 

        xf = F.max_pool1d(F.relu(xf), 2) # 128 , 64

        xf = self.conv2(xf) # 64

        xf = self.conv2b(F.relu(xf))

        xf = F.max_pool1d(F.relu(xf), 2) # 64 -> 32

        xf = self.conv3(xf) # 32

        xf = self.conv3b(F.relu(xf))

        xf = F.max_pool1d(F.relu(xf), 2) # 



        # Flatten and fully connected

        xf = xf.view(-1,self.flat_conv_size)

        xfft= x_fft.view(-1, self.flat_fft_size)

        x = torch.cat((xf, xfft), 1)

        x = F.relu(self.fc1(x))

        x = torch.sigmoid(self.fc2(x).squeeze())

        return x
# X_trn sample, feature, time

X_trn,Y_trn, blist_trn = create_training_with_fft(episode_list = train_episodes, flatten=False, **dataset_params, **augmentation_params) 

X_val,Y_val, blist_val = create_training_with_fft(episode_list = test_episodes, flatten=False, **dataset_params)
train_ds = tdatautils.TensorDataset(torch.from_numpy(X_trn).to(torch.float32),torch.from_numpy(Y_trn).to(torch.float32))

valid_ds = tdatautils.TensorDataset(torch.from_numpy(X_val).to(torch.float32),torch.from_numpy(Y_val).to(torch.float32))
batch_size = 64

model_data = DataBunch.create(train_ds,valid_ds, valid_ds, bs=batch_size) # fastai library's Databunch
zipnet = ZipNet(ks=3, feature_size=X_trn.shape[1])
zipnet(model_data.one_batch(DatasetType.Train)[0])
ziplearner = Learner(model_data, zipnet, opt_func=torch.optim.Adam, loss_func=F.binary_cross_entropy)  # fastai's library Learner
ziplearner.lr_find()

ziplearner.recorder.plot()
ziplearner.fit_one_cycle(10,1e-3,wd=1e-3)
res_trn = ziplearner.get_preds(ds_type=DatasetType.Train)

res_val = ziplearner.get_preds(ds_type=DatasetType.Valid)

trn_pred = res_trn[0].numpy() > 0.5; trn_target = res_trn[1].numpy()

val_pred = res_val[0].numpy() > 0.5; val_target = res_val[1].numpy()
fig,axes = plt.subplots(1,2,figsize=(24,6))

plot_confusion_matrix(trn_target,trn_pred,label="open",ax=axes[0]);axes[0].set_title("Training: " + axes[0].get_title());

plot_confusion_matrix(val_target,val_pred,label="open",ax=axes[1]);axes[1].set_title("Validation: " + axes[1].get_title());
Y_preds_convnet = res_val[0].numpy()
class ReZipNet(nn.Module):

    ## input channel wsx3

    def __init__(self, num_layers = 10, hidden_dim = 5, fs = 128, dropout=0.2):

        # fs: feature size

        super(ReZipNet,self).__init__()

        self.lstm = nn.LSTM(fs, hidden_dim, num_layers=num_layers,dropout=dropout)

        self.hidden2prob = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

        



    def forward(self,x):

        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))

        lstm_out = self.dropout(lstm_out)

        y = self.hidden2prob(lstm_out.view(len(x), -1))

        y = torch.sigmoid(y).squeeze()

        return y

# X_trn sample, feature, time

X_trn,Y_trn, blist_trn = create_training_with_fft(episode_list = train_episodes, flatten=False, **dataset_params, **augmentation_params) 

X_val,Y_val, blist_val = create_training_with_fft(episode_list = test_episodes, flatten=False, **dataset_params)

X_trn = X_trn.reshape(X_trn.shape[0],1,-1)

X_val = X_val.reshape(X_val.shape[0],1,-1)
train_ds = tdatautils.TensorDataset(torch.from_numpy(X_trn).to(torch.float32),torch.from_numpy(Y_trn).to(torch.float32))

valid_ds = tdatautils.TensorDataset(torch.from_numpy(X_val).to(torch.float32),torch.from_numpy(Y_val).to(torch.float32))
batch_size = 8

model_data = DataBunch.create(train_ds,valid_ds, valid_ds, bs=batch_size) # fastai library's Databunch

X_trn[0].shape[1]
nfeat = X_trn[0].shape[1]

zipnet = ReZipNet(num_layers=1, hidden_dim=10, fs = nfeat, dropout=0.0)
ziplearner = Learner(model_data, zipnet, opt_func=torch.optim.Adam, loss_func=F.binary_cross_entropy)
ziplearner.lr_find()

ziplearner.recorder.plot()
ziplearner.fit_one_cycle(10,1e-3)
res_trn = ziplearner.get_preds(ds_type=DatasetType.Train)

res_val = ziplearner.get_preds(ds_type=DatasetType.Valid)

trn_pred = res_trn[0].numpy() > 0.5; trn_target = res_trn[1].numpy()

val_pred = res_val[0].numpy() > 0.5; val_target = res_val[1].numpy()
fig,axes = plt.subplots(1,2,figsize=(24,6))

plot_confusion_matrix(trn_target,trn_pred,label="open",ax=axes[0]);axes[0].set_title("Training: " + axes[0].get_title());

plot_confusion_matrix(val_target,val_pred,label="open",ax=axes[1]);axes[1].set_title("Validation: " + axes[1].get_title());
Y_preds_lstm = res_val[0].numpy()
# average probabilities

fig,axes = plt.subplots(1,3,figsize=(24,6))

Y_preds = np.vstack((Y_preds_lgbm,Y_preds_lstm, Y_preds_convnet)); Y_ens=Y_preds.mean(axis=0) > 0.5

plot_confusion_matrix(val_target,Y_ens,label="open",ax=axes[0]); axes[0].set_title("Ens all: " + axes[0].get_title())

Y_preds = np.vstack((Y_preds_lgbm, Y_preds_convnet)); Y_ens=Y_preds.mean(axis=0) > 0.5

plot_confusion_matrix(val_target,Y_ens,label="open",ax=axes[1]); axes[1].set_title("Ens conv, lgbm: " + axes[1].get_title())

plot_confusion_matrix(val_target,Y_preds_convnet > 0.5,label="open",ax=axes[2]); axes[2].set_title("Convnet only: " + axes[2].get_title())



# majority vote

fig,axes = plt.subplots(1,3,figsize=(24,6))

Y_preds = np.vstack((Y_preds_lgbm,Y_preds_lstm, Y_preds_convnet));Y_ens= np.round((Y_preds>0.5).mean(axis=0))

plot_confusion_matrix(val_target,Y_ens,label="open",ax=axes[0]); axes[0].set_title("Ens all vote: " + axes[0].get_title())

Y_preds = np.vstack((Y_preds_lgbm, Y_preds_convnet));Y_ens= np.round((Y_preds>0.5).mean(axis=0))

plot_confusion_matrix(val_target,Y_ens,label="open",ax=axes[1]); axes[1].set_title("Ens conv, lgbm vote: " + axes[1].get_title())

plot_confusion_matrix(val_target,Y_preds_lgbm > 0.5,label="open",ax=axes[2]); axes[2].set_title("Lgbm only: " + axes[2].get_title())
