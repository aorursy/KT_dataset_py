import IPython.display as ipd  # To play sound in the notebook
telephone = '../input/00d3bba3wav/00d3bba3.wav'   # Telephone
flute = '../input/00d9fa61wav/00d9fa61.wav'   # Flute
squeak = '../input/013264d3wav/013264d3.wav'   # Squeak
cello = '../input/0184c390wav/0184c390.wav'   # Cello
shatter = '../input/01a39e95wav/01a39e95.wav'   # Shatter

ipd.Audio(telephone)
ipd.Audio(flute)
ipd.Audio(squeak)
ipd.Audio(cello)
ipd.Audio(shatter)
ipd.Audio('../input/0ce127f9/0ce127f9.wav')
ipd.Audio('../input/01e6e112wav/01e6e112.wav')
ipd.Audio('../input/026820e6wav/026820e6.wav')
ipd.Audio('../input/013264d3wav/013264d3.wav')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
from scipy.io import wavfile

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from collections import Counter
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import xgboost as xgb
from sklearn.utils import shuffle

from sklearn import manifold, datasets
from sklearn.preprocessing import scale

import os
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def get_spectra_win(y, L, N):
    dft = np.fft.fft(y)
    fl = np.abs(dft)
    xf = np.arange(0.0, N/L, 1/L)
    return (xf,fl)

def get_spectra(signal, fs, M = 1000, sM = 500):

    N = signal.shape[0]
    ind = np.arange(100, N, M)

    spectra = []
    meanspectrum = np.repeat(0,M)

    for k in range(1,len(ind)):
        n1 = ind[k-1]
        n2 = ind[k]
        y = signal[n1:n2]
        L = (n2-n1)/fs
        N = n2-n1
        (xq, fq) = get_spectra_win(y, L, N)
        spectra.append(fq)

    spectra = pd.DataFrame(spectra)
    meanspectrum = spectra.apply(lambda x: np.log(1+np.mean(x)), axis=0)
    stdspectrum = spectra.apply(lambda x: np.log(1+np.std(x)), axis=0)
    
    meanspectrum = meanspectrum[0:sM]
    stdspectrum = stdspectrum[0:sM]
    
    return (meanspectrum, stdspectrum) 

def get_width(w):
    if np.sum(w) == 0:
        return [0,0,0]
    else:
        z = np.diff(np.where(np.insert(np.append(w,0),0,0)==0))-1
        z = z[z>0]
    return [np.log(1+np.mean(z)),np.log(1+np.std(z)),np.log(1+np.max(z)),len(z)]

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def clear():
    os.system( 'cls' )
def get_training_dataset(training=1, dir_path='D:/python/audio_train/audio_train/'):
    
    if training==1:
        trainnames = pd.read_csv(dir_path + 'train.csv')
        labelnames = list(trainnames['label'].unique())
        le = preprocessing.LabelEncoder()
        le.fit(labelnames)
        files_labels = zip(trainnames['fname'].values, trainnames['label'].values)
    elif training==0:
        score_filelist = [str(x) for x in os.listdir(dir_path)]
        # 'D:/python/audio_test/audio_test'
        labelnames = np.repeat('unlabeled',len(score_filelist))
        le = preprocessing.LabelEncoder()
        le.fit(labelnames)
        files_labels = zip(score_filelist, labelnames)
    else:
        return []

    df_m = []
    df_sd = []
    df_sig = []
    df_mfcc = []
    df_fbank = []
    df_ssc = []    
    labels_processed = []
    filenames_processed = []

    i = 0
    
    for filename, labelname in files_labels:
        
        label = le.transform([labelname])[0]
        fname  = dir_path + filename
        fs, rawsignal = wavfile.read(fname)
        if rawsignal.size == 0:
            rawsignal = np.random.randint(0,2,44000)
            
        if rawsignal.dtype == 'int16':
            nb_bits = 16 # -> 16-bit wav files
        elif rawsignal.dtype == 'int32':
            nb_bits = 32 # -> 32-bit wav files
        max_nb_bit = float(2 ** (nb_bits - 1))
        rawsignal = rawsignal/max_nb_bit
        
        # signal features
        rawsignal_sq = rawsignal*rawsignal
        silenced = []
        sound = []
        attack = []
        for wd in [2000,10000]:
            rawsignal_sq_rm = running_mean(rawsignal_sq, wd)            
            w1 = 1*(rawsignal_sq_rm<0.01*np.max(rawsignal_sq_rm))
            silenced = silenced + get_width(w1)
            w2 = 1*(rawsignal_sq_rm<0.05*np.max(rawsignal_sq_rm))
            silenced = silenced + get_width(w2)            
            w3 = 1*(rawsignal_sq_rm>0.05*np.max(rawsignal_sq_rm))
            sound = sound + get_width(w3)
            w4 = 1*(rawsignal_sq_rm>0.25*np.max(rawsignal_sq_rm))
            sound = sound + get_width(w4)
            time_to_attack = np.min(np.where(rawsignal_sq_rm>0.99*np.max(rawsignal_sq_rm)))
            time_rel = np.where(rawsignal_sq_rm<0.2*np.max(rawsignal_sq_rm))[0]
            if (time_rel.size == 0):
                time_to_relax = len(rawsignal_sq_rm)
            elif (time_rel[time_rel>time_to_attack].size==0):
                time_to_relax = len(rawsignal_sq_rm)
            else:
                time_to_relax = np.min(time_rel[time_rel>time_to_attack])
            attack.append(np.log(1+time_to_attack))
            attack.append(np.log(1+time_to_relax))

        lr = len(rawsignal)
        zerocross_tot = np.log(1+np.sum(np.array(rawsignal[0:(lr-1)])*np.array(rawsignal[1:lr])<=0))
        zerocross_prop = np.sum(np.array(rawsignal[0:(lr-1)])*np.array(rawsignal[1:lr])<=0)/lr
        df_sig.append(sound+silenced+attack+[zerocross_tot,zerocross_prop])

        (m, sd) = get_spectra(rawsignal, fs, 2000, 1000 )
        df_m.append(m)
        df_sd.append(sd)

        labels_processed.append(label)     
        filenames_processed.append(filename)
        
        # mfcc
        
        mfcc_feat = librosa.feature.mfcc(rawsignal, sr = fs, n_mfcc=40)
        mfcc_feat = pd.DataFrame(np.transpose(mfcc_feat))       
        
        # mfcc_feat = mfcc(rawsignal, fs, nfft = 1103, numcep = 30 )
        # mfcc_feat = pd.DataFrame(mfcc_feat)
        mfcc_mean = mfcc_feat.apply(lambda x: np.mean(x), axis=0)
        mfcc_sd = mfcc_feat.apply(lambda x: np.std(x), axis=0)
        mfcc_max = mfcc_feat.apply(lambda x: np.max(x), axis=0)
        mfcc_med = mfcc_feat.apply(lambda x: np.median(x), axis=0)        
        mfcc_res = np.array(list(mfcc_mean)+list(mfcc_sd)+list(mfcc_max)+list(mfcc_med)+[np.log(1+len(rawsignal))])
        df_mfcc.append(mfcc_res)
             
        i = i+1
        labelname = labelname + ' '*(20-len(labelname))
        label_string = str(label) + ' '*(3-len(str(label)))
        i_str = str(i) + ' '*(5-len(str(i)))
            
        print('\r', i_str, filename, ' - ',labelname,' - ',label_string, end='', flush=True)
        
    # to data frames
    df_sig = pd.DataFrame(df_sig)
    df_sig.fillna(0, inplace = True)

    df_sd = pd.DataFrame(df_sd)
    df_m = pd.DataFrame(df_m)
    df_mfcc = pd.DataFrame(df_mfcc) 
    
    # predictors related to peaks 
    def num_peaks(x):
        x = np.array(x[0:len(x)])
        n10 = np.sum(x>0.10*np.max(x))
        n20 = np.sum(x>0.20*np.max(x))
        n50 = np.sum(x>0.50*np.max(x))
        n90 = np.sum(x>0.90*np.max(x))
        n99 = np.sum(x>0.99*np.max(x))
        lead_min = np.min(np.where(x==np.max(x)))
        cnt = 0
        w10 = get_width(1*(x>0.10*np.max(x)))
        w20 = get_width(1*(x>0.20*np.max(x)))
        w50 = get_width(1*(x>0.50*np.max(x)))
        w90 = get_width(1*(x>0.90*np.max(x)))
        w99 = get_width(1*(x>0.99*np.max(x)))  
        W = w10+w20+w50+w90+w99

        f_sc = np.sum(np.arange(0,len(x))*(x*x)/np.sum(x*x))


        i1 = np.where(x<0.10*np.max(x))[0]
        if i1.size == 0:
            lincoef_w = [0,0,0]
        else:
            a1 = i1[i1<lead_min]
            a2 = i1[i1>lead_min]

            if a1.size == 0:
                i1_left = 0
            else:
                i1_left = np.max(i1[i1<lead_min])
            if a2.size == 0:
                i1_right = 0
            else:
                i1_right = np.min(i1[i1>lead_min])

            lead_min_width = i1_right - i1_left  
            if (lead_min_width>2):
                poly_w = PolynomialFeatures(degree=2, include_bias = False)
                f_ind_w = poly_w.fit_transform(np.arange(i1_left,i1_right,1).reshape(-1, 1))
                clf_w = linear_model.LinearRegression()
                linmodel_w = clf_w.fit(f_ind_w, np.array(x[i1_left:i1_right]))
                lincoef_w = list(linmodel_w.coef_)+[linmodel_w.intercept_]
            else:
                lincoef_w = [0,0,0]

        S = np.sum(x)
        S_n = np.sum(x)/len(x)
        S2 = np.sqrt(np.sum(x*x))    
        S2_n = np.sqrt(np.sum(x*x))/len(x)
        integrals = [S,S_n,S2,S2_n]
        
        poly = PolynomialFeatures(degree=2, include_bias = False)
        f_ind = poly.fit_transform(np.arange(0,len(x)).reshape(-1, 1))
        clf = linear_model.LinearRegression()
        linmodel = clf.fit(f_ind, x)
        lincoef_spectrum = list(linmodel.coef_)+[linmodel.intercept_]

        high_freq_sum_50 = np.sum(x[0:50]>=0.5*np.max(x))
        high_freq_sum_90 = np.sum(x[0:50]>=0.9*np.max(x))

        r = [f_sc,n10,n20,n50,n90,n99,lead_min,high_freq_sum_50,high_freq_sum_90]+W+lincoef_spectrum+integrals+lincoef_w
        return r

    def runningMeanFast(x, N=20):
        return np.convolve(x, np.ones((N,))/N)[(N-1):]

    df_rm = df_m.apply(runningMeanFast, axis=1)
    df_sc = df_rm.apply(lambda x: x[np.arange(0,len(x),40)],axis=1)
    df_m_filt = df_m.apply(lambda x: x[np.arange(0,2,1)],axis=1)
    df_peaks = df_m.apply(num_peaks,axis=1)
    df_peaks = pd.DataFrame(list(df_peaks)) 
    df_rm = pd.DataFrame(df_rm)
    df_sc = pd.DataFrame(df_sc)
    df_m_filt = pd.DataFrame(df_m_filt)
    
    df_fbank = pd.DataFrame(df_fbank)
    df_ssc = pd.DataFrame(df_ssc)    
    #
    
    df_sd.columns = ['fft_sd'+str(i) for i in range(0,len(df_sd.columns))]
    df_m.columns = ['fft_mean'+str(i) for i in range(0,len(df_m.columns))]
    df_rm.columns = ['fft_rmean'+str(i) for i in range(0,len(df_rm.columns))]
    df_sc.columns = ['fft_scaled'+str(i) for i in range(0,len(df_sc.columns))]
    df_mfcc.columns = ['mfcc'+str(i) for i in range(0,len(df_mfcc.columns))]
    df_fbank.columns = ['fbank'+str(i) for i in range(0,len(df_fbank.columns))]
    df_ssc.columns = ['ssc'+str(i) for i in range(0,len(df_ssc.columns))]
    
    df_sig.columns = ['snd_wd_2000_mean_th005','snd_wd_2000_sd_th005','snd_wd_2000_max_th005'
                      ,'snd_wd_2000_len_th005'
                     ,'snd_wd_2000_mean_th025','snd_wd_2000_sd_th025','snd_wd_2000_max_th025'
                      ,'snd_wd_2000_len_th025'
                     ,'snd_wd_10000_mean_th005','snd_wd_10000_sd_th005','snd_wd_10000_max_th005'
                      ,'snd_wd_10000_len_th005'
                     ,'snd_wd_10000_mean_th025','snd_wd_10000_sd_th025','snd_wd_10000_max_th025'
                      ,'snd_wd_10000_len_th025'

                     ,'sln_wd_2000_mean_th001','sln_wd_2000_sd_th001','sln_wd_2000_max_th001'
                      ,'sln_wd_2000_len_th001'
                     ,'sln_wd_2000_mean_th005','sln_wd_2000_sd_th005','sln_wd_2000_max_th005'
                      ,'sln_wd_2000_len_th005'
                     ,'sln_wd_10000_mean_th001','sln_wd_10000_sd_th001','sln_wd_10000_max_th001'
                      ,'sln_wd_10000_len_th001'
                     ,'sln_wd_10000_mean_th005','sln_wd_10000_sd_th005','sln_wd_10000_max_th005'
                      ,'sln_wd_10000_len_th005'

                     , 'time_to_attack_2000', 'time_to_relax_2000'
                     , 'time_to_attack_10000', 'time_to_relax_10000'
                     , 'zerocross_tot','zerocross_prop'
                     ]
    df_peaks.columns = ['f_sc','n10','n20','n50','n90','n99','lead_min'
                        ,'high_freq_sum_50','high_freq_sum_90'
                       ,'w10_mean','w10_sd','w10_max','w10_len'
                       ,'w20_mean','w20_sd','w20_max','w20_len'
                       ,'w50_mean','w50_sd','w50_max','w50_len'
                       ,'w90_mean','w90_sd','w90_max','w90_len'
                       ,'w99_mean','w99_sd','w99_max','w99_len'
                       ,'coef_deg1','coef_deg2','coef_deg0'
                       ,'S','S_n','S2','S2_n'
                       ,'coef_deg1_w','coef_deg2_w','coef_deg0_w']    
        
    return df_peaks, df_sig, df_mfcc, df_rm, df_m, df_sc, df_fbank, df_ssc, le, labels_processed, filenames_processed

get_data = 0

if get_data == 1:
    dataframe_list_training = get_training_dataset()
    df_peaks, df_sig, df_mfcc, df_rm, df_m, df_sc, df_fbank, df_ssc, le, labels, files = dataframe_list_training
    df_result = pd.concat([df_peaks, df_sig, df_mfcc], axis=1, ignore_index=True)
    df_result.columns = list(df_peaks.columns) + list(df_sig.columns) + list(df_mfcc.columns)

    Xall = np.array(df_result)
    Xall = Xall.reshape(df_result.shape)
    yall = np.array(labels)
    Xall, yall = shuffle(Xall, yall, random_state=0)

    df_result.to_csv('.../df_result_ens.csv')
    np.save('.../Xalle_ens.npy', Xall)
    np.save('.../yall_ens.npy', yall)
    np.save('.../le_ens.npy', le)

get_data = 0
if get_data == 1:
    dataframe_list_training = get_training_dataset(False,'.../audio_test/')
    df_peaks_scoring, df_sig_scoring, df_mfcc_scoring, df_rm_scoring, df_m_scoring, df_sc_scoring, _, _, _, _, files = dataframe_list_scoring
    df_result_scoring = pd.concat([df_peaks_scoring, df_sig_scoring, df_mfcc_scoring], axis=1, ignore_index=True)
    
    X_scoring = np.array(df_result_scoring)
    X_scoring = X_scoring.reshape(df_result_scoring.shape)
    
    np.save('.../X_scoring_ens.npy', X_scoring)
    
    score_filelist = [str(x) for x in os.listdir('.../audio_test/')]
    np.save('.../score_filelist.npy',np.array(score_filelist))

Xall = np.load('../input/ensemble/Xall_ens.npy')
yall = np.load('../input/ensemble/yall_ens.npy')
X_scoring = np.load('../input/ensemble/X_scoring_ens.npy')
le = np.load('../input/ensemble/le_ens.npy')
score_filelist = np.load('../input/score-filelist/score_filelist.npy')
X_eda = Xall[0:2000,:]
y_eda = yall[0:2000]
X_eda = scale(X_eda)
mds = manifold.MDS(2, max_iter=100, n_init=1)
Y = mds.fit_transform(X_eda)
plt.figure(1)
plt.figure(figsize=(5,5))
for (j,cl) in enumerate([29,38,12,1,9,35]):
    cname = le.tolist().inverse_transform([cl])[0]
    plt.subplot(3,2,j+1)
    plt.scatter(Y[y_eda!=cl][:,0], Y[y_eda!=cl][:,1], c='blue', alpha=0.75)
    plt.scatter(Y[y_eda==cl][:,0], Y[y_eda==cl][:,1], c='red', alpha=0.75)
    plt.title(cname+' MDS')
    plt.axis('tight')
plt.tight_layout() 
plt.show()
   

for (i,prplx) in enumerate([5,20,100]):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity = prplx)
    Y = tsne.fit_transform(X_eda)
    plt.figure(1)
    plt.figure(figsize=(5,5))
    for (j,cl) in enumerate([29,38,12,1,9,35]):
        cname = le.tolist().inverse_transform([cl])[0]
        plt.subplot(3,2,j+1)
        plt.scatter(Y[y_eda!=cl][:,0], Y[y_eda!=cl][:,1], c='blue', alpha=0.75)
        plt.scatter(Y[y_eda==cl][:,0], Y[y_eda==cl][:,1], c='red', alpha=0.75)
        plt.title(cname+' t-SNE perplexity: '+str(prplx))
        plt.axis('tight')
    plt.tight_layout() 
    plt.show()
   
for (i,nneigh) in enumerate([20,50,100]):
    se = manifold.SpectralEmbedding(n_components=2, n_neighbors=nneigh)
    Y = se.fit_transform(X_eda)
    plt.figure(1)
    plt.figure(figsize=(10,10))
    for (j,cl) in enumerate([29,38,12,1,9,35]):
        cname = le.tolist().inverse_transform([cl])[0]
        plt.subplot(3,2,j+1)
        plt.scatter(Y[y_eda!=cl][:,0], Y[y_eda!=cl][:,1], c='blue', alpha=0.75)
        plt.scatter(Y[y_eda==cl][:,0], Y[y_eda==cl][:,1], c='red', alpha=0.75)
        plt.title(cname+' Spect Embed nniegh: '+str(nneigh))
        plt.axis('tight')
    plt.tight_layout() 
    plt.show()
   
def get_proba(clf,X,y,Xtest):
    clf.fit(X, y)
    pred_clf_proba = clf.predict_proba(Xtest)
    pred_clf_classes = [list(clf.classes_[np.argsort(x).tolist()[::-1]]) for x in pred_clf_proba] 
    return pred_clf_proba, pred_clf_classes

clf = RandomForestClassifier(n_estimators=10, class_weight = 'balanced', random_state = 7)
pred_rf_proba, pred_rf_classes = get_proba(clf, Xall, yall, X_scoring)

clf = xgb.XGBClassifier(n_estimators=10, learning_rate=0.05, max_depth=2)
pred_xgb_proba, pred_xgb_classes = get_proba(clf, Xall, yall, X_scoring)

pred_ens_proba = (pred_rf_proba * pred_xgb_proba ) ** (1/2)
pred_ens_classes = [list(np.argsort(x).tolist()[::-1]) for x in pred_ens_proba] 

df_output = pd.DataFrame(pred_ens_classes)
df_output = list(df_output.apply(lambda x: list(le.tolist().inverse_transform(x[0:3])), axis=1))
df_output = pd.DataFrame(df_output)
df_output['fname'] = score_filelist 
df_output['label'] = df_output.apply(lambda x: str(x[0]) + ' ' + str(x[1]) +' '+ str(x[2]) , axis=1)

np.save('pred_ens_proba_simple_ens_newfeatures_libr.npy', pred_ens_proba)
df_output[['fname','label']].to_csv('pred_classes_rfxgb_ens.csv', index = False) # gives 0.844 on LB if trained on 1000 trees for both rf and xgb


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Model, Sequential # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Activation, MaxPool2D
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.models import save_model, load_model
from keras.callbacks import Callback
from keras import losses, models, optimizers
from keras import backend as K
from keras.models import load_model
import h5py as h5py
from keras.callbacks import ModelCheckpoint

class mapk_callback(Callback):
    def __init__(self,training_data,validation_data):
        
        self.x_trn = training_data[0]
        self.y_trn = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]        
    
    def on_train_begin(self, logs={}):
        return
    def on_train_end(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):        
        y_pred_trn = self.model.predict(self.x_trn)
        y_pred_top3_classes_trn = [np.argsort(x).tolist()[::-1][0:3] for x in y_pred_trn]
        obs_y_trn = [[x] for x in self.y_trn.tolist()]
        mapk_score_trn = mapk(obs_y_trn, y_pred_top3_classes_trn,k=3)

        y_pred_val = self.model.predict(self.x_val)
        y_pred_top3_classes_val = [np.argsort(x).tolist()[::-1][0:3] for x in y_pred_val]
        obs_y_val = [[x] for x in self.y_val.tolist()]
        mapk_score_val = mapk(obs_y_val, y_pred_top3_classes_val,k=3)
        
        print('\rmapk: %s - mapk_val: %s' % (str(round(mapk_score_trn,4)),str(round(mapk_score_val,4))),end=100*' '+'\n')
        return
    
    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return   

load = 0
if load == 1:
    Xall = np.load('.../Xall_cnn_libr.npy')
    yall = np.load('.../yall_cnn_libr.npy')
    le = np.load('.../le_cnn_libr.npy')
    X_scoring = np.load('.../X_scoring_cnn_libr.npy')


# (9)

def get_compiled_cnn_model():
    
    inp = Input(shape=(173, 40, 1))

    lr = Convolution2D(32, (11,5), padding="same")(inp)
    lr = BatchNormalization()(lr)
    lr = Activation("relu")(lr)
    lr = MaxPool2D()(lr)
    lr = Dropout(0.1)(lr)

    lr = Convolution2D(64, (7,5), padding="same")(lr)
    lr = BatchNormalization()(lr)
    lr = Activation("relu")(lr)
    lr = MaxPool2D()(lr)
    lr = Dropout(0.1)(lr)

    lr = Convolution2D(64, (7,5), padding="same")(lr)
    lr = BatchNormalization()(lr)
    lr = Activation("relu")(lr)
    lr = MaxPool2D()(lr)
    lr = Dropout(0.1)(lr)

    lr = Convolution2D(128, (5,3), padding="same")(lr)
    lr = BatchNormalization()(lr)
    lr = Activation("relu")(lr)
    lr = MaxPool2D()(lr)
    lr = Dropout(0.1)(lr)

    lr = Flatten()(lr)
    lr = Dense(128)(lr)
    lr = BatchNormalization()(lr)
    lr = Activation("relu")(lr)
    out = Dense(num_classes, activation='softmax')(lr)

    model = Model(inputs=inp, outputs=out)
    
    model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy'],
              )
    
    return model



num_classes = np.unique(yall).shape[0]
pred_proba_cnn_strat = np.ones(shape=(X_scoring.shape[0],num_classes))

my_n_splits = 5
strat_split = StratifiedShuffleSplit(n_splits=my_n_splits, test_size=0.1, random_state=2)
i = 0
run_training = 0

if run_training == 1:
    for ii, (train_index, val_index) in enumerate(strat_split.split(Xall, yall)):

        # K.clear_session()
        print()
        print('-------------------------- Strata',ii,' --------------------------')
        print()

        X_train = Xall[train_index]
        y_train = yall[train_index]
        X_val = Xall[val_index]
        y_val = yall[val_index]

        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_train = (X_train - mean)/std
        X_val = (X_val - mean)/std
        X_scoring_iter = (X_scoring - mean)/std

        y_train_1dim = y_train
        y_train = np_utils.to_categorical(y_train, num_classes)

        y_val_1dim = y_val
        y_val = np_utils.to_categorical(y_val, num_classes)

        cnn_model = get_compiled_cnn_model()

        callbacks = [mapk_callback(training_data=(X_train,y_train_1dim),validation_data=(X_val, y_val_1dim))
                     , ModelCheckpoint('.../weights_valid.model',
                                        monitor='val_acc',
                                        verbose=1,
                                        save_best_only=True,
                                        save_weights_only=True)
        ]

        history = cnn_model.fit(X_train, y_train
                                , validation_data=(X_val, y_val)
                                , callbacks = callbacks
                                , verbose=1
                                , batch_size = 64
                                , epochs = 1)

        pred_iter = cnn_model.predict(X_scoring_iter)
        np.save('.../pred_cnn_proba_strat_%d.npy'%ii, pred_iter)
        pred_proba_cnn_strat = pred_proba_cnn_strat * pred_iter

    df_output = pd.DataFrame(pred_cnn_classes)
    df_output = list(df_output.apply(lambda x: list(le.tolist().inverse_transform(x[0:3])), axis=1))
    df_output = pd.DataFrame(df_output)
    df_output['fname'] = score_filelist 
    df_output['label'] = df_output.apply(lambda x: str(x[0]) + ' ' + str(x[1]) +' '+ str(x[2]) , axis=1)

    np.save('.../predictions_cnn_final.npy', pred_cnn_proba)
    df_output[['fname','label']].to_csv('.../predictions_cnn_strat.csv', index = False)

        
load_data = 0

if load_data == 1:
    Xall = np.load('.../Xall_cnn_libr.npy')
    yall = np.load('.../yall_cnn_libr.npy')
    le = np.load('.../le_cnn_libr.npy')
    X_scoring = np.load('.../X_scoring_cnn_libr.npy')

    Zall = np.load('.../Xall_ens.npy')
    Z_scoring = np.load('.../X_ens.npy')

def get_compiled_CNN_model():
    inp1 = Input(shape=(173,40,1))

    lr = Convolution2D(32, (10,4), padding="same")(inp1)
    lr = BatchNormalization()(lr)
    lr = Activation("relu")(lr)
    lr = MaxPool2D()(lr)
    lr = Dropout(0.1)(lr)

    lr = Convolution2D(32, (10,4), padding="same")(lr)
    lr = BatchNormalization()(lr)
    lr = Activation("relu")(lr)
    lr = MaxPool2D()(lr)
    lr = Dropout(0.1)(lr)

    lr = Convolution2D(32, (10,4), padding="same")(lr)
    lr = BatchNormalization()(lr)
    lr = Activation("relu")(lr)
    lr = MaxPool2D()(lr)
    lr = Dropout(0.1)(lr)

    lr = Convolution2D(32, (10,4), padding="same")(lr)
    lr = BatchNormalization()(lr)
    lr = Activation("relu")(lr)
    lr = MaxPool2D()(lr)
    lr = Dropout(0.1)(lr)

    flat = Flatten()(lr)

    inp2 = Input(shape = (Z_train.shape[1], ))
    den2 = Dense(64)(inp2)
    concatFeatures = Concatenate(axis = -1)([flat, den2])

    lr = Dense(256)(concatFeatures)
    lr = BatchNormalization()(lr)
    lr = Activation("relu")(lr)
    out = Dense(num_classes, activation='softmax')(lr)

    model = Model(inputs=[inp1,inp2], outputs=out)
    
    model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
                  optimizer=optimizers.Adam(lr=0.001),
                  metrics=['accuracy'],
                  )    
    
    return model


