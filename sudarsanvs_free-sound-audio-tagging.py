import os
os.getcwd()
!apt update
!apt-get install -y libsndfile1
!pip install SoundFile

import matplotlib.pyplot as plt
import IPython.display as ipd
import IPython.display as ipd
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.signal import find_peaks
from scipy.fftpack import dct
from scipy.io import wavfile
from skimage import util
import scipy.signal as signal
from pandas import read_csv
import seaborn as sns
import scipy
import pandas as pd
import numpy as np
import pylab
import librosa
import librosa.display
import wave
import struct
import sklearn
plt.style.use('classic')
%matplotlib inline
# Set default font size
#plt.rcParams['font.size'] = 20
#pd.set_option('display.max_columns',60)
#plt.style.available
def printsig():
    print('fs_rate:',fs_rate, 'Channels:', audio_chnl, 'duration:',duration,'sec.', 'Ts:', Ts, 'time:',time,'time_len:',time_len)
    # Plot original Audio signal
    plt.figure(figsize=(10, 3))
    plt.plot(time, sig_o)
    plt.title('Origin wave signal')
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.show() 
    plt.figure(figsize=(10, 3))
    plt.plot(time, sig_s)
    plt.title('windowed wave signal')
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.show() 
def print_mfcc():
    # Plot mfcc - the amplitude envelope of a waveform.
    plt.figure(figsize=(10, 3)) 
    librosa.display.waveplot(mfcc, sr=fs_rate)
    plt.title('mfcc envelope')
    plt.ylabel('Envelope Amplitude')
    plt.show()    
    # Plot mfcc spectrum.
    plt.figure(figsize=(10, 3))
    plt.title('mfcc spectogram')
    mfcc_spec=librosa.feature.mfcc(y=sig_s, sr=fs_rate)
    mfcc_spec=sklearn.preprocessing.scale(mfcc_spec, axis=1)
    librosa.display.specshow(mfcc_spec, sr=fs_rate, x_axis='time') 
    plt.show()     
def print_fft(sigfft,sigfreqs):
    # plot fft of signal
    plt.figure(figsize=(10, 3))
    plt.plot(sigfreqs,sigfft) 
    plt.title('FFT of signal')
    plt.ylabel('Power')
    plt.xlabel('Frequency')
    plt.show() 

    plt.figure(figsize=(10, 3))
    l=int(len(sigfft)/2)
    plt.plot(sigfreqs[1:l],sigfft[1:l]) 
    plt.title('positive FFT of signal')
    plt.ylabel('Power')
    plt.xlabel('Frequency')
    plt.show()  
def print_spectogram():
    frqs, times, xsig = signal.spectrogram(sig_s, fs_rate,  nperseg=1024,  # window='hamming',
                                       noverlap = 1024-256,detrend=False, scaling='spectrum')
    plt.figure(figsize=(10,5));
    plt.pcolormesh(times, frqs/1000, 20*np.log10(xsig), cmap='viridis' ) #viridis  magma  
    plt.title('Signal Spectrogram')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [kHz]');
    plt.show()  
def find_N(N_len):
    return 2**14 if N_len>=2**14 else N_len
def process_fft():
    #entire signal 
    N=find_N(time_len)
    #N=time_len
    fftsig= np.fft.fft(sig,N)
    sigfft=np.abs(fftsig)
    sigfreqs = np.fft.fftfreq(N,Ts)

    # slice signal to small frames
    frame_len = 1024 # about 20msec
    frame_time = Ts
    frames = util.view_as_windows(sig, window_shape=(frame_len,), step=512) #step=128)
    win = np.hanning(frame_len+1)[:-1]   # to try with hamming as well
    frames = frames*win
    frames = frames.T
    num_of_frames=frames.shape[0]
    #print(f'Signal shape: {sig.shape}, framed signal shape: {frames.shape[1]}')
    spects= np.fft.fft(frames, n=frame_len, axis=0)
    spects=np.abs(spects)
    #max_pwr = spects.max()
    frame_f = np.fft.fftfreq(frame_len,frame_time) #scipy.fftpack.fftfreq(frame_len,frame_time)
    pos=int(frame_len/2)
    max_val = np.amax(spects) # find fft max value    
            
    for x in range(frames.shape[1]):      
        idxs = np.argmax(spects[:pos,x])   # index of max peak in frame :pos
        if spects[idxs,x] > (0.5 * max_val): # 0.65
            sig_freqs.append(frame_f[idxs])  # create a list of frequencies in each frame        

#         if fname=='962ffc01.wav':#3315ec7f.wav 0006ae4e 6459fc05 
#             print('column', x)
#             print(f'maximum value in column {x} {np.amax(spects[:pos,x])}')
#             print(f'the idxs {idxs} with the value  of {spects[idxs,x]}') 
    
    # prints and plots a single wave file
    if fname=='074a72f0.wav':  # 6459fc05 / 3315ec7f / 962ffc01.wav  / 54073d7e / d41f1ae5.wav
        print('File: 074a72f0.wav, Labled: Bicycle bell ')  
        printsig()
        print_mfcc()
        print_fft(sigfft,sigfreqs)
        print_spectogram()
        #print('max val',max_val)
        print(f'sig_freqs: {sig_freqs}')
        
    static_feats=[np.mean(sig_freqs), np.median(sig_freqs),np.std(sig_freqs),np.var(sig_freqs),
                  min(sig_freqs), max(sig_freqs)]

    return static_feats

classes = ['Bark','Bass_drum','Bicycle_bell','Bus','Car_passing_by','Male_speech_and_man_speaking',
           'Male_speech_and_man_speaking,Male_singing','Female_speech_and_woman_speaking','Female_speech_and_woman_speaking,Whispering']

# Reading wave files name
df_train_curated = pd.read_csv("../input/freesound-audio-tagging-2019/train_curated.csv")  
# Creating file names df for selected categories
df_train_curated = df_train_curated.loc[df_train_curated['labels'].isin(classes)]
df_files = df_train_curated.copy()
df_files.sort_values('labels', ascending=True, inplace=True);
df_files = df_files.reset_index(drop=True)
print(f'From dataset: {len(df_files)} Sound files')
df_files.head()
df_files
# combine Male / Female voices
classes = classes[:-2]
classes[-2]='Male_voice'
classes[-1]='Female_voice'
male = dict.fromkeys(['Male_speech_and_man_speaking','Male_speech_and_man_speaking,Male_singing'], 'Male_voice')    
female = dict.fromkeys(['Female_speech_and_woman_speaking','Female_speech_and_woman_speaking,Whispering'],'Female_voice')  
df_files = df_files.replace(male)
df_files = df_files.replace(female)
classes
#find the number of files for each class
for cls in classes: 
    files_len=len(df_files[df_files.labels==cls])  # The number of files 
    print(cls, files_len, 'files')
# New features dataframe
features_df = pd.DataFrame(columns=['file_name', 'avrg_freq','med_freq','std_Freq','var_freq',
                                    'min_freq', 'max_freq',
                                    'mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8',
                                    'mfcc9','mfcc10','mfcc11','mfcc12','class',]) 

for cls in classes: 
    fnames = df_files[df_files['labels']==cls].fname
    
    for fname in fnames: #i in range(len(fnames)-1): 
            # Reading wav files
            fs_rate, sig = scipy.io.wavfile.read(fname)
            duration = len(sig) / fs_rate
            audio_chnl = len(sig.shape)    # number of Channels
            Ts = 1.0 / fs_rate            # Timestep between samples
            time=np.linspace(0, duration, num = len(sig), endpoint=True)
            time_len = len(time)

            fltr_coef = 0.97 # filter coefficiant typical value
            sig_o = np.append(sig[0], sig[1:] - fltr_coef * sig[:-1])  #  floating point

            sig_freqs = []
            static_feats = []

            # windowing the signal
            wind = np.hanning(len(sig)+1)[:-1]  
            sig_s  = sig_o * wind

            # mel-frequency ceptral coefficiens
            mfcc = np.mean(librosa.feature.mfcc(y=sig_s, sr=fs_rate, n_mfcc=12).T,axis=0)

            static_feats = process_fft()   # wave processing in the frequency domain

            features_df.loc[len(features_df)] = [fname]+list(static_feats)+list(mfcc)+[cls]   
features_df=features_df[features_df['max_freq']>0]    

map_std=features_df[features_df['std_Freq']>0].groupby('class')['std_Freq'].mean()
map_var=features_df[features_df['var_freq']>0].groupby('class')['var_freq'].mean()

features_df.loc[features_df['std_Freq']==0,'std_Freq'] = features_df[features_df['std_Freq']==0]\
                                                         .apply(lambda df_row: map_std.loc[df_row['class']], axis=1)

features_df.loc[features_df['var_freq']==0,'var_freq'] = features_df[features_df['var_freq']==0]\
                                                         .apply(lambda r: map_var.loc[r['class']], axis=1)

features_df.loc[features_df['min_freq']==0,'min_freq'] = features_df[features_df['min_freq']==0]\
                                                         .apply(lambda r: map_var.loc[r['class']], axis=1)

features_df.loc[features_df['med_freq']==0,'med_freq'] = features_df[features_df['med_freq']==0]\
                                                         .apply(lambda r: map_var.loc[r['class']], axis=1)

# Save all features in csv file
features_df.to_csv('features.csv', encoding='utf-8') #  (index=False)

b_s = '\033[1m'
b_e = '\033[0m'
print(b_s,'\033[34m ......{} files were recorded in features.csv'.format(len(features_df.file_name)),b_e)
ipd.Audio('bfa6c58b.wav')     
features_df.describe()
df_dumm = pd.get_dummies(features_df[features_df.columns[2:]])  
df_dumm.head()
plt.figure(figsize=(16,10))
corr=df_dumm.corr()
sns.heatmap(corr[(corr >= 0.1) | (corr <= -0.1)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);
plt.show()
plt.figure(figsize=(10,5))
ax = sns.kdeplot(df_dumm.max_freq[(df_dumm['class_Bass_drum'] == 1) ],
                color="blue", shade = True)
ax = sns.kdeplot(df_dumm.max_freq[(df_dumm['class_Bicycle_bell'] == 1) ],
                ax =ax, color="red", shade= True)
ax = sns.kdeplot(df_dumm.max_freq[(df_dumm['class_Bark'] == 1) ],
                ax =ax, color="green", shade= True)
ax = sns.kdeplot(df_dumm.max_freq[(df_dumm['class_Bus'] == 1) ],
                ax =ax, color="yellow", shade= True)
ax = sns.kdeplot(df_dumm.max_freq[(df_dumm['class_Male_voice'] == 1) ],
                ax =ax, color="black", shade= True)
ax = sns.kdeplot(df_dumm.max_freq[(df_dumm['class_Female_voice'] == 1) ],
                ax =ax, color="orange", shade= True)
ax = sns.kdeplot(df_dumm.max_freq[(df_dumm['class_Car_passing_by'] == 1) ],
                ax =ax, color="grey", shade= True)
ax.legend(['Bass drum', 'Bicycle_bell', 'Bark','Bus','Male_voice','Female_voice','Car passing'], loc='best')
ax.set_ylabel('Density')
ax.set_xlabel('max Frequencies [Hz]')
ax.set_title('Distribution of class noises Vs. max Frequencies', size = 13);
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score,\
    GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, BaggingClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
np.random.seed(0)
def classification_results(y, y_pred, name='', add_rep=True):  # False
    acc = accuracy_score(y, y_pred)
                        
    cm = pd.DataFrame(confusion_matrix(y, y_pred), 
                      index=classes, 
                      columns=classes)

    print(name + ' accuracy: ', round(acc,4),'\n') # round(acc,4)
    print(cm,'\n')
    if (add_rep):
        print(classification_report(y, y_pred))
df = read_csv("features.csv")
train, test = train_test_split(df, test_size=0.3,  random_state=0)  #, random_state=0

X_train = train[train.columns[2:-1]]
y_train = train[train.columns[-1]]
X_test = test[test.columns[2:-1]]
y_test = test[test.columns[-1]]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer, RobustScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.model_selection import validation_curve
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, log_loss, precision_score 
scalers =np.array( [['Standard', StandardScaler()], 
                   ['MinMax', MinMaxScaler()], 
                   ['RobustScaler', RobustScaler()],
                   ['Normalizer' , Normalizer()],
                   ['QuantileTransformer', QuantileTransformer(n_quantiles=len(X_train))]])

plt.figure(figsize=(12, 6))
legend =[]
for [scaler_name, scaler_func] in scalers:
    for metric in ['manhattan', 'euclidean']:    #'hamming'
        scaler = scaler_func
        scaler.fit(X_train)
        X = scaler.transform(X_train) 
        y = y_train
        param='n_neighbors'
        param_range = range(2,15,1)
        train_scores, test_scores = validation_curve(KNeighborsClassifier(metric=metric),
                                                     X, y,                                        
                                                     param_name=param,
                                                     param_range=param_range,
                                                     scoring="accuracy",
                                                     cv=5) 
        
        test_scores_mean = np.mean(test_scores, axis=1)
        plt.plot(param_range, test_scores_mean);
        legend.append(scaler_name+'-'+metric)

    plt.title("Validation Curve Vs. KNN - on Train set")
    plt.xlabel(param)
    plt.ylabel("Score - Accuracy")
    plt.ylim(0.65, 0.85)      
    plt.legend(legend, loc='lower right')
plt.show()
knn = KNeighborsClassifier(n_neighbors=4, metric='manhattan')   #6,euclidean manhattan

# Scale Transform and normalization   
scaler = QuantileTransformer(n_quantiles=len(X_train)).fit(X_train)   # 

X = scaler.transform(X_train)  
y = y_train

knn.fit(X, y);
y_train_pred = knn.predict(X)
cm = confusion_matrix(y_true=y, y_pred=y_train_pred)

sns.set(font_scale=1.4)#for label size
sns.heatmap(cm, annot=True,annot_kws={"size": 14}, fmt='g')
print ('\nClassification_report on Test set\n',classification_report(y_true=y, y_pred=y_train_pred))
X_test_knn = scaler.transform(X_test) 
y_test_pred = knn.predict(X_test_knn)
cm = confusion_matrix(y_true=y_test, y_pred=y_test_pred)
print('\nConfusion matrix on Test set:\n')
sns.set(font_scale=1.4)#for label size
sns.heatmap(cm, annot=True,annot_kws={"size": 14}, fmt='g')
print ('\nClassification_report on Test set\n',classification_report(y_true=y_test, y_pred=y_test_pred))
