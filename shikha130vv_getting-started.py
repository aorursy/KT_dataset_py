#References
#https://www.groundai.com/project/environment-sound-classification-using-multiple-feature-channels-and-deep-convolutional-neural-networks/1

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os     
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
import os
import librosa
import librosa.display
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
#!pip install python_speech_features
%matplotlib inline
plt.style.use('ggplot')
import glob
import glob
import librosa
from librosa import feature
import numpy as np
from pathlib import Path
# Detect hardware, return appropriate distribution strategy
def get_strategy():
    gpu = ""
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())     
    except ValueError:
        tpu = None
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gpu = tf.config.list_physical_devices("GPU")
        if len(gpu) == 1:
            print('Running on GPU ', gpu)
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        GCS_PATH = KaggleDatasets().get_gcs_path('birdsong-recognition')
    elif len(gpu) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision":True})
        GCS_PATH = "/kaggle/input/birdsong-recognition/"
    else:
        strategy = tf.distribute.get_strategy()
        GCS_PATH = "/kaggle/input/birdsong-recognition/"

    print("REPLICAS: ", strategy.num_replicas_in_sync)
    base_dir = "../input/birdsong-recognition/"
    print(base_dir)
    return strategy, GCS_PATH, base_dir

strategy,GCS_PATH, base_dir = get_strategy()
sns.set_palette("pastel")
palette = sns.color_palette()
#Load sound file
def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds
#EDA
#from python_speech_features import mfcc
from scipy.signal.windows import hann
def load_sample_files():
    
    list_class_names = np.array(class_list)[np.random.randint(0, len(class_list), 2)]
    sample1 = train_data[train_data["ebird_code"] == list_class_names[0]].sample(2)
    sample2 = train_data[train_data["ebird_code"] == list_class_names[1]].sample(2)

    sound_file_paths1 = [base_dir + "train_audio/" + list_class_names[0] + "/" + file for file in sample1["filename"].values]
    sound_file_paths2 = [base_dir + "train_audio/" + list_class_names[1] + "/" + file for file in sample2["filename"].values]
    sound_file_paths = sound_file_paths1 + sound_file_paths2
    sound_names = list(sample1["ebird_code"].values) + list(sample2["ebird_code"].values)
    raw_sounds = load_sound_files(sound_file_paths)
    return sound_names, raw_sounds

def plot_waves(sound_names,raw_sounds, plot_type):
    i = 1
    max_row = 2
    max_col = 2
    fig, ax = plt.subplots(max_row, max_col, figsize=(20,8))
    row,col = 0,  0
    c = palette[3]
    
    
    n_mfcc = 13
    n_mels = 40
    n_fft = 512 
    hop_length = 160
    fmin = 0
    fmax = None
    sr = 22050


    for n,f in zip(sound_names,raw_sounds):
        if plot_type == "mfcc":
            mfcc_librosa = librosa.feature.mfcc(y=f, sr=sr, n_fft=n_fft,
                                        n_mfcc=n_mfcc, n_mels=n_mels,
                                        hop_length=hop_length,
                                        fmin=fmin, fmax=fmax, htk=False)

        #mfcc_speech = mfcc(signal=f, samplerate=sr, winlen=n_fft / sr, winstep=hop_length / sr,
        #                                   numcep=n_mfcc, nfilt=n_mels, nfft=n_fft, lowfreq=fmin, highfreq=fmax,
        #                                  preemph=0.0, ceplifter=0, appendEnergy=False, winfunc=hann)
        
        #sns.heatmap(mfcc_librosa, ax=ax[row,col])
            ax[row,col].plot(mfcc_librosa.T)
        else:
            librosa.display.waveplot(f,sr=22050, ax=ax[row,col], color=c)
        ax[row,col].set_title(n)
        col = col + 1
        if col == max_col:
            col = 0
            row = row + 1
            c = palette[0]
    if plot_type == "mfcc":
        plt.suptitle('Figure 1: Waveplot',x=0.5, y=0.915,fontsize=18)
    else:
        plt.suptitle('Figure 1: MFCC',x=0.5, y=0.915,fontsize=18)
    plt.show()


    
def group_n_plot():
    group_data = train_data.groupby("ebird_code").agg(num_audio=("filename","count"), tot_audio_length=("duration","sum"), median_audio_length=("duration","median"))
    group_data = group_data.reset_index().reset_index()
    
    fig, ax= plt.subplots(1,3,figsize=(20,4))
    sns.lineplot(x="index", y="num_audio", data=group_data, ax=ax[0], color=palette[0])
    sns.lineplot(x="index", y="median_audio_length", data=group_data, ax=ax[1], color=palette[1])
    sns.lineplot(x="index", y="tot_audio_length", data=group_data, ax=ax[2], color=palette[2])
    ax[0].set_title("No Of Audios");
    ax[1].set_title("Median Audio Length");
    ax[2].set_title("Total Audio Length");
    return group_data


#Unused EDA MEthods
def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(len(sound_names),1,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60))
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()
#Data Cleansing
def drop_excess_files(train_data, group_data):
    median_audio_length = train_data["duration"].median()
    print("Median Audio Length",median_audio_length)
    train_data["deviation_from_median"] = np.abs(train_data["duration"] - median_audio_length)
    train_data.sort_values(["ebird_code","deviation_from_median"], inplace=True)
    train_data["cum_duration"] = train_data.groupby("ebird_code")["duration"].cumsum()
    train_data["cum_duration"] = train_data["cum_duration"] - train_data["duration"]
    median_duration = group_data["tot_audio_length"].median()
    train_data[["ebird_code","duration","deviation_from_median","cum_duration"]].iloc[-100:].head(5)
    train_data = train_data[train_data["cum_duration"] <= median_duration]
    return train_data
#Feature Engineering - Not using this code as of now
fn_list_i = [
     feature.spectral_centroid,
     feature.spectral_bandwidth,
     feature.spectral_rolloff,
     feature.melspectrogram,
     feature.spectral_contrast
]
    
fn_list_ii = [
     feature.rms,
     feature.zero_crossing_rate
]

def parse_audio_files(file_name):
    y, sr = librosa.load(file_name)
    feat_vect_i = [ np.mean(funct(y,sr).T, axis=0) for funct in fn_list_i]
    feat_vect_ii = [ np.mean(funct(y).T, axis=0) for funct in fn_list_ii] 
    stft = np.abs(librosa.stft(y))
    chroma = [ np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0) ]
    mfccs = [ np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0) ]
    tonnetz = [ np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y),sr=sr).T, axis=0) ]
    features = feat_vect_i + feat_vect_ii + chroma + mfccs + tonnetz
    return features

#Feature Engineering

def extract_feature1(file_name, X=None, sample_rate=0):
    global global_X
    global global_sr
    if X is None:
        X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft,sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files1(file_name):
    try:
        #print(file_name)
        features = np.empty((0,193))
        mfccs, chroma, mel, contrast,tonnetz = extract_feature1(file_name)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
        return features[0]
    except:
        return None
    
def parse_audio_files2(X, sample_rate):
    try:
        #print(file_name)
        features = np.empty((0,193))
        mfccs, chroma, mel, contrast,tonnetz = extract_feature1("",X, sample_rate)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
        return features[0]
    except:
        return None

num_features = 0
def process_class_name(class_name):
    global num_features
    #print(class_name)
    feature_file_name = class_name + "_features.csv"
    full_feature_file_name = "/kaggle/input/" + "birdcall/" + feature_file_name
    if os.path.exists(full_feature_file_name):
        #print("File found!", full_feature_file_name)
        features_df = pd.read_csv(full_feature_file_name)
        num_features = len(features_df.columns.values)
    else:
        print("Not found:", full_feature_file_name)
        if 1==2:
            df = pd.DataFrame({"file_name":os.listdir(base_dir + "train_audio/" + class_name)})
            df["class_name"] = class_name
            df["file_name"] = df.apply(lambda row: base_dir + "train_audio/" + row["class_name"] + "/" + row["file_name"], axis=1)
            df["features"] = df["file_name"].map(lambda x: parse_audio_files1(x))
            num_features= len(df.head(1)["features"].values[0])
            features_df = df[["features"]]
            for i in range(num_features):
                features_df["features_" + str(i)] = features_df["features"].map(lambda x: x[i] if x else None)
            features = features_df.pop("features")
        return None
    #features_df.to_csv(feature_file_name, index=False)
    features_df["class_name"] = class_name
    features_df["file_name"] = os.listdir(base_dir + "train_audio/" + class_name)
    features_df["file_name"] = features_df.apply(lambda row: base_dir + "train_audio/" + row["class_name"] + "/" + row["file_name"], axis=1)

    return features_df
import cv2
num_train_data_per_class = 1
n_fft1 = int(0.0025 * 22050)
hop_length1 = int(0.001 * 22050)

n_fft2 = int(0.005 * 22050)
hop_length2 = int(0.0025 * 22050)

n_fft3 = int(0.01 * 22050)
hop_length3 = int(0.005 * 22050)
n_mels = 128
fmin = 20
fmax = 8000

def load_test_clip(path, start_time, duration=5):
    #if os.path.exists(TEST_FOLDER):
    return librosa.load(path, offset=start_time, duration=duration)

def get_audio_length(path):
    data, sr = librosa.load(path)
    return len(data),sr
    
    

def process_class_name_for_spectrogram(ebird_code, train_data):
    int_ebird_code = dic_ebird_code[ebird_code]
    df = train_data[train_data["int_ebird_code"] == int_ebird_code][["ebird_code", "filename", "duration", "channels"]]
    num_files = df.shape[0]
    for i in range(num_train_data_per_class):
        int_file = np.random.randint(0,num_files, (1))
        row = df.iloc[int_file]
        print(row)
        filename =row["filename"].values[0]
        duration = row["duration"].values[0]
        print(duration)
        filepath = base_dir + "train_audio/" + ebird_code + "/" + filename
        if duration == 5:
            start_time=0
        else:
            start_time = np.random.randint(0,int(duration)-5, (1))
        clip, sr = load_test_clip(filepath, start_time)
        print(sr)
        mel_spec1 = librosa.feature.melspectrogram(clip, n_fft=n_fft1, hop_length=hop_length1, n_mels=n_mels, sr=sr, power=1.0, fmin=fmin, fmax=fmax)
        mel_spec_db1 = librosa.amplitude_to_db(mel_spec1, ref=np.max)
        print(mel_spec_db1.shape)
        
        mel_spec2 = librosa.feature.melspectrogram(clip, n_fft=n_fft2, hop_length=hop_length2, n_mels=n_mels, sr=sr, power=1.0, fmin=fmin, fmax=fmax)
        mel_spec_db2 = librosa.amplitude_to_db(mel_spec2, ref=np.max)
        print(mel_spec_db2.shape)
        
        mel_spec3 = librosa.feature.melspectrogram(clip, n_fft=n_fft3, hop_length=hop_length3, n_mels=n_mels, sr=sr, power=1.0, fmin=fmin, fmax=fmax)
        mel_spec_db3 = librosa.amplitude_to_db(mel_spec3, ref=np.max)
        print(mel_spec_db3.shape)
        
        mel_spec1 = cv2.resize(mel_spec1, (224, 224))
        mel_spec2 = cv2.resize(mel_spec2, (224, 224))
        mel_spec3 = cv2.resize(mel_spec3, (224, 224))
        mel_spec1 = mel_spec1*255/mel_spec1.max()
        mel_spec2 = mel_spec2*255/mel_spec2.max()
        mel_spec3 = mel_spec3*255/mel_spec3.max()
       
        mel_spec = np.stack([mel_spec1, mel_spec2, mel_spec3], axis=-1)
        print(mel_spec.shape)
    
    return df, mel_spec, mel_spec1, mel_spec2, mel_spec3

train_data = pd.read_csv(base_dir + "train.csv")

ebird_code_list = train_data["ebird_code"].unique()
dic_ebird_code = {k:v for v,k in enumerate(ebird_code_list)}
train_data["int_ebird_code"] = train_data["ebird_code"].map(dic_ebird_code)

df, mel_spec, mel_spec1, mel_spec2, mel_spec3 = process_class_name_for_spectrogram("amecro", train_data)

plt.imshow(mel_spec)
def get_model(num_features, n_classes):
    keras.backend.clear_session()
    l1 = keras.layers.Input(shape=(num_features,), name="feature")
    l2 = keras.layers.Dense(2048, activation="tanh")(l1)
    l3 = keras.layers.Dense(1024, activation="tanh")(l2)
    l4 = keras.layers.Dense(512, activation="tanh")(l3)
    l5 = keras.layers.Dense(n_classes, activation = "sigmoid")(l4)
    model = keras.models.Model(inputs={"feature":l1},outputs=l5)
    return model


class_list = os.listdir(base_dir+"train_audio/")
train_data = pd.read_csv(base_dir + "train.csv")
train_data.head(1)
group_data = group_n_plot()
train_data = drop_excess_files(train_data, group_data)
group_n_plot()
sound_names, raw_sounds =   load_sample_files()  
plot_waves(sound_names,raw_sounds, "")
plot_waves(sound_names,raw_sounds, "mfcc")
with strategy.scope():
    arr_df = []
    from multiprocessing import Pool
    p = Pool(2)
    arr_df = p.map(process_class_name, class_list)
    p.close()
    p.join() 
arr_df_new = []
for df in arr_df:
    if df is not None:
        arr_df_new.append(df)
all_train_data = pd.concat(arr_df_new)
all_train_data.head(1)
class_list = list(all_train_data["class_name"].unique())
n_classes = all_train_data["class_name"].unique().shape[0]
dic_class_name = {k:v for v,k in enumerate(class_list)}
dic_class_name_rev = {v:k for v,k in enumerate(class_list)}
all_train_data["label"] = all_train_data["class_name"].map(dic_class_name)
label_data = all_train_data.pop("label")
label_data = tf.keras.utils.to_categorical(label_data, n_classes)

class_name = all_train_data.pop("class_name")
file_name = all_train_data.pop("file_name")

all_train_data = all_train_data.astype(np.float32)
all_train_data.head(1)
data = all_train_data.values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)
all_train_data[list(all_train_data.columns.values)] = data
from sklearn.model_selection import train_test_split
train_data, test_data, train_label, test_label = train_test_split(all_train_data, label_data, test_size=0.1, stratify=label_data)
with strategy.scope():
    model = get_model(193, n_classes)
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"])
    model.summary()
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)

for train_idx, val_idx in kf.split(range(train_data.shape[0])):
    train_ds = tf.data.Dataset.from_tensor_slices(({"feature":train_data.values[train_idx].reshape(-1,193,)}, train_label[train_idx]))
    train_ds = train_ds.shuffle(40000).batch(1024) 
    for data,label in train_ds.take(1):
        print(data["feature"].shape, label.shape)
    
    val_ds = tf.data.Dataset.from_tensor_slices(({"feature":train_data.values[val_idx].reshape(-1,193,)}, train_label[val_idx]))
    val_ds = val_ds.batch(1024) 
    for data,label in val_ds.take(1):
        print(data["feature"].shape, label.shape)
    model_history = model.fit(train_ds, epochs = 1, verbose = 1, validation_data=val_ds)
model.evaluate(test_data.values, test_label)
### Create predictions
def load_test_clip(path, start_time, duration=5):
    #if os.path.exists(TEST_FOLDER):
    return librosa.load(path, offset=start_time, duration=duration)
    #else:
    #    path = base_dir + "train_audio/aldfly/XC134874.mp3"
    #    return librosa.load(path, offset=start_time, duration=duration)
    
def make_prediction(block, sr):
    split_file_data = pd.DataFrame({"X":[block]})
    split_file_data["features"] = split_file_data["X"].map(lambda x: parse_audio_files2(x, sr))
    test_feature_df = split_file_data[["features"]]
    for i in range(193):
        test_feature_df["features_" + str(i)] = test_feature_df["features"].map(lambda x: x[i])
    test_features = test_feature_df.pop("features")
    test_feature_data = scaler.transform(test_feature_df.values)
    return list((model.predict(test_feature_data)>0.5).astype(int))[0]

TEST_FOLDER = '../input/birdsong-recognition/test_audio/'
test_info = pd.read_csv('../input/birdsong-recognition/test.csv')
test_info.head()

try:
    preds = []
    for index, row in test_info.iterrows():
        # Get test row information
        site = row['site']
        start_time = row['seconds'] - 5
        row_id = row['row_id']
        audio_id = row['audio_id']

        # Get the test sound clip
        if site == 'site_1' or site == 'site_2':
            sound_clip, sr = load_test_clip(TEST_FOLDER + audio_id + '.mp3', start_time)
        else:
            sound_clip, sr = load_test_clip(TEST_FOLDER + audio_id + '.mp3', 0, duration=None)

        # Make the prediction
        pred = make_prediction(sound_clip, sr)

        # Store prediction
        preds.append([row_id, pred])

    preds = pd.DataFrame(preds, columns=['row_id', 'pred'])
    preds["pred2"] = preds["pred"].map(lambda x: [i for i in range(x.shape[0]) if x[i]>0])
    preds["birds"] = preds["pred2"].map(lambda x: " ".join(list(np.sort([dic_class_name_rev[i] for i in x]))))
    preds["birds"] = preds["birds"].map(lambda x: "nocall" if x=="" else x)


    preds[["row_id","birds"]].to_csv('submission.csv', index=False)
except:
    preds = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')
    preds[["row_id","birds"]].to_csv('submission.csv', index=False)

if 1==2:
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(data_broken[:,0], data_broken[:,1], data_broken[:,2],lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")


    np.fft.fft(data_broken).real
