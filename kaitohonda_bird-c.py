import sklearn

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.image as mpimg

from matplotlib.offsetbox import AnnotationBbox, OffsetImage



# Map 1 library

import plotly.express as px



import descartes

import geopandas as gpd



#Librossa Libraries

import librosa

import librosa.display

import IPython.display as ipd



train = pd.read_csv("../input/birdsong-recognition/train.csv")

test = pd.read_csv("../input/birdsong-recognition/test.csv")



#Create some time features

train['year'] = train['date'].apply(lambda x: x.split('-')[0])#2013

train['month'] = train['date'].apply(lambda x: x.split('-')[1])#5

train['day_of_month'] = train['date'].apply(lambda x: x.split('-')[2])#25



print("There are {:,} unique bird species in the dataset.".format(len(train['species'].unique())))#unique()で重複を避けて、要素の個数を返す

ipd
train.info()
train.head(20)
print(len(train['type'].unique()))
train['type'].unique()
print(train.shape)
test.info()
bird = mpimg.imread('../input/bird-samples/orange bird.jpg')

imagebox = OffsetImage(bird, zoom=0.5)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(6.5, 2000))



plt.figure(figsize=(16, 6))

ax = sns.countplot(train['year'], palette="hls")#ヒストグラムを作成

ax.add_artist(ab)



plt.title("Year of the Audio Files Registration", fontsize=16)

plt.xticks(rotation=90, fontsize=13)

plt.yticks(fontsize=13)

plt.ylabel("Frequency", fontsize=14)

plt.xlabel("");
bird = mpimg.imread("../input/birdsamples/wasi.jpeg")

imagebox = OffsetImage(bird,zoom=0.7)

xy = (0.5,0.7)

ab = AnnotationBbox(imagebox,xy,frameon = False,pad=1,xybox = (11,3000))



plt.figure(figsize=(16,6))

ax = sns.countplot(train['month'], palette="hls")

ax.add_artist(ab)



plt.title("Month of the Audio Files Registration", fontsize=16)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.ylabel("Frequency", fontsize=14)

plt.xlabel("");
#音程（ピッチ）の頻度を表示

bird = mpimg.imread("../input/birdsamples2/white_bird.jpg")

imagebox = OffsetImage(bird,zoom = 0.2)

xy = (0.5,0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(3.9,8600))



plt.figure(figsize=(16,6))

ax = sns.countplot(train['pitch'],palette='hls',order = train['pitch'].value_counts().index)

ax.add_artist(ab)



plt.title("Pitch(quality of sound - how high/low was the tone)",fontsize=16)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.ylabel("Frecuency",fontsize=18)

plt.xlabel("")
train.columns
adjusted_type = train['type'].apply(lambda x : x.split(',')).reset_index().explode('type')#reste_indexインデックスを振り直す

adjusted_type = adjusted_type['type'].apply(lambda x : x.strip().lower()).reset_index()#stirp()空白文字・指定した文字を削除 / lower()すべての文字を小文字に変換

adjusted_type['type'] = adjusted_type['type'].replace('calls', 'call')
top_15 = list(adjusted_type['type'].value_counts().head(15).reset_index()['index'])

data = adjusted_type[adjusted_type['type'].isin(top_15)] #isin()は列の要素が引数に渡したリストの要素に含まれているかを選び抽出


#表示

bird = mpimg.imread('../input/sample3/blue_bird.jpg')

imagebox = OffsetImage(bird, zoom=0.43)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(12.4, 5700))



plt.figure(figsize=(16, 6))

ax = sns.countplot(data['type'], palette="hls", order = data['type'].value_counts().index)

ax.add_artist(ab)



plt.title("Top 15 Song Types", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
top_15 = list(train['elevation'].value_counts().head(15).reset_index()['index'])

data = train[train['elevation'].isin(top_15)]
#plot

bird = mpimg.imread('../input/sample3/blue_bird.jpg')

imagebox = OffsetImage(bird, zoom=0.43)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(12.4, 1450))



plt.figure(figsize=(16, 6))

ax = sns.countplot(data['elevation'], palette="hls", order = data['elevation'].value_counts().index)

ax.add_artist(ab)



plt.title("Top 15 Elevation Types", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
train[train.bird_seen == "no"].shape
#Cretate data

data = train['bird_seen'].value_counts().reset_index()

data
#Plot

bird = mpimg.imread('../input/sample4/cute_o_bird.jpg')

imagebox = OffsetImage(bird, zoom=0.22)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(15300, 0.95))



plt.figure(figsize=(16,6))

ax = sns.barplot(x = 'bird_seen' ,y = 'index' ,data = data ,palette="hls")

ax.add_artist(ab)

plt.title("Song was heard,but was Bird Seen?",fontsize=16)

plt.xlabel("")

plt.ylabel("Frecuency",fontsize=16)

plt.xticks( rotation= 45,fontsize=16)

plt.yticks()
train.country
#top 15 most common elevations

top_15 = list(train['country'].value_counts().head(15).reset_index()['index'])

data = train[train['country'].isin(top_15)]
#PLOT

bird = mpimg.imread("../input/example5/chick.jpg")

imagebox = OffsetImage(bird, zoom=0.3)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(12.2, 7000))



#plot

plt.figure(figsize=(16, 6))

ax = sns.countplot(data['country'], palette='hls', order = data['country'].value_counts().index)

ax.add_artist(ab)



plt.title("Top 15 Countries with most Recordings", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)
# Import gapminde data, where we hace counrty and iso ALPHA cpdes

#df = px.data.gapminder().query("year==2007")[["country","iso_alpha"]]

# Merge table together

#data = pd.merge(left= train , right=df,how = 'inner',on="country")

# Group by county adn count how many spicies can be found in each

#data = data.groupby(by=["country","iso_alpha"]).count()["species"].reset_index()

#fig = px.choropleth( data , locations="iso_alpha", color = "species" , hover_name = 'country',#

                    #color_continuous_scale = px.colors.sequential.Teal,title = "World Map:Recordings per Country")

#fig.show()
train.duration.describe()
#データの準備 

train["duration_interval"] = ">500"

train.loc[train["duration"] <= 100,'duration_intercal'] = "<= 100"

train.loc[(train['duration'] > 100) & (train['duration'] <= 200),'duration_interval'] ='100~200'

train.loc[(train['duration'] > 200) & (train['duration'] <= 300),'duration_interval'] ='200~300'

train.loc[(train['duration'] > 300) & (train['duration'] <= 400),'duration_interval'] ='300~400'

train.loc[(train['duration'] > 400) & (train['duration'] <= 500),'duration_interval'] ='400~500'



#データを読み込み

bird = mpimg.imread('../input/sample5/samplebird.jpg')

imagebox = OffsetImage(bird, zoom=0.2)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(3, 10000))

#図を表示

plt.figure(figsize=(16,6))

ax = sns.countplot(train['duration_interval'],palette ='hls')

ax.add_artist(ab)



# 図を調整

plt.title("Distribution of Recordings Duration",fontsize=16)

plt.ylabel("Distribution of Recording Duration")

plt.xlabel("Freucuency")

plt.yticks(fontsize=13)

plt.xticks(rotation=45,fontsize=13)
train["file_type"].unique()
bird = mpimg.imread('../input/sample6/superbird.jpg')

imagebox = OffsetImage(bird, zoom=0.1)

xy = (0.5, 0.5)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(2.7, 7000))



plt.figure(figsize=(16,6))

ax = sns.countplot(train['file_type'],palette = 'hls', order = train["file_type"].value_counts().index)

ax.add_artist(ab)



plt.title("Recording File Types", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
base_dir = "../input/birdsong-recognition/train_audio/"



train["full_path"] =  base_dir + train['ebird_code'] + '/' + train['filename'] #音声ファイルのパスに繋がるカラムを作成



# 音声のサンプルを抽出する

amered = train[train['ebird_code'] == "amered"].sample(1,random_state = 33)['full_path'].values[0]#sample ランダムで１つ抽出

cangoo = train[train['ebird_code'] == "cangoo"].sample(1,random_state = 33)['full_path'].values[0]

haiwoo = train[train['ebird_code'] == "haiwoo"].sample(1,random_state = 33)['full_path'].values[0]

pingro = train[train['ebird_code'] == "pingro"].sample(1,random_state = 33)['full_path'].values[0]

vesspa = train[train['ebird_code'] == "vesspa"].sample(1,random_state = 33)['full_path'].values[0]





bird_sample_list = ["amered","cangoo","haiwoo","pingro","vesspa"]
ipd.Audio(amered)
ipd.Audio(cangoo)
ipd.Audio(haiwoo)
ipd.Audio(pingro)
ipd.Audio(vesspa)
# Imprting 1 file

y,sr = librosa.load(vesspa)



print('y:',y)

print('y shape:',np.shape(y))

print('Sample Rate(KHz):',sr)



#Vertify length of the audio

print("Check Len of Audio:",661794/sr)
#音を整形する

# yの音の無音部分を切り取る（trim) 　　　　　　　※　_ その変数を使ってませんという表示

audio_file,_ = librosa.effects.trim(y)



#結果を出す

print('Audio File:', audio_file )

print('Audio File shape:',np.shape(audio_file))
# Importing the 5 files

y_amered,sr_amered = librosa.load(amered)

audio_amered,_  = librosa.effects.trim(y_amered)



y_cangoo,sr_cangoo = librosa.load(cangoo)

audio_cangoo,_  = librosa.effects.trim(y_cangoo)



y_haiwoo,sr_haiwoo = librosa.load(haiwoo)

audio_haiwoo,_  = librosa.effects.trim(y_haiwoo)



y_pingro,sr_pingro = librosa.load(pingro)

audio_pingro,_  = librosa.effects.trim(y_pingro)



y_vesspa,sr_vesspa = librosa.load(vesspa)

audio_vesspa,_  = librosa.effects.trim(y_vesspa)
fig,ax = plt.subplots(5,figsize = (16,9))

fig.suptitle("Sound Waves",fontsize = 16)



librosa.display.waveplot( y = audio_amered, sr = sr_amered, color = "#A300F9", ax = ax[0])

librosa.display.waveplot( y = audio_cangoo, sr = sr_cangoo, color = "#4300FF", ax = ax[1])

librosa.display.waveplot( y = audio_haiwoo, sr = sr_haiwoo, color = "#009DFF", ax = ax[2])

librosa.display.waveplot( y = audio_pingro, sr = sr_pingro, color = "#00FFB0", ax = ax[3])

librosa.display.waveplot( y = audio_vesspa, sr = sr_vesspa, color = "#D9FF00", ax = ax[4])



for i, name in zip(range(5),bird_sample_list):

    ax[i].set_ylabel(name, fontsize =13)
# FFT(高速フーリエ変換)の画面サイズ FFT 技術を使うと、時間信号をリアルタイムに周波数分析が可能となる。

# STFT(短時間フーリエ変換)は、統計的特性が時間により変化する非定常信号の分析に使われる信号処理手法。

# オーディオ CD 上のデータは、複数のフレームに分割されています。

n_fft = 2048 # フレームのサイズ

hop_length = 512 #audio_frameの数



#Short_time Fourier transfrom(STFT) フ-リエ変換！！

D_amered = np.abs(librosa.stft(audio_amered , n_fft = n_fft, hop_length = hop_length))

D_cangoo = np.abs(librosa.stft(audio_cangoo , n_fft = n_fft, hop_length = hop_length))

D_haiwoo = np.abs(librosa.stft(audio_haiwoo , n_fft = n_fft, hop_length = hop_length))

D_pingro = np.abs(librosa.stft(audio_pingro , n_fft = n_fft, hop_length = hop_length))

D_vesspa = np.abs(librosa.stft(audio_vesspa , n_fft = n_fft, hop_length = hop_length))
print("Shape of D object:", np.shape(D_amered))
# スペクトラムの振幅をデシベルのスケールに調整する。

DB_amered = librosa.amplitude_to_db(D_amered, ref = np.max)

DB_cangoo = librosa.amplitude_to_db(D_cangoo, ref = np.max)

DB_haiwoo = librosa.amplitude_to_db(D_haiwoo, ref = np.max)

DB_pingro = librosa.amplitude_to_db(D_pingro, ref = np.max)

DB_vesspa = librosa.amplitude_to_db(D_vesspa, ref = np.max)



# PLOT

fig,ax = plt.subplots(2,3,figsize=(16,9))

fig.suptitle("Spectrogram", fontsize = 16)

fig.delaxes(ax[1,2])# del + axes  axesを取り除く



librosa.display.specshow(DB_amered, sr = sr_amered, hop_length = hop_length, x_axis = 'time', y_axis = 'log', cmap = 'cool',ax = ax[0,0])

librosa.display.specshow(DB_cangoo, sr = sr_cangoo, hop_length = hop_length, x_axis = 'time', y_axis = 'log', cmap = 'cool',ax = ax[0,1])

librosa.display.specshow(DB_haiwoo, sr = sr_haiwoo, hop_length = hop_length, x_axis = 'time', y_axis = 'log', cmap = 'cool',ax = ax[0,2])

librosa.display.specshow(DB_pingro, sr = sr_pingro, hop_length = hop_length, x_axis = 'time', y_axis = 'log', cmap = 'cool',ax = ax[1,0])

librosa.display.specshow(DB_vesspa, sr = sr_vesspa, hop_length = hop_length, x_axis = 'time', y_axis = 'log', cmap = 'cool',ax = ax[1,1])



for i, name in zip(range(0,2*3), bird_sample_list):

    x = i // 3

    y = i % 3

    ax[x,y].set_title(name,fontsize=13)

bird_sample_list
# Create the Mel Spectograms

S_amered = librosa.feature.melspectrogram(y_amered, sr = sr_amered)

S_DB_amered = librosa.amplitude_to_db(S_amered, ref = np.max)



S_cangoo = librosa.feature.melspectrogram(y_cangoo, sr = sr_cangoo)

S_DB_cangoo= librosa.amplitude_to_db(S_cangoo, ref = np.max)



S_haiwoo = librosa.feature.melspectrogram(y_haiwoo, sr = sr_haiwoo)

S_DB_haiwoo = librosa.amplitude_to_db(S_haiwoo, ref = np.max)



S_pingro = librosa.feature.melspectrogram(y_pingro, sr = sr_pingro)

S_DB_pingro = librosa.amplitude_to_db(S_pingro, ref = np.max)



S_vesspa = librosa.feature.melspectrogram(y_vesspa, sr = sr_vesspa)

S_DB_vesspa = librosa.amplitude_to_db(S_amered, ref = np.max)



# PLOT

fig ,ax = plt.subplots(2,3,figsize = (16,9))

fig.suptitle("Mel Spectogram",fontsize = 16 )

fig.delaxes(ax[1,2])



librosa.display.specshow(S_DB_amered , sr = sr_amered , hop_length = hop_length , x_axis = 'time', y_axis = 'log' , cmap = 'rainbow',ax = ax[0,0])

librosa.display.specshow(S_DB_cangoo , sr = sr_cangoo , hop_length = hop_length , x_axis = 'time', y_axis = 'log' , cmap = 'rainbow',ax = ax[0,1])

librosa.display.specshow(S_DB_haiwoo , sr = sr_haiwoo , hop_length = hop_length , x_axis = 'time', y_axis = 'log' , cmap = 'rainbow',ax = ax[0,2])

librosa.display.specshow(S_DB_pingro , sr = sr_pingro , hop_length = hop_length , x_axis = 'time', y_axis = 'log' , cmap = 'rainbow',ax = ax[1,0])

librosa.display.specshow(S_DB_vesspa , sr = sr_vesspa , hop_length = hop_length , x_axis = 'time', y_axis = 'log' , cmap = 'rainbow',ax = ax[1,1])





# titleをつける

for i,name in zip(range(0,2*3), bird_sample_list):

    x = i // 3

    y = i % 3

    ax[x,y].set_title(name,fontsize=13)

# + α　Zero crossing Rate　を実装(参考)

def zcr(data):

    count = 0

    for i in range(len(data)-1):

        if fata[i]*data[i+1] < 0:

            count += 1

    zcr = count/(len(data))

    return zcr
#Total Zero_crossing in our 1song

zero_amered = librosa.zero_crossings(audio_amered, pad = False)

zero_cangoo = librosa.zero_crossings(audio_cangoo, pad = False)

zero_haiwoo = librosa.zero_crossings(audio_haiwoo, pad = False)

zero_pingro = librosa.zero_crossings(audio_pingro, pad = False)

zero_vesspa = librosa.zero_crossings(audio_vesspa, pad = False)



zero_birds_list = [zero_amered,zero_cangoo,zero_haiwoo,zero_pingro,zero_vesspa]



for bird,name in zip(zero_birds_list,bird_sample_list):

    print("{} change rate is {:,}".format(name,sum(bird)))
zero_amered
y_harm_haiwoo, y_perc_haiwoo = librosa.effects.hpss(audio_haiwoo)



plt.figure(figsize = (16,9))

plt.plot(y_perc_haiwoo,color="#FFB100")

plt.plot(y_harm_haiwoo,color = "#A300F9")

plt.legend(("Perceptrual","Harmonis"))

plt.title("Harmonics and Perceptrual : Haiwoo Bird",fontsize=16)
spectral_centroids = librosa.feature.spectral_centroid(audio_cangoo, sr = sr)[0]



print('Centroids:',spectral_centroids)

print("Shape of Spectral Centoids:",spectral_centroids.shape)



# 時間の変化を表す

frames = range(len(spectral_centroids))



#frame count を time(second) に変換する

t = librosa.frames_to_time(frames)



print("frames:",frames)

print("t:",t)



# 音のデータを標準化する関数

def normalize( x, axis=0):

    return sklearn.preprocessing.minmax_scale(x,axis = axis)
# spectral centroidの音波を図示する

plt.figure(figsize=(16,9))

librosa.display.waveplot(audio_cangoo, sr = sr ,alpha = 0.4, color = "#A300F9",lw = 3)

plt.plot(t,normalize(spectral_centroids),color ='#FFB100' , lw=2 )

plt.legend(["Spectral Centroid","Wave"])

plt.title("Spectral Centroid: Cangoo Bird",fontsize = 16)
hop_length = 5000



#chromagram vesspa

chromagram = librosa.feature.chroma_stft(audio_vesspa, sr = sr_vesspa, hop_length = hop_length)

print("Chromagram Vesspa shape:",chromagram.shape)



plt.figure(figsize=(16,6))

librosa.display.specshow(chromagram, x_axis = 'time', y_axis = "chroma" ,hop_length = hop_length ,cmap = 'twilight')

plt.title("Chromagram:Vesspa",fontsize=16)
chromagram
tempo_amered,_ = librosa.beat.beat_track(y_amered, sr = sr_amered)

tempo_cangoo,_ = librosa.beat.beat_track(y_cangoo, sr = sr_cangoo)

tempo_haiwoo,_ = librosa.beat.beat_track(y_haiwoo, sr = sr_haiwoo)

tempo_pingro,_ = librosa.beat.beat_track(y_pingro, sr = sr_pingro)

tempo_vesspa,_ = librosa.beat.beat_track(y_vesspa, sr = sr_vesspa)



data = pd.DataFrame({"Type":bird_sample_list,"BPM":[tempo_amered,tempo_cangoo,tempo_haiwoo,tempo_pingro,tempo_vesspa]})

data
bird = mpimg.imread("../input/sample8/two_bird.jpg")

imagebox = OffsetImage(bird,zoom = 0.2)

xy = (0.5,0.7)

ab = AnnotationBbox(imagebox , xy , frameon = False ,pad = 1 , xybox=(0.3,158))



plt.figure(figsize = (16,6))

ax = sns.barplot(y = data["BPM"] , x = data["Type"] , palette = "hls")

ax.add_artist(ab)



plt.ylabel("BPM",)

plt.yticks(fontsize=13)

plt.xticks(fontsize=13)

plt.xlabel("")

plt.title("BPM for 5 Different Bird Species",fontsize=16)

# Spectral RollOff Vector Rolooff

spectral_rolloff = librosa.feature.spectral_rolloff(audio_amered, sr=sr_amered)[0]



# Computing the time variable for visualization

frames = range(len(spectral_rolloff))

# Converts frame counts to time (seconds)

t = librosa.frames_to_time(frames)



# The plot

plt.figure(figsize = (16, 6))

librosa.display.waveplot(audio_amered, sr=sr_amered, alpha=0.4, color = '#A300F9', lw=3)

plt.plot(t, normalize(spectral_rolloff), color='#FFB100', lw=3)

plt.legend(["Spectral Rolloff", "Wave"])

plt.title("Spectral Rolloff: Amered Bird", fontsize=16);