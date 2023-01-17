# **Step 1: Import Python Packages** 



# Fastai, Librosa, Spacy, Scispacy, PySound, Seaborn, etc
!pip install scispacy

!pip install pysoundfile

!apt-get install libav-tools -y

!apt-get install zip

!pip freeze > '../working/dockerimage_snapshot.txt'
from fastai.text import *

from fastai.vision import *

import spacy

from spacy import displacy

import scispacy

import librosa

import librosa.display

import soundfile as sf

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS

from collections import Counter

import IPython

import os

from glob import glob

from tqdm import tqdm

import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

import pylab

import gc

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
# **Step 3: Define Helper Functions**



# Create spectrograms and word frequency plots
def get_wav_info(wav_file):

    data, rate = sf.read(wav_file)

    return data, rate



def create_spectrogram(wav_file):

    # adapted from Andrew Ng Deep Learning Specialization Course 5

    data, rate = get_wav_info(wav_file)

    nfft = 200 # Length of each window segment

    fs = 8000 # Sampling frequencies

    noverlap = 120 # Overlap between windows

    nchannels = data.ndim

    if nchannels == 1:

        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)

    elif nchannels == 2:

        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)

    return pxx



def create_melspectrogram(filename,name):

    # adapted from https://www.kaggle.com/devilsknight/sound-classification-using-spectrogram-images

    plt.interactive(False)

    clip, sample_rate = librosa.load(filename, sr=None)

    fig = plt.figure(figsize=[0.72,0.72])

    ax = fig.add_subplot(111)

    ax.axes.get_xaxis().set_visible(False)

    ax.axes.get_yaxis().set_visible(False)

    ax.set_frame_on(False)

    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)

    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    filename  = Path('/kaggle/working/spectrograms/' + name + '.jpg')

    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)

    plt.close()    

    fig.clf()

    plt.close(fig)

    plt.close('all')

    del filename,name,clip,sample_rate,fig,ax,S



def wordBarGraphFunction(df,column,title):

    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    plt.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:50])])

    plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))

    plt.title(title)

    plt.show()



def wordCloudFunction(df,column,numWords):

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    word_string=str(popular_words_nonstop)

    wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white',

                          max_words=numWords,

                          width=1000,height=1000,

                         ).generate(word_string)

    plt.clf()

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
overview = pd.read_csv('../input/medical-speech-transcription-and-intent/medical speech transcription and intent/Medical Speech, Transcription, and Intent/overview-of-recordings.csv')

overview = overview[['file_name','phrase','prompt','overall_quality_of_the_audio','speaker_id']]

overview=overview.dropna()

overviewAudio = overview[['file_name','prompt']]

overviewAudio['spec_name'] = overviewAudio['file_name'].str.rstrip('.wav')

overviewAudio = overviewAudio[['spec_name','prompt']]

overviewText = overview[['phrase','prompt']]

noNaNcsv = '../input/medical-speech-transcription-and-intent/medical speech transcription and intent/Medical Speech, Transcription, and Intent/overview-of-recordings.csv'

noNaNcsv = pd.read_csv(noNaNcsv)

noNaNcsv = noNaNcsv.dropna()

noNaNcsv = noNaNcsv.to_csv('overview-of-recordings.csv',index=False)

noNaNcsv
overview[110:120]
sns.set_style("whitegrid")

promptsPlot = sns.countplot(y='prompt',data=overview)

promptsPlot



qualityPlot = sns.FacetGrid(overview,aspect=2.5)

qualityPlot.map(sns.kdeplot,'overall_quality_of_the_audio',shade= True)

qualityPlot.set(xlim=(2.5, overview['overall_quality_of_the_audio'].max()))

qualityPlot.set_axis_labels('overall_quality_of_the_audio', 'Proportion')

qualityPlot
overview[62:63]
en_core_sci_sm = '../input/scispacy-pretrained-models/scispacy pretrained models/Scispacy Pretrained Models/en_core_sci_sm-0.1.0/en_core_sci_sm/en_core_sci_sm-0.1.0'

nlp = spacy.load(en_core_sci_sm)

text = overview['phrase'][62]

doc = nlp(text)

print(list(doc.sents))

print(doc.ents)

displacy.render(next(doc.sents), style='dep', jupyter=True,options = {'compact': True, 'word_spacing': 45, 'distance': 90})
IPython.display.Audio('../input/medical-speech-transcription-and-intent/medical speech transcription and intent/Medical Speech, Transcription, and Intent/recordings/test/1249120_20518958_23074828.wav')
overview[118:119]
en_core_sci_sm = '../input/scispacy-pretrained-models/scispacy pretrained models/Scispacy Pretrained Models/en_core_sci_sm-0.1.0/en_core_sci_sm/en_core_sci_sm-0.1.0'

nlp = spacy.load(en_core_sci_sm)

text = overview['phrase'][118]

doc = nlp(text)

print(list(doc.sents))

print(doc.ents)

displacy.render(next(doc.sents), style='dep', jupyter=True,options = {'compact': True, 'word_spacing': 45, 'distance': 90})
IPython.display.Audio('../input/medical-speech-transcription-and-intent/medical speech transcription and intent/Medical Speech, Transcription, and Intent/recordings/test/1249120_43788827_53247832.wav')
plt.figure(figsize=(15,15))

wordCloudFunction(overview,'phrase',10000000)
plt.figure(figsize=(10,10))

wordBarGraphFunction(overview,'phrase',"Most Common Words in Medical Text Transcripts")
np.random.seed(7)

path = Path('../input/medical-speech-transcription-and-intent/medical speech transcription and intent/Medical Speech, Transcription, and Intent/')

data_clas = (TextList.from_csv(path, 'overview-of-recordings.csv', 

                               cols='phrase')

                   .random_split_by_pct(.2)

                   .label_from_df(cols='prompt')

                   .databunch(bs=42))

MODEL_PATH = "/tmp/model/"

learn = text_classifier_learner(data_clas,model_dir=MODEL_PATH,arch=AWD_LSTM)

learn.fit_one_cycle(5)
learn.unfreeze()

learn.fit_one_cycle(5)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(10,10), dpi=60)
testAudio = "../input/medical-speech-transcription-and-intent/medical speech transcription and intent/Medical Speech, Transcription, and Intent/recordings/train/1249120_44176037_58635902.wav"

x = create_spectrogram(testAudio)
filename = "../input/medical-speech-transcription-and-intent/medical speech transcription and intent/Medical Speech, Transcription, and Intent/recordings/train/1249120_44176037_58635902.wav"

clip, sample_rate = librosa.load(filename, sr=None)

fig = plt.figure(figsize=[5,5])

S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)

librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
!mkdir /kaggle/working/spectrograms



Data_dir_train=np.array(glob("../input/medical-speech-transcription-and-intent/medical speech transcription and intent/Medical Speech, Transcription, and Intent/recordings/train/*"))

Data_dir_test=np.array(glob("../input/medical-speech-transcription-and-intent/medical speech transcription and intent/Medical Speech, Transcription, and Intent/recordings/test/*"))

Data_dir_val=np.array(glob("../input/medical-speech-transcription-and-intent/medical speech transcription and intent/Medical Speech, Transcription, and Intent/recordings/validate/*"))



for file in tqdm(Data_dir_train):

    filename,name = file,file.split('/')[-1].split('.')[0]

    create_melspectrogram(filename,name)

for file in tqdm(Data_dir_test):

    filename,name = file,file.split('/')[-1].split('.')[0]

    create_melspectrogram(filename,name)

for file in tqdm(Data_dir_val):

    filename,name = file,file.split('/')[-1].split('.')[0]

    create_melspectrogram(filename,name)
path = Path('/kaggle/working/')

np.random.seed(7)

data = ImageDataBunch.from_df(path,df=overviewAudio, folder="spectrograms", valid_pct=0.2, suffix='.jpg',

        ds_tfms=get_transforms(), size=299, num_workers=0).normalize(imagenet_stats)

learn = create_cnn(data, models.resnet50, metrics=accuracy)

learn.fit_one_cycle(10)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(50)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(10,10), dpi=60)
!zip -r spectrograms.zip /kaggle/working/spectrograms/

!rm -rf spectrograms/*