import numpy as np

import pandas as pd

import seaborn as sns

import IPython

import matplotlib.pyplot as plt

import librosa

from nltk.corpus import stopwords 

import random

from scipy.io import wavfile

from tqdm import tqdm_notebook as tqdm

import tensorflow as tf

import os

from collections import Counter

from sklearn.preprocessing import OneHotEncoder

from tensorflow.python.keras.optimizer_v2.adam import Adam

from nltk.util import ngrams

from keras.callbacks import LearningRateScheduler

import nltk

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import glob

from PIL import Image
import_df = pd.read_csv('../input/medical-speech-with-spectrograms/Medical Speech, Transcription, and Intent/overview-of-recordings.csv')

import_df = import_df[['file_name','phrase','prompt','overall_quality_of_the_audio','speaker_id']]

print(import_df.shape)

import_df.head()
test_num = random.randrange(0, len(import_df))

test_file_name = import_df.loc[test_num, 'file_name']

print(import_df.loc[test_num, 'prompt'] + '\n' + import_df.loc[test_num, 'phrase'])

display_audio_file = f'../input/medical-speech-with-spectrograms/Medical Speech, Transcription, and Intent/recordings/test/{test_file_name}'

IPython.display.Audio(display_audio_file)
grouped_series = import_df.groupby('prompt').agg('count')['speaker_id'].sort_values(ascending=False)

unique_prompts = len(import_df['prompt'].unique())

print("Number of unique prompts : ", unique_prompts)

sns.barplot(grouped_series.values, grouped_series.index)

plt.title('Prompts to be Used as Classification Targets')

sns.despine()
preprocess_df = import_df.drop('overall_quality_of_the_audio', axis=1)



fig = plt.figure(figsize=(7,4))

sns.distplot(import_df['overall_quality_of_the_audio'], hist=False, color='teal')

sns.despine()
stop_words = set(stopwords.words('english')) 

word_dict = {}

preprocess_df['phrase'] = [i.lower() for i in preprocess_df['phrase']]

preprocess_df['phrase'] = [i.replace('can\'t', 'can not') for i in preprocess_df['phrase']]

preprocess_df['phrase'] = [i.replace('i\'m', 'i am') for i in preprocess_df['phrase']]

preprocess_df['phrase'] = [i.replace('i\'ve', 'i have') for i in preprocess_df['phrase']]

preprocess_df['phrase'] = [' '.join([j for j in i.split(' ') if j not in stop_words]) for i in preprocess_df['phrase']]



for phrase in preprocess_df['phrase']:

    for word in phrase.split(' '):

        word = word.lower()

        if word in stop_words or word == '':

            pass

        elif word not in word_dict:

            word_dict[word] = 1

        else:

            word_dict[word] += 1

            

sorted_word_list = sorted(word_dict.items(), key=lambda kv: kv[1], reverse=True)
n = 30

fig = plt.figure(figsize=(7,10))

plt.style.use('ggplot')

sns.barplot([i[1] for i in sorted_word_list[:n]], [i[0] for i in sorted_word_list[:n]])
def get_ngrams(text, n):

    n_grams = ngrams((text), n)

    return [' '.join(i) for i in n_grams]



def gramfreq(phrases, n, num):

    ngram_dict = {}

    for phrase in phrases:

        result = get_ngrams(phrase.split(' '),n)

        result_count = Counter(result)

        for gram in result_count.keys():

            if gram not in ngram_dict.keys():

                ngram_dict[gram] = 1

            else:

                ngram_dict[gram] += 1

    df = pd.DataFrame.from_dict(ngram_dict, orient='index')

    df = df.rename(columns={'index':'words', 0:'frequency'}) # Renaming index column name

    return df.sort_values(["frequency"],ascending=[0])[:num]



def gram_table(x, ns, result_length):

    output = pd.DataFrame(index=None)

    for n in ns:

        table = pd.DataFrame(gramfreq(x, n, result_length).reset_index())

        table.columns = [f"{n}-Gram",f"{n}-Occurence"]

        output = pd.concat([output, table], axis=1)

    return output



gram_df = gram_table(x=preprocess_df['phrase'], ns=[1,2,3,4], result_length=30)

gram_df.head(20)
fig = plt.figure(figsize=(10,5))

plt.title('Frequency of 2-grams, 3-grams and 4-grams', fontsize=15)



sns.distplot(gram_df['4-Occurence'], kde=False, label = '4-grams')

sns.distplot(gram_df['3-Occurence'], kde=False, label = '3-grams')

sns.distplot(gram_df['2-Occurence'], kde=False, label = '2-grams')



plt.ylabel('# of N-Grams')

plt.xlabel('# of Appearences')

plt.legend(facecolor='white')
base_dir = '../input/medical-speech-with-spectrograms/Medical Speech, Transcription, and Intent/recordings/'



train_files = [base_dir + 'train/' + i for i in os.listdir(base_dir + 'train')]

val_files = [base_dir + 'validate/' + i for i in os.listdir(base_dir + 'validate')]

test_files = [base_dir + 'test/' + i for i in os.listdir(base_dir + 'test')]



all_files = train_files + test_files + val_files

len(all_files)
tokenizer = Tokenizer(oov_token="<OOV>")

tokenizer.fit_on_texts(preprocess_df['phrase'])

word_index = tokenizer.word_index

vocab_size = len(word_index)

print(f'vocab_size : {vocab_size}')



phrases_seq = tokenizer.texts_to_sequences(preprocess_df['phrase'])

padded_phrases_seq = pad_sequences(phrases_seq, padding='post')

padded_phrases_seq = np.asarray(padded_phrases_seq)

max_seq_length = padded_phrases_seq.shape[0]

print("padded_phrases_seq shape : ", padded_phrases_seq.shape)
random_phrase_num = random.randrange(0, len(preprocess_df))

random_import_phrase = import_df.loc[random_phrase_num, 'phrase']

random_phrase = preprocess_df.loc[random_phrase_num, 'phrase']



print('padded_phrase example : ' + '\n' + random_import_phrase + '\n' + random_phrase + '\n' + str(padded_phrases_seq[random_phrase_num]))
#os.listdir('../input/x-wav-array/x_wav_array.npy')#/x_wav_array.npy')
wav_list = []

import librosa.display

from pathlib import Path

spec_dir = base_dir + 'spectrograms/'



!mkdir /kaggle/working/spectro





### Function from https://www.kaggle.com/devilsknight/sound-classification-using-spectrogram-images

### Originally linked by kaggle user _____

def create_spectrogram(filename,name):

    plt.interactive(False)

    clip, sample_rate = librosa.load(filename, sr=None)

    fig = plt.figure(figsize=[2,2])

    ax = fig.add_subplot(111)

    ax.axes.get_xaxis().set_visible(False)

    ax.axes.get_yaxis().set_visible(False)

    ax.set_frame_on(False)

    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)

    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    filename = Path("/kaggle/working/spectro/" + name + '.jpg')

    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)

    plt.close()    

    fig.clf()

    plt.close(fig)

    plt.close('all')

    del filename,name,clip,sample_rate,fig,ax,S



### Not going to run this, I loaded the files from local to minimize run time on Kaggle.

#for file in tqdm(all_files, total=len(all_files)):

#    create_spectrogram(file, file.split('/')[-1])

#spec_filelist = [f'../input/medical-speech-with-spectrograms/Medical Speech, Transcription, and Intent/recordings/spectrograms/{i}.jpg' for i in preprocess_df.file_name]

#x_wav_array = np.array([np.array(Image.open(fname)) for fname in spec_filelist])

x_wav_array = np.load('../input/x-wav-array/x_wav_array.npy')

print(x_wav_array.shape)
enc = OneHotEncoder(handle_unknown='ignore')

prompt_array = preprocess_df['prompt'].values.reshape(-1,1)

labels_onehot = enc.fit_transform(prompt_array).toarray()



labels_onehot.shape
x_train, x_test, y_train, y_test = train_test_split(preprocess_df.index, labels_onehot, test_size = .2)
x_phrase_train = padded_phrases_seq[x_train]

x_phrase_test = padded_phrases_seq[x_test]



x_wav_train = x_wav_array[x_train]

x_wav_test = x_wav_array[x_test]



try:

    del x_wav_array

except:

    pass



x_wav_train = np.stack(x_wav_train, axis=0)

x_wav_test = np.stack(x_wav_test, axis=0)



print(x_phrase_train.shape)

print(x_phrase_test.shape)



print(x_wav_train.shape)

print(x_wav_test.shape)
def build_phrase_model(vocab_size, embedding_dim, rnn_units, max_seq_length):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Embedding(vocab_size + 1, ### Without +1, layer expects [0,1160) and our onehot encoded values include 1160

                                        embedding_dim, ### Output layer size

                                        input_length =  14))

    model.add(tf.keras.layers.LSTM(rnn_units))

    model.add(tf.keras.layers.Dense(100, activation='relu'))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(unique_prompts, activation='softmax'))

    return model



model = build_phrase_model(

    vocab_size = vocab_size,

    embedding_dim=100,

    rnn_units=150,

    max_seq_length=max_seq_length)



adam_opt = Adam(lr=0.01)



model.compile(optimizer=adam_opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
vocab_size
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=2, min_delta=.005)





def exp_decay(epoch):

    initial_lrate = 0.01

    k = 0.1

    lrate = initial_lrate * np.exp(-k*epoch)

    return lrate

lrate = LearningRateScheduler(exp_decay)



callbacks_list = [earlystop_callback, lrate]



history = model.fit(x_phrase_train, y_train,

                    epochs=15, batch_size=30, validation_split = .2,

                    callbacks=callbacks_list)
model_cm = confusion_matrix(np.argmax(y_test,axis=1), np.argmax(model.predict(x_phrase_test),axis=1))



fig = plt.figure(figsize=(15,10))

sns.heatmap(model_cm, annot=True, xticklabels=enc.categories_[0].tolist(), yticklabels=enc.categories_[0].tolist())
from keras.constraints import max_norm

def build_wav_model(filters, input_shape):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters, 2, 2, activation='relu', padding="same", input_shape=input_shape, kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))

    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

    model.add(tf.keras.layers.Conv2D(int(filters / 2), 2, 2, activation='relu', padding="same"))

    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dropout(.2))

    #model.add(tf.keras.layers.Dense(256, activation='relu'))

    model.add(tf.keras.layers.Dense(256, activation='relu'))

    model.add(tf.keras.layers.Dense(unique_prompts, activation='softmax'))

    return model



wav_model = build_wav_model(

    filters = 32,

    input_shape = x_wav_train[0].shape)



adam_opt = Adam(lr=0.001)



wav_model.compile(optimizer=adam_opt, loss='categorical_crossentropy', metrics=['accuracy'])

wav_model.summary()
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, min_delta=.005)



callbacks_list = [earlystop_callback]



#history = wav_model.fit(x_wav_train, y_train,epochs=15, batch_size=20, validation_split = .2,callbacks=callbacks_list)
def alexnet(in_shape=x_wav_train[0].shape, n_classes=unique_prompts, opt='sgd'):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(in_shape))

    model.add(tf.keras.layers.Conv2D(96,11, strides=4, activation='relu'))

    model.add(tf.keras.layers.MaxPool2D(3, 2))

    model.add(tf.keras.layers.Conv2D(256,5, strides=1, padding='same', activation='relu'))

    model.add(tf.keras.layers.MaxPool2D(3, 2))

    model.add(tf.keras.layers.Conv2D(384, 3, strides=1, padding='same', activation='relu'))

    model.add(tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', activation='relu'))

    model.add(tf.keras.layers.MaxPool2D(3, 2))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(4096, activation='relu'))

    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(4096, activation='relu'))

    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))



    return model
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, min_delta=.005)



callbacks_list = [earlystop_callback]



alexnet_model = alexnet()



alexnet_model.compile(loss="categorical_crossentropy", optimizer='adam',

	              metrics=["accuracy"])



#history = alexnet_model.fit(x_wav_train, y_train,epochs=15, batch_size=20, validation_split = .2,callbacks=callbacks_list)