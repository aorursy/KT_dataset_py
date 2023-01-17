import audioread
import librosa
import librosa.display
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
#import random as rnd
from matplotlib import pyplot as plt
import os

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

AUDIO_FOLDER = '.\Cats'


def remove_frames(cats, cats_frames, sr=16000):
    detect_cats = []

    for cat in cats:
        detect_cats.append(librosa.onset.onset_detect(y=cat[0]))

    new_cats_frames = []

    for k in range(len(cats_frames)):
        new_cats_frames.append([[cats_frames[k][i][j] for j in detect_cats[k]] for i in range(len(cats_frames[0]))])

    return new_cats_frames


def build_df_mfccs(X, y):
    df_cats = pd.DataFrame(np.transpose(X[0]), index=np.ones(len(np.transpose(X[0]))) * (0 + 1))
    for i in range(1, len(X)):
        df_temp = pd.DataFrame(np.transpose(X[i]), index=np.ones(len(np.transpose(X[i]))) * (i + 1))
        frames = [df_cats, df_temp]
        df_cats = pd.concat(frames)

    df_cats['feeling'] = pd.Series(y, index=df_cats.index)

    df_cats = df_cats.sample(frac=1)
    return df_cats


def compute_deltas(cats_mfccs):
    cats_deltas = []

    for i in range(len(cats_mfccs)):
        cats_deltas.append(librosa.feature.delta(cats_mfccs[i]))

    return cats_deltas


def compute_mfccs(cats_mel_frequencies, sr=16000):
    cats_mfccs = []

    for i in range(len(cats_mel_frequencies)):
        cats_mfccs.append(librosa.feature.mfcc(S=librosa.power_to_db(cats_mel_frequencies[i]), sr=sr))

    return cats_mfccs


def compute_mel_frequencies(cats, sr=16000):
    cats_mel_frequencies = []

    for i in range(len(cats)):
        cats_mel_frequencies.append(librosa.feature.melspectrogram(y=np.array(cats[i][0]), sr=sr))

    return cats_mel_frequencies


def load_audio_channels(cat_sound_file_names, sr=16000):

    cats = []
    channel_cats = []

    for i in range(len(cat_sound_file_names)):
        path = os.path.join(AUDIO_FOLDER, cat_sound_file_names[i])
        cats.append(librosa.load(path , sr=sr))
        with audioread.audio_open(path) as input_file:
            channel_cats.append(input_file.channels)

    channel_cats = list(set(channel_cats))
    print('Cats audio files loaded: ' + str(len(cats)) + ' samples')

    if len(channel_cats) == 1 and channel_cats[0] == 1:
        print("Only one channel for cats files")
    else:
        print("Attention ! Some audio cats files have several channels")

    return cats


def get_wav_from_dir(directory):
    result = []
    temp_list = os.listdir(directory)
    temp_list.sort()

    for f in temp_list:
        if f.endswith('.wav'):
            result.append(f)
            print('Found wav file: ' + f)
    return result


def display_wav(wav):
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(wav, sr=16000)
    plt.show()


def display_mfccs(mfccs):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


def display_mel_freq(mel):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), y_axis='mel', fmax=None, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()


def display_non_silent_frames_for(x):
    # Plot onset detection results of audio file cats_1
    detect = librosa.onset.onset_detect(y=x)
    onset_times = librosa.frames_to_time(detect)
    plt.plot(onset_times, np.zeros_like(onset_times) + 0, 'x')
    plt.show()
    print("Non silent frame indexes: " + str(detect))


def get_expected_feelings_from_csv():
    dataset = pd.read_csv('train_test_split.csv')
    return dataset.iloc[:, 2].values


def get_train_and_test_df(frames, test_size=0.3):
    y = get_expected_feelings_from_csv()
    X_train, X_test, y_train, y_test = train_test_split(frames, y, test_size=test_size, random_state=42)
    df_train = build_df_mfccs(X_train, y_train)
    df_test = build_df_mfccs(X_test, y_test)

    return df_train, df_test

def extract_sounds_features():

    sound_file_names = get_wav_from_dir(AUDIO_FOLDER)
    cats = load_audio_channels(sound_file_names)
    display_wav(cats[0][0])

    mel_frequencies = compute_mel_frequencies(cats)
    display_mel_freq(mel_frequencies[0])

    mfccs = compute_mfccs(mel_frequencies)
    display_mfccs(mfccs[0])

    # Computing of the delta features for each audio file
    deltas = compute_deltas(mfccs)

    # Removing silent frames for delta features and mfccs for each audio file
    nonsilent_mfccs = remove_frames(cats, mfccs)
    nonsilent_deltas = remove_frames(cats, deltas)

    display_non_silent_frames_for(cats[0][0])
    # Merging mfccs and delta features ?
    #features_cats = [np.vstack((nonsilent_mfccs[i], nonsilent_deltas[i])) for i in range(len(nonsilent_mfccs))]

    df_train, df_test = get_train_and_test_df(nonsilent_mfccs)

#Used to export a random result inot csv the first time for training/test
    # random for now#
    #cat_feelings = []
    #for _ in range(len(sound_file_names)):
    #    cat_feelings.append(rnd.randint(0, 3))

    ##TODO: add more data!
    #matrix = np.stack((sound_file_names, cat_feelings), -1)
    #df = pd.DataFrame(matrix, columns=['filename', 'feeling'])
    #df.to_csv('train_test_split.csv')




