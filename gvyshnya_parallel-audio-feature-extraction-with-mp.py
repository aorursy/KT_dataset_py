import multiprocessing as mp
import datetime as dt
import pandas as pd
import numpy as np
import librosa
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

# config settings
in_kaggle = True
NUMBER_OF_MFCC = 20

NUMBER_OF_CPU_IN_POOL = 6

# Fourier Transform Settings
# Default FFT window size
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # number audio of frames between STFT columns (looks like a good default)

# data output settings
TRANSFORMED_DATA_PATH = "/kaggle/working"
def get_data_file_path(is_in_kaggle: bool) -> Tuple[str, str]:
    train_path = ''
    test_path = ''

    if is_in_kaggle:
        # running in Kaggle, inside the competition
        train_path = '../input/birdsong-recognition/train.csv'
        test_path = '../input/birdsong-recognition/test.csv'
    else:
        # running locally
        train_path = 'data/train.csv'
        test_path = 'data/test.csv'

    return train_path, test_path


def get_base_train_audio_folder_path(is_in_kaggle: bool) -> str:
    folder_path = ''
    if is_in_kaggle:
        folder_path = '../input/birdsong-recognition/train_audio/'
    else:
        folder_path = 'data/train_audio/'
    return folder_path


def extract_feautres(trial_audio_file_path: str):
    # process data frame
    function_start_time = dt.datetime.now()
    print("Started a file processing at ", function_start_time)

    df0 = extract_feature_means(trial_audio_file_path)

    function_finish_time = dt.datetime.now()
    print("Fininished the file processing at ", function_finish_time)

    processing = function_finish_time - function_start_time
    print("Processed the file: ", trial_audio_file_path, "; processing time: ", processing)

    return df0


# inspirations: https://musicinformationretrieval.com/basic_feature_extraction.html
def extract_feature_means(audio_file_path: str) -> pd.DataFrame:
    # config settings
    number_of_mfcc = NUMBER_OF_MFCC

    # 1. Importing 1 file
    y, sr = librosa.load(audio_file_path)

    # Trim leading and trailing silence from an audio signal (silence before and after the actual audio)
    signal, _ = librosa.effects.trim(y)

    # 2. Fourier Transform
    # Default FFT window size
    n_fft = N_FFT  # FFT window size
    hop_length = HOP_LENGTH  # number audio of frames between STFT columns (looks like a good default)

    # Short-time Fourier transform (STFT)
    d_audio = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))

    # 3. Spectrogram
    # Convert an amplitude spectrogram to Decibels-scaled spectrogram.
    db_audio = librosa.amplitude_to_db(d_audio, ref=np.max)

    # 4. Create the Mel Spectrograms
    s_audio = librosa.feature.melspectrogram(signal, sr=sr)
    s_db_audio = librosa.amplitude_to_db(s_audio, ref=np.max)

    # 5 Zero crossings

    # #6. Harmonics and Perceptrual
    # Note:
    #
    # Harmonics are characteristichs that represent the sound color
    # Perceptrual shock wave represents the sound rhythm and emotion
    y_harm, y_perc = librosa.effects.hpss(signal)

    # 7. Spectral Centroid
    # Note: Indicates where the ”centre of mass” for a sound is located and is calculated
    # as the weighted mean of the frequencies present in the sound.

    # Calculate the Spectral Centroids
    spectral_centroids = librosa.feature.spectral_centroid(signal, sr=sr)[0]
    spectral_centroids_delta = librosa.feature.delta(spectral_centroids)
    spectral_centroids_accelerate = librosa.feature.delta(spectral_centroids, order=2)

    # spectral_centroid_feats = np.stack((spectral_centroids, delta, accelerate))  # (3, 64, xx)

    # 8. Chroma Frequencies¶
    # Note: Chroma features are an interesting and powerful representation
    # for music audio in which the entire spectrum is projected onto 12 bins
    # representing the 12 distinct semitones ( or chromas) of the musical octave.

    # Increase or decrease hop_length to change how granular you want your data to be
    hop_length = HOP_LENGTH

    # Chromogram
    chromagram = librosa.feature.chroma_stft(signal, sr=sr, hop_length=hop_length)

    # 9. Tempo BPM (beats per minute)¶
    # Note: Dynamic programming beat tracker.

    # Create Tempo BPM variable
    tempo_y, _ = librosa.beat.beat_track(signal, sr=sr)

    # 10. Spectral Rolloff
    # Note: Is a measure of the shape of the signal. It represents the frequency below which a specified
    #  percentage of the total spectral energy(e.g. 85 %) lies.

    # Spectral RollOff Vector
    spectral_rolloff = librosa.feature.spectral_rolloff(signal, sr=sr)[0]

    # spectral flux
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)

    # Spectral Bandwidth¶
    # The spectral bandwidth is defined as the width of the band of light at one-half the peak
    # maximum (or full width at half maximum [FWHM]) and is represented by the two vertical
    # red lines and λSB on the wavelength axis.
    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(signal, sr=sr)[0]
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(signal, sr=sr, p=3)[0]
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(signal, sr=sr, p=4)[0]

    audio_features = {
        "file_name": audio_file_path,
        "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(signal)[0]),
        "zero_crossings": np.sum(librosa.zero_crossings(signal, pad=False)),
        "spectrogram": np.mean(db_audio[0]),
        "mel_spectrogram": np.mean(s_db_audio[0]),
        "harmonics": np.mean(y_harm),
        "perceptual_shock_wave": np.mean(y_perc),
        "spectral_centroids": np.mean(spectral_centroids),
        "spectral_centroids_delta": np.mean(spectral_centroids_delta),
        "spectral_centroids_accelerate": np.mean(spectral_centroids_accelerate),
        "chroma1": np.mean(chromagram[0]),
        "chroma2": np.mean(chromagram[1]),
        "chroma3": np.mean(chromagram[2]),
        "chroma4": np.mean(chromagram[3]),
        "chroma5": np.mean(chromagram[4]),
        "chroma6": np.mean(chromagram[5]),
        "chroma7": np.mean(chromagram[6]),
        "chroma8": np.mean(chromagram[7]),
        "chroma9": np.mean(chromagram[8]),
        "chroma10": np.mean(chromagram[9]),
        "chroma11": np.mean(chromagram[10]),
        "chroma12": np.mean(chromagram[11]),
        "tempo_bpm": tempo_y,
        "spectral_rolloff": np.mean(spectral_rolloff),
        "spectral_flux": np.mean(onset_env),
        "spectral_bandwidth_2": np.mean(spectral_bandwidth_2),
        "spectral_bandwidth_3": np.mean(spectral_bandwidth_3),
        "spectral_bandwidth_4": np.mean(spectral_bandwidth_4),
    }

    # extract mfcc feature
    mfcc_df = extract_mfcc_feature_means(audio_file_path,
                                    signal,
                                    sample_rate=sr,
                                    number_of_mfcc=number_of_mfcc)

    df = pd.DataFrame.from_records(data=[audio_features])

    df = pd.merge(df, mfcc_df, on='file_name')

    return df


def extract_mfcc_feature_means(audio_file_name: str,
                          signal: np.ndarray,
                          sample_rate: int,
                          number_of_mfcc: int) -> pd.DataFrame:
    # another MFCC approach
    # as suggested by https://github.com/Cocoxili/DCASE2018Task2/blob/master/data_transform.py,
    # https://arxiv.org/abs/1810.12832, and https://www.kaggle.com/c/freesound-audio-tagging
    mfcc_alt = librosa.feature.mfcc(y=signal, sr=sample_rate,
                                    n_mfcc=number_of_mfcc)
    delta = librosa.feature.delta(mfcc_alt)
    accelerate = librosa.feature.delta(mfcc_alt, order=2)

    mfcc_features = {
        "file_name": audio_file_name,
    }

    for i in range(0, number_of_mfcc):
        # dict.update({'key3': 'geeks'})

        # mfcc coefficient
        key_name = "".join(['mfcc', str(i)])
        mfcc_value = np.mean(mfcc_alt[i])
        mfcc_features.update({key_name: mfcc_value})

        # mfcc delta coefficient
        key_name = "".join(['mfcc_delta_', str(i)])
        mfcc_value = np.mean(delta[i])
        mfcc_features.update({key_name: mfcc_value})

        # mfcc accelerate coefficient
        key_name = "".join(['mfcc_accelerate_', str(i)])
        mfcc_value = np.mean(accelerate[i])
        mfcc_features.update({key_name: mfcc_value})

    df = pd.DataFrame.from_records(data=[mfcc_features])
    return df
start_time = dt.datetime.now()
print("Started at ", start_time)

# Import data
train_set_path, test_set_path = get_data_file_path(in_kaggle)
train_csv = pd.read_csv(train_set_path)
test_csv = pd.read_csv(test_set_path)

# Create some time features
train_csv['year'] = train_csv['date'].apply(lambda x: x.split('-')[0])
train_csv['month'] = train_csv['date'].apply(lambda x: x.split('-')[1])
train_csv['day_of_month'] = train_csv['date'].apply(lambda x: x.split('-')[2])


# Create Full Path so we can access data more easily
base_dir = get_base_train_audio_folder_path(in_kaggle)
train_csv['full_path'] = base_dir + train_csv['ebird_code'] + '/' + train_csv['filename']

final_data = ['American Avocet', 'American Bittern', 'American Crow',]

for ebird in final_data:
    print("Starting to process a new species: ", ebird)
    ebird_data = train_csv[train_csv['species'] == ebird]

    short_file_name = ebird_data['ebird_code'].unique()[0]
    print("Short file name: ", short_file_name)

    pool = mp.Pool(NUMBER_OF_CPU_IN_POOL)  # use the number of parallel processes as per the configured

    funclist = []

    for index, row in ebird_data.iterrows():
            # process each audio file
            f = pool.apply_async(extract_feautres, [row['full_path']])
            funclist.append(f)

    result = []
    for f in funclist:
        result.append(f.get(timeout=600))  # timeout in 600 seconds = 10 mins

    # combine chunks with transformed data into a single training set
    extracted_features = pd.concat(result)

    # save extracted features to CSV
    output_path = "".join([TRANSFORMED_DATA_PATH, short_file_name, ".csv"])
    extracted_features.to_csv(output_path, index=False)

    # clean up
    pool.close()
    pool.join()

    print("Finished processing: ", ebird)

print('We are done. That is all, folks!')
finish_time = dt.datetime.now()
print("Finished at ", finish_time)
elapsed = finish_time - start_time
print("Elapsed time: ", elapsed)